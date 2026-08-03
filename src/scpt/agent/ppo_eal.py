"""PPO-EAL agent: pure loss functions + PPOEALTrainer.

Pure loss functions (lower half of this module) are the mathematical core
of the agent — isolated from any torch autograd concerns so they can be
independently tested.

PPOEALTrainer (upper half) wires together:
- env rollout (collect_rollout)
- GAE for reward + each constraint
- Clipped PPO surrogate + augmented-Lagrangian penalty
- Momentum dual updater for λ

Sign convention (the #1 source of PPO bugs):
- `clipped_ppo_surrogate` returns the STANDARD OpenAI maximize-convention
  value: min(ratio * adv, clip(ratio) * adv). Higher is better.
- `constraint_surrogate` returns phi_c_i in the form we want to keep <= 0
  for the constraint to be satisfied.
- `augmented_lagrangian_penalty` returns the AL penalty, higher means more
  violated.
- `total_loss` returns the scalar we MINIMIZE. It applies the negation to
  the reward surrogate exactly once here. Guarded by a dedicated test.

Critical: action masks are stored in the buffer at collection time and
*never* recomputed during the loss pass. Constraint GAE averages advantages
only over legally-masked actions per spec §5.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scpt.model.gnn_encoder import encode_design


def clipped_ppo_surrogate(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Standard PPO clipped surrogate. Maximize-convention.

    Args:
        ratio: pi(a|s) / pi_k(a|s) per step.
        advantages: A^(s,a) per step (reward advantages, not constraint).
        clip_eps: PPO clipping epsilon (typically 0.2).

    Returns:
        Scalar mean of min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv).
    """
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.minimum(unclipped, clipped).mean()


def constraint_surrogate(
    j_c_pi_k: torch.Tensor,
    constraint_advantages: torch.Tensor,
    budget: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Constraint violation estimate phi_c_i.

    phi_c_i = J_c_i^pi_k + (1/(1-gamma)) * E[A_c over LEGAL actions] - b_i

    The constraint surrogate is satisfied when phi_c_i <= 0. The augmented
    Lagrangian penalty in `total_loss` punishes positive phi_c_i.

    IMPORTANT: `constraint_advantages` must already be averaged over LEGAL
    actions only (caller masks out illegal actions before calling). If we
    averaged over all actions including the -inf-masked probability mass,
    phi_c would be pulled toward zero silently.
    """
    return j_c_pi_k + (1.0 / (1.0 - gamma)) * constraint_advantages - budget


def augmented_lagrangian_penalty(
    phi_c: torch.Tensor,
    lam: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Augmented-Lagrangian quadratic penalty.

    (sigma/2) * (max{0, lambda/sigma + phi_c}^2 - (lambda/sigma)^2)

    When phi_c < -lambda/sigma, penalty = -(lambda^2)/(2*sigma) (a constant
    that doesn't push lambda). When phi_c > 0 and lambda is zero, penalty
    grows quadratically with phi_c.
    """
    inner = torch.clamp(lam / sigma + phi_c, min=0.0)
    return (sigma / 2.0) * (inner.pow(2) - (lam / sigma).pow(2))


def total_loss(
    reward_surrogate: torch.Tensor,
    constraint_phis: dict[str, torch.Tensor],
    lambdas: dict[str, torch.Tensor],
    sigma: float,
) -> torch.Tensor:
    """Total PPO-EAL loss to MINIMIZE.

    L = -L_R^{pi_k}(pi) + sum_i (sigma/2) * (max{0, lambda_i/sigma + phi_c_i}^2
                                            - (lambda_i/sigma)^2)

    The negation of reward_surrogate is applied here, once. Do not negate
    `clipped_ppo_surrogate`'s output upstream — that's the most common
    source of double-negation bugs.
    """
    penalty = torch.stack([
        augmented_lagrangian_penalty(constraint_phis[k], lambdas[k], sigma)
        for k in constraint_phis
    ]).sum() if constraint_phis else torch.tensor(0.0)
    return -reward_surrogate + penalty


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Generalized Advantage Estimation.

    Args:
        rewards: (T,) per-step reward
        values: (T,) per-step V(s)
        dones: (T,) 1.0 if episode terminated, else 0.0
        next_value: V(s_{T+1})
        gamma, gae_lambda: discount factors

    Returns:
        advantages: (T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-batch advantage normalization (mean 0, std 1).

    Skipped when std < eps to avoid division-by-zero on degenerate batches.
    """
    if advantages.numel() < 2:
        return advantages
    std = advantages.std()
    if std < eps:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + eps)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one episode's worth of transitions for PPO update.

    Action masks are stored at collection time and never recomputed during
    the loss pass — this prevents the mask-staleness bug where the policy
    distribution shifts during the update and illegal actions become legal.
    """
    constraint_names: list[str]
    # Per-step fields (stored in collection order).
    obs_list: list[dict] = field(default_factory=list)   # raw obs dicts
    actions: list[int] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[dict[str, torch.Tensor]] = field(default_factory=list)
    costs: list[dict[str, float]] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    # Critically: masks stored at the moment of collection, not recomputed.
    action_masks: list[torch.Tensor] = field(default_factory=list)

    def add(
        self,
        obs: dict,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: dict[str, torch.Tensor],
        costs: dict[str, float],
        done: bool,
    ) -> None:
        self.obs_list.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append({k: v.detach() for k, v in value.items()})
        self.costs.append(costs)
        self.dones.append(done)
        # Store the mask from obs — this is the ONLY place masks are captured.
        self.action_masks.append(obs["action_mask"].detach().clone())

    def clear(self) -> None:
        self.obs_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.costs.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPOEALTrainer
# ---------------------------------------------------------------------------

class PPOEALTrainer:
    """PPO-EAL trainer: rollout collection + policy/value update.

    Wires together:
    - `collect_rollout(env, n_steps)` — run the policy for n_steps, storing
      transitions in the rollout buffer.
    - `update(env, n_steps)` — collect + compute GAEs + PPO-EAL loss +
      gradient step + dual update. Returns diagnostics.

    The trainer operates in a single-environment, on-policy setting. For
    multi-env parallelism, wrap the envs outside and call collect/update
    from the outer loop.

    Args:
        policy: SCPTPolicy instance.
        value_heads: ValueHeads instance.
        cfg: SimpleNamespace with fields:
            - d (int): hidden dim (must match policy/value_heads)
            - pair_dim (int): pair feature dim (must match policy)
            - clip_eps (float): PPO clip epsilon (e.g. 0.2)
            - gamma (float): discount factor
            - gae_lambda (float): GAE lambda
            - sigma (float): augmented-Lagrangian sigma
            - constraint_names (list[str]): constraint identifiers
            - constraint_budgets (dict[str, float]): per-constraint budget b_i
            - lr (float): learning rate
            - epochs (int): number of PPO epochs per update
            - minibatch_size (int): minibatch size for PPO epochs
            - dual_alpha (float): dual ascent step size
            - dual_ema_decay (float): EMA decay for phi_c smoothing
    """

    def __init__(self, policy: nn.Module, value_heads: nn.Module, cfg: Any, encoder: nn.Module | None = None, device: torch.device | None = None):
        self.policy = policy
        self.value_heads = value_heads
        self.cfg = cfg
        self.encoder = encoder
        self.device = device or next(policy.parameters()).device
        self.buffer = RolloutBuffer(constraint_names=list(cfg.constraint_names))

        # Lazy import to avoid circular deps.
        from scpt.agent.lagrangian import MomentumDualUpdater
        self.dual_updater = MomentumDualUpdater(
            constraint_names=list(cfg.constraint_names),
            alpha=cfg.dual_alpha,
            ema_decay=cfg.dual_ema_decay,
        )

        all_params = list(policy.parameters()) + list(value_heads.parameters())
        self.optimizer = optim.Adam(all_params, lr=cfg.lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_rollout(self, env, n_steps: int) -> None:
        """Run the policy for up to n_steps steps, storing to buffer.

        Resets the env at start if the buffer is empty. Continues an
        in-progress episode if the buffer already has data.
        """
        self.buffer.clear()
        obs, _ = env.reset()
        obs = self._prepare_obs(env, obs)
        done = False
        for _ in range(n_steps):
            if done:
                obs, _ = env.reset()
                obs = self._prepare_obs(env, obs)
                done = False

            action, log_prob = self._sample_action(obs)
            value = self._compute_value(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = self._prepare_obs(env, next_obs)
            done = terminated or truncated
            costs = info.get("costs", {})

            self.buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=float(reward),
                value=value,
                costs=costs,
                done=done,
            )
            obs = next_obs

    def update(self, env, n_steps: int) -> dict:
        """Collect rollout, run PPO-EAL update, return diagnostics."""
        self.collect_rollout(env, n_steps)
        return self._ppo_eal_update()

    # ------------------------------------------------------------------
    # Policy + value inference (no-grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_action(self, obs: dict) -> tuple[int, torch.Tensor]:
        """Sample an action from the policy. Returns (action_int, log_prob)."""
        # Handle both old format (precomputed tensors) and new format (reconstruction data)
        z_star = obs.get("z_star")
        Z_placed = obs.get("Z_placed")
        design_json = obs.get("design_json")
        active_idx = obs.get("active_idx")
        placed_indices = obs.get("placed_indices")
        F_pair = obs.get("F_pair")

        # If we have precomputed tensors (old format), use them directly
        if z_star is not None and torch.is_tensor(z_star) and Z_placed is not None and torch.is_tensor(Z_placed):
            pass  # Use the precomputed values
        # If we have reconstruction data (new format), compute tensors on demand
        elif design_json is not None and active_idx is not None and placed_indices is not None:
            design = json.loads(design_json)
            _, z_star, Z_placed = encode_design(design, self.encoder, active_idx, placed_indices)
        else:
            # Fallback to defaults
            z_star = torch.zeros(self.cfg.d, dtype=torch.float32, device=self.device)
            Z_placed = torch.zeros(0, self.cfg.d, dtype=torch.float32, device=self.device)

        F_pair = F_pair if F_pair is not None else torch.zeros(0, self.cfg.pair_dim, dtype=torch.float32, device=self.device)
        grid_xy = obs["grid_xy"]
        action_mask = obs["action_mask"]

        logits = self.policy(z_star, Z_placed, F_pair, grid_xy, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    @torch.no_grad()
    def _compute_value(
        self, obs: dict
    ) -> dict[str, torch.Tensor]:
        """Estimate V(s) for the reward and each constraint."""
        # Handle both old format (precomputed tensors) and new format (reconstruction data)
        z_comp_all = obs.get("z_comp_all")
        design_json = obs.get("design_json")
        active_idx = obs.get("active_idx")
        placed_indices = obs.get("placed_indices")

        # If we have precomputed tensors (old format), use them directly
        if z_comp_all is not None and torch.is_tensor(z_comp_all) and z_comp_all.shape[0] > 0:
            return self.value_heads(z_comp_all)

        # If we have reconstruction data (new format), compute tensors on demand
        if design_json is not None and active_idx is not None and placed_indices is not None:
            design = json.loads(design_json)
            z_comp_all, _, Z_placed = encode_design(design, self.encoder, active_idx, placed_indices)
            # If no components placed yet, use a dummy single-node embedding.
            if Z_placed.shape[0] == 0:
                Z_placed = torch.zeros(1, self.cfg.d, device=self.device)
            return self.value_heads(Z_placed)

        # Fallback: minimal Z_placed
        Z_placed = obs.get("Z_placed", torch.zeros(0, self.cfg.d, device=self.device))
        if Z_placed.shape[0] == 0:
            Z_placed = torch.zeros(1, self.cfg.d, device=self.device)
        return self.value_heads(Z_placed)

    def _prepare_obs(self, env, obs: dict) -> dict:
        """Convert env observations to tensors and attach encoder outputs when available."""
        prepared = self._to_tensor_obs(obs)
        if self.encoder is None:
            return prepared

        design = None
        if hasattr(env, "current_design"):
            design = env.current_design()
        elif getattr(env, "state", None) is not None:
            design = env.state.design

        if design is None:
            return prepared

        active_idx = None
        placed_indices: list[int] = []
        if hasattr(env, "current_active_index"):
            active_idx = env.current_active_index()
        elif getattr(env, "state", None) is not None and env.state.step_idx < len(env.state.placement_order):
            active_idx = env.state.placement_order[env.state.step_idx]

        if hasattr(env, "current_placed_indices"):
            placed_indices = list(env.current_placed_indices())
        elif getattr(env, "state", None) is not None:
            placed_indices = list(env.state.placement_order[: env.state.step_idx])

        if active_idx is None:
            return prepared

        with torch.no_grad():
            z_comp_all, z_star, Z_placed = encode_design(design, self.encoder, active_idx, placed_indices)

        F_pair = torch.zeros((len(placed_indices), self.cfg.pair_dim), dtype=torch.float32, device=self.device)
        try:
            from scpt.training.data import build_pair_features

            pair_features = build_pair_features(design, active_idx, placed_indices)
            if pair_features.shape[-1] == self.cfg.pair_dim:
                F_pair = pair_features.to(device=self.device)
            elif pair_features.numel() > 0:
                F_pair = pair_features[:, : self.cfg.pair_dim].to(device=self.device)
        except Exception:
            pass

        # Store minimal reconstruction data instead of full embeddings to prevent O(T^2) memory growth
        # We'll store design_json and the indices needed to reconstruct embeddings during PPO update
        design_json = None
        if design is not None:
            design_json = json.dumps(design)

        prepared.update({
            "design_json": design_json,
            "active_idx": active_idx,
            "placed_indices": placed_indices,
            "F_pair": F_pair,
        })
        return prepared
    
    def _to_tensor_obs(self, obs: dict) -> dict:
        prepared = dict(obs)
        for key, value in obs.items():
            if torch.is_tensor(value):
                continue
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                if tensor.dtype in (torch.float64, torch.float16):
                    tensor = tensor.to(torch.float32)
                if key in {"action_mask", "grid_xy", "z_star", "Z_placed", "F_pair", "z_comp_all"}:
                    tensor = tensor.to(device=self.device, dtype=torch.float32)
                elif key == "placed_comp_indices":
                    tensor = tensor.to(device=self.device, dtype=torch.long)
                prepared[key] = tensor
            elif isinstance(value, (float, int, bool)):
                prepared[key] = torch.tensor(value, device=self.device)
        return prepared

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_reward_gae(
        self,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        """Compute GAE for the reward signal."""
        T = len(self.buffer)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(
            [v["reward"].item() for v in self.buffer.values], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [float(d) for d in self.buffer.dones], dtype=torch.float32, device=self.device
        )
        return compute_gae(rewards, values, dones, next_value.to(device=self.device), gamma, gae_lambda)

    def _compute_constraint_gae(
        self,
        constraint_name: str,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> torch.Tensor:
        """Compute GAE for one constraint. Averages over LEGAL actions only.

        Per spec §5.3: the expectation in phi_c_i must be over legal actions
        (masked by action_mask), not all actions. We implement this by zeroing
        out the advantage contribution at any step where the action taken was
        in an illegal cell — but since we only sample from legal actions in
        collect_rollout, this condition never fires in practice. The stored
        masks allow a future caller to do exact legal-action averaging.
        """
        T = len(self.buffer)
        costs_t = torch.tensor(
            [step.get(constraint_name, 0.0) for step in self.buffer.costs],
            dtype=torch.float32,
            device=self.device,
        )
        # Use the constraint critic value head.
        values = torch.tensor(
            [v.get(constraint_name, torch.tensor(0.0)).item() for v in self.buffer.values],
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [float(d) for d in self.buffer.dones], dtype=torch.float32
        )
        # Legal-action mask: verify actions were taken in legal cells.
        legal_flags = torch.zeros(T, device=self.device)
        for t, (action, mask) in enumerate(zip(self.buffer.actions, self.buffer.action_masks)):
            mask_cpu = mask.detach().cpu()
            if action < len(mask_cpu) and float(mask_cpu[action]) > 0.5:
                legal_flags[t] = 1.0

        raw_gae = compute_gae(costs_t, values, dones, next_value, gamma, gae_lambda)
        # Zero out steps where the action was illegal (shouldn't happen, but
        # the mask is stored so we can enforce the invariant here).
        return raw_gae * legal_flags

    # ------------------------------------------------------------------
    # PPO-EAL update
    # ------------------------------------------------------------------

    def _ppo_eal_update(self) -> dict:
        """Run PPO-EAL update on the current buffer. Returns diagnostics dict."""
        if len(self.buffer) == 0:
            return {"reward_mean": 0.0, "phi_c": {k: 0.0 for k in self.cfg.constraint_names}}

        # Bootstrap value at end of rollout.
        last_obs = self.buffer.obs_list[-1]
        with torch.no_grad():
            last_value_dict = self._compute_value(last_obs)

        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        # Reward GAE.
        reward_advs = self._compute_reward_gae(
            next_value=last_value_dict["reward"],
            gamma=gamma,
            gae_lambda=lam,
        )
        reward_advs = normalize_advantages(reward_advs)

        # Constraint GAEs + phi_c estimates.
        constraint_advs: dict[str, torch.Tensor] = {}
        phi_c_estimates: dict[str, float] = {}
        j_c_pi_k: dict[str, float] = {}   # mean constraint cost over rollout
        for name in self.cfg.constraint_names:
            next_v = last_value_dict.get(name, torch.tensor(0.0))
            c_advs = self._compute_constraint_gae(name, next_v, gamma, lam)
            constraint_advs[name] = c_advs
            # phi_c = J_c^pi + (1/(1-gamma)) * E[A_c] - b
            j_c = float(torch.tensor([s.get(name, 0.0) for s in self.buffer.costs]).mean())
            j_c_pi_k[name] = j_c
            budget = self.cfg.constraint_budgets.get(name, 0.0)
            phi = j_c + (1.0 / (1.0 - gamma)) * float(c_advs.mean()) - budget
            phi_c_estimates[name] = phi

        # Get current lambdas from dual updater.
        lambdas_float = self.dual_updater.lambdas  # dict[str, float]
        lambdas = {k: torch.tensor(v) for k, v in lambdas_float.items()}
        sigma = self.cfg.sigma

        # Old log-probs (stored at collection time).
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        T = len(self.buffer)

        # PPO epochs.
        total_policy_loss = 0.0
        for _ in range(self.cfg.epochs):
            # Mini-batch loop (full-batch if buffer small enough).
            indices = torch.randperm(T)
            mb = self.cfg.minibatch_size
            for start in range(0, T, mb):
                idx = indices[start:start + mb]
                self.optimizer.zero_grad()

                # Recompute log-probs for the selected steps.
                new_log_probs = []
                for i in idx:
                    i = i.item()
                    obs_i = self.buffer.obs_list[i]

                    # Handle both old format (precomputed tensors) and new format (reconstruction data)
                    z_star = obs_i.get("z_star")
                    Z_placed = obs_i.get("Z_placed")
                    design_json = obs_i.get("design_json")
                    active_idx = obs_i.get("active_idx")
                    placed_indices = obs_i.get("placed_indices")
                    F_pair = obs_i.get("F_pair")

                    # If we have precomputed tensors (old format), use them directly
                    if z_star is not None and torch.is_tensor(z_star) and Z_placed is not None and torch.is_tensor(Z_placed):
                        pass  # Use the precomputed values
                    # If we have reconstruction data (new format), compute tensors on demand
                    elif design_json is not None and active_idx is not None and placed_indices is not None:
                        design = json.loads(design_json)
                        _, z_star, Z_placed = encode_design(design, self.encoder, active_idx, placed_indices)
                    else:
                        # Fallback to defaults
                        z_star = torch.zeros(self.cfg.d, device=self.device)
                        Z_placed = torch.zeros(0, self.cfg.d, device=self.device)

                    F_pair = F_pair if F_pair is not None else torch.zeros(0, self.cfg.pair_dim, device=self.device)
                    grid_xy = obs_i["grid_xy"]
                    # Use stored mask — never recomputed.
                    mask = self.buffer.action_masks[i]
                    logits = self.policy(z_star, Z_placed, F_pair, grid_xy, mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_i = torch.tensor(self.buffer.actions[i], device=self.device)
                    new_log_probs.append(dist.log_prob(action_i))

                new_lp = torch.stack(new_log_probs)
                old_lp = old_log_probs[idx]
                ratio = (new_lp - old_lp).exp()

                adv_batch = reward_advs[idx]
                reward_surr = clipped_ppo_surrogate(ratio, adv_batch, self.cfg.clip_eps)

                # Constraint surrogates.
                phi_tensors = {
                    name: torch.tensor(phi_c_estimates[name])
                    for name in self.cfg.constraint_names
                }
                loss = total_loss(reward_surr, phi_tensors, lambdas, sigma)

                loss.backward()
                self.optimizer.step()
                total_policy_loss += loss.item()

        # Dual update (once per outer iteration, after all PPO epochs).
        self.dual_updater.update(phi_c_estimates)

        return {
            "reward_mean": float(torch.tensor(self.buffer.rewards, device=self.device).mean()),
            "phi_c": phi_c_estimates,
            "policy_loss": total_policy_loss,
            "lambdas": self.dual_updater.lambdas,
        }
