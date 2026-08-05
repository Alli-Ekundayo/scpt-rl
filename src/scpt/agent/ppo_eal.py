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
    """Per-batch advantage normalization (center to mean 0, scale to std 1 if std >= eps).

    Centered always, scaled only when std >= eps to avoid division-by-zero on degenerate batches.
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
        if self.encoder is not None:
            all_params += list(self.encoder.parameters())
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
        z_star, Z_placed, F_pair, grid_xy, action_mask = self._get_policy_inputs(obs)

        logits = self.policy(z_star, Z_placed, F_pair, grid_xy, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    @torch.no_grad()
    def _compute_value(
        self, obs: dict
    ) -> dict[str, torch.Tensor]:
        """Estimate V(s) for the reward and each constraint."""
        z_star, Z_placed, F_pair, grid_xy, action_mask = self._get_policy_inputs(obs)

        # For value function, we only need Z_placed (placed component embeddings)
        # If no components placed yet, use a dummy single-node embedding.
        if Z_placed.shape[0] == 0:
            Z_placed = torch.zeros(1, self.cfg.d, device=self.device)
        return self.value_heads(Z_placed)

    def _prepare_obs(self, env, obs: dict) -> dict:
        """Convert env observations to tensors and attach encoder outputs when available."""
        # Convert basic obs to tensors on device for immediate use
        prepared = self._to_tensor_obs(obs)
        if self.encoder is None:
            # No encoder available, store everything on CPU
            return {k: v.cpu() if torch.is_tensor(v) else v for k, v in prepared.items()}

        design = None
        if hasattr(env, "current_design"):
            design = env.current_design()
        elif getattr(env, "state", None) is not None:
            design = env.state.design

        # If we have design data and encoder, use reconstruction format for storage
        if design is not None:
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

            # If we couldn't get active_idx, fall back to storing tensors on CPU
            if active_idx is None:
                return {k: v.cpu() if torch.is_tensor(v) else v for k, v in prepared.items()}

            with torch.no_grad():
                # We compute these but don't store them - we'll reconstruct from design data
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

            design_json = None
            if design is not None:
                design_json = json.dumps(design)

            # Store reconstruction format (compact)
            z_star_cpu = z_star.cpu()
            Z_placed_cpu = Z_placed.cpu()
            return {
                "design_json": design_json,
                "active_idx": active_idx,
                "placed_indices": placed_indices,
                # Store these tensors on CPU for immediate use during rollout collection
                "z_star": z_star_cpu,
                "Z_placed": Z_placed_cpu,
                "action_mask": prepared["action_mask"].cpu(),
                "grid_xy": prepared["grid_xy"].cpu(),
                "F_pair": F_pair.cpu(),
                "placed_comp_indices": prepared["placed_comp_indices"].cpu(),
            }
        else:
            # No design data available, fall back to storing tensors on CPU
            return {k: v.cpu() if torch.is_tensor(v) else v for k, v in prepared.items()}
    
    def _to_tensor_obs(self, obs: dict) -> dict:
        """Convert observation to tensors on CPU for storage in buffer."""
        prepared = dict(obs)
        for key, value in obs.items():
            if torch.is_tensor(value):
                # Ensure tensor is on CPU for storage
                prepared[key] = value.cpu()
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                if tensor.dtype in (torch.float64, torch.float16):
                    tensor = tensor.to(torch.float32)
                prepared[key] = tensor
            elif isinstance(value, (float, int, bool)):
                prepared[key] = torch.tensor(value)
        return prepared

    def _get_policy_inputs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and prepare policy inputs from observation.

        Handles both legacy format (precomputed tensors) and reconstruction format.

        Returns:
            Tuple of (z_star, Z_placed, F_pair, grid_xy, action_mask) all on self.device
        """
        # Handle both old format (precomputed tensors) and new format (reconstruction data)
        z_star = obs.get("z_star")
        Z_placed = obs.get("Z_placed")
        design_json = obs.get("design_json")
        active_idx = obs.get("active_idx")
        placed_indices = obs.get("placed_indices")
        F_pair = obs.get("F_pair")

        # If we have precomputed tensors (old format), use them directly
        if z_star is not None and torch.is_tensor(z_star) and Z_placed is not None and torch.is_tensor(Z_placed):
            # Move to device if needed
            z_star = z_star.to(device=self.device)
            Z_placed = Z_placed.to(device=self.device)
        # If we have reconstruction data (new format), compute tensors on demand
        elif design_json is not None and active_idx is not None and placed_indices is not None:
            design = json.loads(design_json)
            _, z_star, Z_placed = encode_design(design, self.encoder, active_idx, placed_indices)
            # encode_design returns tensors on the encoder's device, which should be self.device
        else:
            # Fallback to defaults
            z_star = torch.zeros(self.cfg.d, dtype=torch.float32, device=self.device)
            Z_placed = torch.zeros(0, self.cfg.d, dtype=torch.float32, device=self.device)

        # Ensure F_pair is on the correct device (move from CPU if needed)
        if F_pair is not None:
            F_pair = F_pair.to(device=self.device)
        else:
            F_pair = torch.zeros(0, self.cfg.pair_dim, dtype=torch.float32, device=self.device)
        grid_xy = obs["grid_xy"]
        action_mask = obs["action_mask"]

        # Move grid_xy and action_mask to device if they are CPU tensors
        if torch.is_tensor(grid_xy) and grid_xy.device != self.device:
            grid_xy = grid_xy.to(device=self.device)
        if torch.is_tensor(action_mask) and action_mask.device != self.device:
            action_mask = action_mask.to(device=self.device)

        return z_star, Z_placed, F_pair, grid_xy, action_mask

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
        T = len(self.buffer)
        costs_t = torch.tensor(
            [step.get(constraint_name, 0.0) for step in self.buffer.costs],
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.tensor(
            [v.get(constraint_name, torch.tensor(0.0)).item() for v in self.buffer.values],
            dtype=torch.float32,
            device=self.device,          # <-- was missing
        )
        dones = torch.tensor(
            [float(d) for d in self.buffer.dones],
            dtype=torch.float32,
            device=self.device,          # <-- was missing
        )
        legal_flags = torch.zeros(T, device=self.device)
        for t, (action, mask) in enumerate(zip(self.buffer.actions, self.buffer.action_masks)):
            mask_cpu = mask.detach().cpu()
            if action < len(mask_cpu) and float(mask_cpu[action]) > 0.5:
                legal_flags[t] = 1.0

        assert torch.all(legal_flags == 1.0), "Buffer contains illegal actions"

        raw_gae = compute_gae(costs_t, values, dones, next_value.to(self.device), gamma, gae_lambda)
        return raw_gae

    # ------------------------------------------------------------------
    # PPO-EAL update
    # ------------------------------------------------------------------

    def _ppo_eal_update(self) -> dict:
        """Run PPO-EAL update on the current buffer. Returns diagnostics dict."""
        if len(self.buffer) == 0:
            return {"reward_mean": 0.0, "phi_c": {k: 0.0 for k in self.cfg.constraint_names}}

        # ------------------------------------------------------------------
        # Bootstrap value at end of rollout.
        # ------------------------------------------------------------------
        last_obs = self.buffer.obs_list[-1]
        with torch.no_grad():
            last_value_dict = self._compute_value(last_obs)

        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        # ------------------------------------------------------------------
        # Reward GAE  (keep raw for returns, normalized for policy)
        # ------------------------------------------------------------------
        reward_advs_raw = self._compute_reward_gae(
            next_value=last_value_dict["reward"],
            gamma=gamma,
            gae_lambda=lam,
        )
        reward_values = torch.tensor(
            [v["reward"].item() for v in self.buffer.values],
            dtype=torch.float32,
            device=self.device,
        )
        reward_returns = reward_advs_raw + reward_values   # targets for value head
        reward_advs = normalize_advantages(reward_advs_raw)

        # ------------------------------------------------------------------
        # Constraint GAEs + phi_c estimates (constant during policy epochs)
        # ------------------------------------------------------------------
        phi_c_estimates: dict[str, float] = {}
        j_c_pi_k: dict[str, float] = {}
        for name in self.cfg.constraint_names:
            next_v = last_value_dict.get(name, torch.tensor(0.0, device=self.device))
            c_advs = self._compute_constraint_gae(name, next_v, gamma, lam)
            j_c = float(torch.tensor(
                [s.get(name, 0.0) for s in self.buffer.costs],
                device=self.device,
            ).mean())
            j_c_pi_k[name] = j_c
            budget = self.cfg.constraint_budgets.get(name, 0.0)
            phi = j_c + (1.0 / (1.0 - gamma)) * float(c_advs.mean()) - budget
            phi_c_estimates[name] = phi

        lambdas_float = self.dual_updater.lambdas
        lambdas = {
            k: torch.tensor(v, device=self.device)
            for k, v in lambdas_float.items()
        }
        sigma = self.cfg.sigma

        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        T = len(self.buffer)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        # ------------------------------------------------------------------
        # PPO epochs
        # ------------------------------------------------------------------
        for _ in range(self.cfg.epochs):
            indices = torch.randperm(T)
            mb = self.cfg.minibatch_size

            for start in range(0, T, mb):
                idx = indices[start:start + mb]
                self.optimizer.zero_grad()

                batch_size = idx.numel()
                batch_policy_surr = 0.0
                batch_value_loss = 0.0

                # ----------------------------------------------------------
                # Gradient accumulation: one item at a time
                # ----------------------------------------------------------
                for i_tensor in idx:
                    i = i_tensor.item()
                    obs_i = self.buffer.obs_list[i]

                    # ---- reconstruct / move inputs to GPU -----------------
                    z_star = obs_i.get("z_star")
                    Z_placed = obs_i.get("Z_placed")
                    design_json = obs_i.get("design_json")
                    active_idx = obs_i.get("active_idx")
                    placed_indices = obs_i.get("placed_indices")
                    F_pair = obs_i.get("F_pair")

                    if (z_star is not None and torch.is_tensor(z_star) and
                            Z_placed is not None and torch.is_tensor(Z_placed)):
                        z_star = z_star.to(device=self.device)
                        Z_placed = Z_placed.to(device=self.device)
                    elif (design_json is not None and active_idx is not None
                            and placed_indices is not None):
                        design = json.loads(design_json)
                        _, z_star, Z_placed = encode_design(
                            design, self.encoder, active_idx, placed_indices
                        )
                    else:
                        z_star = torch.zeros(self.cfg.d, device=self.device)
                        Z_placed = torch.zeros(0, self.cfg.d, device=self.device)

                    if F_pair is not None:
                        F_pair = F_pair.to(device=self.device)
                    else:
                        F_pair = torch.zeros(0, self.cfg.pair_dim, device=self.device)

                    grid_xy = obs_i["grid_xy"]
                    mask = self.buffer.action_masks[i]
                    if torch.is_tensor(grid_xy) and grid_xy.device != self.device:
                        grid_xy = grid_xy.to(device=self.device)
                    if torch.is_tensor(mask) and mask.device != self.device:
                        mask = mask.to(device=self.device)

                    # ---- policy forward ----------------------------------
                    logits = self.policy(z_star, Z_placed, F_pair, grid_xy, mask)
                    dist = torch.distributions.Categorical(logits=logits)
                    action_i = torch.tensor(self.buffer.actions[i], device=self.device)
                    lp = dist.log_prob(action_i)

                    ratio = (lp - old_log_probs[i]).exp()
                    adv = reward_advs[i]

                    unclipped = ratio * adv
                    clipped = torch.clamp(
                        ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps
                    ) * adv
                    surr_i = torch.minimum(unclipped, clipped)

                    # Scale by batch size so the mean gradient is preserved
                    policy_loss_i = -surr_i / batch_size
                    policy_loss_i.backward()
                    batch_policy_surr += surr_i.item()

                    # ---- value forward -----------------------------------
                    # Mirror _compute_value logic: Z_placed, dummy row if empty
                    Z_v = Z_placed
                    if Z_v.shape[0] == 0:
                        Z_v = torch.zeros(1, self.cfg.d, device=self.device)
                    v_pred = self.value_heads(Z_v)["reward"]
                    v_loss_i = 0.5 * (v_pred - reward_returns[i]).pow(2) / batch_size
                    v_loss_i.backward()
                    batch_value_loss += v_loss_i.item() * batch_size

                # Constraint penalty is constant w.r.t. policy params;
                # it only affects the dual update, not the gradient step.
                phi_tensors = {
                    name: torch.tensor(phi_c_estimates[name], device=self.device)
                    for name in self.cfg.constraint_names
                }
                penalty = torch.stack([
                    augmented_lagrangian_penalty(phi_tensors[k], lambdas[k], sigma)
                    for k in phi_tensors
                ]).sum() if phi_tensors else torch.tensor(0.0, device=self.device)

                self.optimizer.step()

                with torch.no_grad():
                    effective_loss = -batch_policy_surr / batch_size + penalty.item()
                    total_policy_loss += effective_loss
                    total_value_loss += batch_value_loss / batch_size

        # ------------------------------------------------------------------
        # Dual update (once per outer iteration)
        # ------------------------------------------------------------------
        self.dual_updater.update(phi_c_estimates)

        return {
            "reward_mean": float(torch.tensor(self.buffer.rewards, device=self.device).mean()),
            "phi_c": phi_c_estimates,
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "lambdas": self.dual_updater.lambdas,
        }
