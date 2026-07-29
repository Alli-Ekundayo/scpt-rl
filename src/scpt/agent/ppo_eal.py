"""Pure PPO-EAL loss functions.

These are the mathematical core of the agent — isolated from any torch
autograd concerns so they can be independently tested. Each function is
either maximize-convention (returns a scalar we want to *maximize*) or
minimize-convention (returns a scalar we want to *minimize*). The boundary
between the two is in `total_loss`, which applies the negation **once**.

Sign convention (the #1 source of PPO bugs):
- `clipped_ppo_surrogate` returns the STANDARD OpenAI maximize-convention
  value: min(ratio * adv, clip(ratio) * adv). Higher is better.
- `constraint_surrogate` returns phi_c_i in the form we want to keep <= 0
  for the constraint to be satisfied.
- `augmented_lagrangian_penalty` returns the AL penalty, higher means more
  violated.
- `total_loss` returns the scalar we MINIMIZE. It applies the negation to
  the reward surrogate exactly once here. Guarded by a dedicated test.
"""
from __future__ import annotations

import torch


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
    advantages = torch.zeros(T, dtype=rewards.dtype)
    last_gae = 0.0
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
