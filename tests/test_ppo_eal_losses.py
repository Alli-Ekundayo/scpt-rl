"""Tests for pure PPO-EAL loss functions.

The most important test is the sign-convention guard: holding constraint
terms fixed, increasing reward_surrogate (maximize-convention) must
DECREASE total_loss. This catches double-negation bugs that silently
invert the policy gradient.
"""
from __future__ import annotations

import pytest
import torch

from scpt.agent.ppo_eal import (
    augmented_lagrangian_penalty,
    clipped_ppo_surrogate,
    compute_gae,
    constraint_surrogate,
    normalize_advantages,
    total_loss,
)


# ---------------------------------------------------------------------------
# clipped_ppo_surrogate
# ---------------------------------------------------------------------------

def test_clipped_ppo_matches_openai_formula():
    ratio = torch.tensor([0.9, 1.1, 1.3])
    adv = torch.tensor([1.0, -1.0, 0.5])
    out = clipped_ppo_surrogate(ratio, adv, clip_eps=0.2)
    expected = torch.minimum(
        ratio * adv,
        torch.clamp(ratio, 0.8, 1.2) * adv,
    ).mean()
    assert torch.allclose(out, expected)


def test_clipped_ppo_at_ratio_one_equals_advantage_mean():
    ratio = torch.ones(5)
    adv = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0])
    out = clipped_ppo_surrogate(ratio, adv, clip_eps=0.2)
    assert torch.allclose(out, adv.mean())


# ---------------------------------------------------------------------------
# constraint_surrogate
# ---------------------------------------------------------------------------

def test_constraint_surrogate_zero_when_at_budget_no_advantage():
    phi = constraint_surrogate(
        j_c_pi_k=torch.tensor(1.0),
        constraint_advantages=torch.tensor(0.0),
        budget=1.0,
        gamma=0.99,
    )
    assert abs(phi.item()) < 1e-6


def test_constraint_surrogate_positive_when_violated():
    phi = constraint_surrogate(
        j_c_pi_k=torch.tensor(2.0),
        constraint_advantages=torch.tensor(0.0),
        budget=1.0,
        gamma=0.99,
    )
    assert phi.item() > 0.0  # violated: j_c > budget


def test_constraint_surrogate_scales_with_advantage():
    phi_a = constraint_surrogate(torch.tensor(1.0), torch.tensor(0.5), 1.0, 0.99)
    phi_b = constraint_surrogate(torch.tensor(1.0), torch.tensor(1.0), 1.0, 0.99)
    assert phi_b > phi_a


# ---------------------------------------------------------------------------
# augmented_lagrangian_penalty
# ---------------------------------------------------------------------------

def test_augmented_lagrangian_zero_when_phi_c_negative_and_lambda_zero():
    out = augmented_lagrangian_penalty(
        phi_c=torch.tensor(-0.5),
        lam=torch.tensor(0.0),
        sigma=1.0,
    )
    assert abs(out.item()) < 1e-6


def test_augmented_lagrangian_positive_when_phi_c_positive():
    out = augmented_lagrangian_penalty(
        phi_c=torch.tensor(0.5),
        lam=torch.tensor(0.0),
        sigma=1.0,
    )
    assert out.item() > 0.0


def test_augmented_lagrangian_grows_with_phi_c():
    a = augmented_lagrangian_penalty(torch.tensor(0.1), torch.tensor(0.0), 1.0)
    b = augmented_lagrangian_penalty(torch.tensor(0.5), torch.tensor(0.0), 1.0)
    assert b > a


# ---------------------------------------------------------------------------
# total_loss — the sign-convention guard
# ---------------------------------------------------------------------------

def test_total_loss_decreases_when_reward_surrogate_increases():
    """THE critical test. Holding constraint terms fixed, increasing the
    reward surrogate (maximize-convention) must DECREASE total_loss.
    Catches double-negation regression."""
    phi_c = {"c_hpwl": torch.tensor(0.1)}
    lam = {"c_hpwl": torch.tensor(1.0)}
    sigma = 1.0

    loss_a = total_loss(reward_surrogate=torch.tensor(0.5),
                        constraint_phis=phi_c, lambdas=lam, sigma=sigma)
    loss_b = total_loss(reward_surrogate=torch.tensor(1.0),
                        constraint_phis=phi_c, lambdas=lam, sigma=sigma)
    assert loss_b < loss_a, (
        f"double-negation regression: loss_b={loss_b.item()} should be "
        f"< loss_a={loss_a.item()}"
    )


def test_total_loss_increases_when_phi_c_worsens():
    """With reward surrogate fixed, worsening constraint (higher phi_c)
    should INCREASE total_loss."""
    phi_c_good = {"c_hpwl": torch.tensor(-0.1)}
    phi_c_bad = {"c_hpwl": torch.tensor(0.5)}
    lam = {"c_hpwl": torch.tensor(1.0)}
    sigma = 1.0
    rew = torch.tensor(0.5)
    loss_good = total_loss(rew, phi_c_good, lam, sigma)
    loss_bad = total_loss(rew, phi_c_bad, lam, sigma)
    assert loss_bad > loss_good


def test_total_loss_with_no_constraints_equals_negative_reward_surrogate():
    out = total_loss(reward_surrogate=torch.tensor(0.5),
                     constraint_phis={}, lambdas={}, sigma=1.0)
    assert torch.allclose(out, torch.tensor(-0.5))


# ---------------------------------------------------------------------------
# GAE + normalization
# ---------------------------------------------------------------------------

def test_compute_gae_shape():
    rewards = torch.randn(10)
    values = torch.randn(10)
    dones = torch.zeros(10)
    next_v = torch.tensor(0.0)
    adv = compute_gae(rewards, values, dones, next_v, gamma=0.99, gae_lambda=0.95)
    assert adv.shape == (10,)


def test_compute_gae_terminal_resets():
    # When done[t]=1, GAE should reset at step t+1.
    rewards = torch.ones(5)
    values = torch.zeros(5)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    next_v = torch.tensor(0.0)
    adv = compute_gae(rewards, values, dones, next_v, gamma=0.99, gae_lambda=0.95)
    # After done at step 2, GAE at step 3 should start fresh.
    # Step 3: delta = 1 + 0.99*0 - 0 = 1, last_gae = 1
    # Step 2: last_gae was reset (done=1), so step 2's delta is just the reward 1
    assert adv[3].item() > 0.99  # approximately 1.0


def test_normalize_advantages_zero_mean_unit_std():
    adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalize_advantages(adv)
    assert abs(normed.mean().item()) < 1e-5
    assert abs(normed.std().item() - 1.0) < 0.1


def test_normalize_advantages_handles_constant():
    adv = torch.ones(5)
    normed = normalize_advantages(adv)
    # Constant input → std=0, should return (adv - mean) = zeros
    assert torch.allclose(normed, torch.zeros(5), atol=1e-5)
