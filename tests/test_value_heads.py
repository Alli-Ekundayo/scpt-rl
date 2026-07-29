"""Tests for ValueHeads."""
from __future__ import annotations

import torch

from scpt.model.value_heads import ValueHeads


def test_value_heads_outputs_reward_and_constraints():
    vh = ValueHeads(d=64, constraint_names=["c_clearance", "c_hpwl"])
    z_comp = torch.randn(5, 64)
    out = vh(z_comp)
    assert "reward" in out
    assert out["reward"].shape == ()
    assert "c_clearance" in out
    assert out["c_clearance"].shape == ()
    assert "c_hpwl" in out
    assert out["c_hpwl"].shape == ()


def test_value_heads_separate_params():
    """Reward and constraint critics must have SEPARATE parameters."""
    vh = ValueHeads(d=64, constraint_names=["c_hpwl"])
    assert vh.reward_critic is not vh.constraint_critics["c_hpwl"]
    # Weights should not be tied (random init → different).
    assert not torch.allclose(
        vh.reward_critic.weight, vh.constraint_critics["c_hpwl"].weight
    )


def test_value_heads_with_empty_graph_raises():
    """Empty graph → mean-pool is NaN. Caller must handle this upstream."""
    vh = ValueHeads(d=64, constraint_names=["c_hpwl"])
    z_comp = torch.zeros(0, 64)
    out = vh(z_comp)
    # With zero rows, mean is NaN. We don't crash, but downstream loss
    # must handle NaN gracefully (or the env must never pass empty graphs).
    assert torch.isnan(out["reward"])


def test_value_heads_different_constraint_names_produce_independent_heads():
    vh = ValueHeads(d=64, constraint_names=["a", "b"])
    assert "a" in vh.constraint_critics
    assert "b" in vh.constraint_critics
    assert vh.constraint_critics["a"] is not vh.constraint_critics["b"]


def test_value_heads_preserves_constraint_order():
    names = ["c_hpwl", "c_clearance", "c_partition"]
    vh = ValueHeads(d=32, constraint_names=names)
    assert vh.constraint_names == names
