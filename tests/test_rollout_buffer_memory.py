"""Tests for RolloutBuffer compact format compatibility and memory handling."""
from __future__ import annotations

import json
from types import SimpleNamespace
import torch

from scpt.agent.ppo_eal import PPOEALTrainer
from scpt.model.gnn_encoder import HeteroPCBEncoder
from scpt.model.scpt_transformer import SCPTPolicy
from scpt.model.value_heads import ValueHeads


def test_old_format_compatibility():
    """Trainer handles precomputed tensors (legacy format)."""
    encoder = HeteroPCBEncoder({"component": 5, "pad": 4, "net": 6}, hidden=16)
    policy = SCPTPolicy(d=16, pair_dim=8, n_heads=2, n_layers=2)
    value_heads = ValueHeads(d=16, constraint_names=["c_test"])

    cfg = SimpleNamespace(
        d=16,
        pair_dim=8,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        sigma=1.0,
        constraint_names=["c_test"],
        constraint_budgets={"c_test": 1.0},
        lr=1e-3,
        epochs=1,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9,
    )

    trainer = PPOEALTrainer(policy, value_heads, cfg, encoder=encoder)

    obs_old = {
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10, dtype=torch.int64),
        "z_star": torch.randn(16, dtype=torch.float32),
        "Z_placed": torch.randn(3, 16, dtype=torch.float32),
        "F_pair": torch.zeros(3, 8, dtype=torch.float32),
    }

    z_star, Z_placed, F_pair, grid_xy, action_mask = trainer._get_policy_inputs(obs_old)
    assert z_star.shape == (16,)
    assert Z_placed.shape == (3, 16)
    assert F_pair.shape == (3, 8)


def test_reconstruction_format_compatibility():
    """Trainer handles compact reconstruction format."""
    encoder = HeteroPCBEncoder({"component": 5, "pad": 4, "net": 6}, hidden=16)
    policy = SCPTPolicy(d=16, pair_dim=8, n_heads=2, n_layers=2)
    value_heads = ValueHeads(d=16, constraint_names=["c_test"])

    cfg = SimpleNamespace(
        d=16,
        pair_dim=8,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        sigma=1.0,
        constraint_names=["c_test"],
        constraint_budgets={"c_test": 1.0},
        lr=1e-3,
        epochs=1,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9,
    )

    trainer = PPOEALTrainer(policy, value_heads, cfg, encoder=encoder)

    sample_design = {
        "board": {"bounds": {"x": 0, "y": 0, "w": 10, "h": 10}},
        "components": [
            {"ref_des": "R1", "courtyard": {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
            {"ref_des": "R2", "courtyard": {"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}},
        ],
        "nets": [{"name": "NET1", "pads": [["R1", 1], ["R2", 1]]}],
        "placement": {"positions": [{"position": [1.0, 1.0]}, None]},
    }

    obs_new = {
        "design": sample_design,
        "active_idx": 1,
        "placed_indices": [0],
        "F_pair": torch.zeros(1, 8, dtype=torch.float32),
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10, dtype=torch.int64),
    }

    z_star, Z_placed, F_pair, grid_xy, action_mask = trainer._get_policy_inputs(obs_new)
    assert z_star.shape == (16,)
    assert Z_placed.shape == (1, 16)
    assert F_pair.shape == (1, 8)
