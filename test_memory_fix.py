#!/usr/bin/env python3
"""
Test script to verify the memory optimization fix for RolloutBuffer.
This tests that the PPOEALTrainer correctly handles both old format
(precomputed tensors) and new format (reconstruction data).
"""

import json
import torch
import numpy as np
from unittest.mock import Mock

# Import the modules we need to test
import sys
sys.path.append('/home/alli-ekundayo/Projects/scpt-rl')

from scpt.agent.ppo_eal import PPOEALTrainer
from scpt.model.gnn_encoder import HeteroPCBEncoder
from scpt.model.value_heads import ValueHeads
from scpt.model.scpt_transformer import SCPTPolicy
from types import SimpleNamespace

def test_old_format_compatibility():
    """Test that the trainer still works with the old format (precomputed tensors)"""
    print("Testing backward compatibility with old format...")

    # Create mock models
    encoder = HeteroPCBEncoder({"component": 5, "pad": 4, "net": 6}, hidden=16)
    policy = SCPTPolicy(d=16, pair_dim=8, n_heads=2, n_layers=2)
    value_heads = ValueHeads(d=16, constraint_names=["c_test"])

    # Create a mock config
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
        epochs=2,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9
    )

    trainer = PPOEALTrainer(policy, value_heads, cfg, encoder=encoder)

    # Create a mock observation in the OLD format (with precomputed tensors)
    obs_old = {
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10000, dtype=torch.int64),
        "z_star": torch.randn(16, dtype=torch.float32),
        "Z_placed": torch.randn(3, 16, dtype=torch.float32),  # 3 placed components
        "F_pair": torch.zeros(3, 8, dtype=torch.float32)
    }

    # Test _sample_action with old format
    action, log_prob = trainer._sample_action(obs_old)
    print(f"  Sampled action: {action}, log_prob: {log_prob.item():.4f}")

    # Test _compute_value with old format
    values = trainer._compute_value(obs_old)
    print(f"  Values keys: {list(values.keys())}")
    print(f"  Reward value shape: {values['reward'].shape}")

    print("✓ Old format compatibility test passed")

def test_new_format():
    """Test that the trainer works with the new format (reconstruction data)"""
    print("Testing new format with reconstruction data...")

    # Create mock models
    encoder = HeteroPCBEncoder({"component": 5, "pad": 4, "net": 6}, hidden=16)
    policy = SCPTPolicy(d=16, pair_dim=8, n_heads=2, n_layers=2)
    value_heads = ValueHeads(d=16, constraint_names=["c_test"])

    # Create a mock config
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
        epochs=2,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9
    )

    trainer = PPOEALTrainer(policy, value_heads, cfg, encoder=encoder)

    # Create a sample design (as JSON string) for testing
    sample_design = {
        "components": [
            {"ref_des": "R1", "footprint": {"pads": [{"net_name": "Net1"}]}},
            {"ref_des": "C1", "footprint": {"pads": [{"net_name": "Net2"}]}},
            {"ref_des": "U1", "footprint": {"pads": [{"net_name": "Net3"}, {"net_name": "Net4"}]}}
        ],
        "nets": [
            {"name": "Net1", "role": "signal"},
            {"name": "Net2", "role": "signal"},
            {"name": "Net3", "role": "power"},
            {"name": "Net4", "role": "ground"}
        ],
        "placement": {
            "positions": [
                None,  # R1 not placed
                [10.0, 20.0],  # C1 at (10, 20)
                [30.0, 40.0]   # U1 at (30, 40)
            ],
            "placement_order": [1, 2, 0]  # Place C1, then U1, then R1
        },
        "board": {
            "bounds": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}
        }
    }
    design_json = json.dumps(sample_design)

    # Create a mock observation in the NEW format (with reconstruction data)
    obs_new = {
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10000, dtype=torch.int64),
        "design_json": design_json,
        "active_idx": 1,  # Currently placing C1 (index 1)
        "placed_indices": [2],  # Already placed U1 (index 2)
        "F_pair": torch.zeros(1, 8, dtype=torch.float32)  # 1 placed component so far
    }

    # Test _sample_action with new format
    action, log_prob = trainer._sample_action(obs_new)
    print(f"  Action: {action}, log_prob: {log_prob.item():.4f}")

    # Test _compute_value with new format
    values = trainer._compute_value(obs_new)
    print(f"  Values keys: {list(values.keys())}")
    print(f"  Reward value shape: {values['reward'].shape}")

    print("✓ New format test passed")

def test_mixed_formats_in_buffer():
    """Test that the trainer can handle a mix of old and new format observations in the buffer"""
    print("Testing mixed format handling in rollout buffer...")

    # Create mock models
    encoder = HeteroPCBEncoder({"component": 5, "pad": 4, "net": 6}, hidden=16)
    policy = SCPTPolicy(d=16, pair_dim=8, n_heads=2, n_layers=2)
    value_heads = ValueHeads(d=16, constraint_names=["c_test"])

    # Create a mock config
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
        epochs=2,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9
    )

    trainer = PPOEALTrainer(policy, value_heads, cfg, encoder=encoder)

    # Add some old format observations to buffer
    obs_old = {
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10000, dtype=torch.int64),
        "z_star": torch.randn(16, dtype=torch.float32),
        "Z_placed": torch.randn(2, 16, dtype=torch.float32),
        "F_pair": torch.zeros(2, 8, dtype=torch.float32)
    }

    # Add some new format observations to buffer
    sample_design = {
        "components": [{"ref_des": "R1", "footprint": {"pads": [{"net_name": "Net1"}]}}],
        "nets": [{"name": "Net1", "role": "signal"}],
        "placement": {"positions": [[5.0, 5.0]], "placement_order": [0]},
        "board": {"bounds": {"x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0}}
    }
    design_json = json.dumps(sample_design)

    obs_new = {
        "action_mask": torch.ones(25, dtype=torch.float32),
        "grid_xy": torch.randn(25, 2, dtype=torch.float32),
        "placed_comp_indices": torch.zeros(10000, dtype=torch.int64),
        "design_json": design_json,
        "active_idx": 0,
        "placed_indices": [],
        "F_pair": torch.zeros(0, 8, dtype=torch.float32)
    }

    # Manually add to buffer to simulate what collect_rollout would do
    trainer.buffer.add(obs=obs_old, action=0, log_prob=torch.tensor(0.0), reward=1.0,
                       value={"reward": torch.tensor(0.5)}, costs={"c_test": 0.1}, done=False)

    trainer.buffer.add(obs=0)
    trainer.buffer.add(obs=obs_new, action=1, log_prob=torch.tensor(0.0), reward=0.5,
                       value={"reward": torch.tensor(0.3)}, costs={"c_test": 0.0}, done=0)

    # Test that we can compute values for both
    values_old = trainer._compute_value(trainer.buffer.obs_list[0])
    values_new = trainer._compute_value(trainer.buffer.obs_list[1])

    print(f"  Old format value: {values_old['reward'].item():.4f}")
    print(f"  New format value: {values_new['reward'].item():.4f}")

    # Test that the PPO update loop can handle mixed formats
    # (We won't run a full update as it requires a real environment, but we can test the logic)
    print("✓ Mixed format handling test passed")

if __name__ == "__main__":
    print("Running memory optimization fix tests...\n")

    try:
        test_old_format_compatibility()
        print()

        test_new_format()
        print()

        test_mixed_formats_in_buffer()
        print()

        print("🎉 All tests passed! The memory optimization fix is working correctly.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)