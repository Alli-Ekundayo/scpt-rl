"""Tests for the SCPT-RL Gymnasium env."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpt.env.pcb_env import EnvConfig, PcbPlacementEnv

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"


def test_reset_returns_obs_and_info():
    env = PcbPlacementEnv(FIXTURE)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert "action_mask" in obs
    assert "grid_xy" in obs
    assert "placed_comp_indices" in obs
    assert isinstance(info, dict)


def test_observation_shapes():
    env = PcbPlacementEnv(FIXTURE)
    obs, _ = env.reset(seed=42)
    # Grid is 100mm x 100mm board at 0.5mm resolution → 200x200 = 40000 cells
    assert obs["action_mask"].shape == (env.H * env.W,)
    assert obs["grid_xy"].shape == (env.H * env.W, 2)
    assert obs["placed_comp_indices"].shape == (10_000,)


def test_action_space_is_discrete():
    env = PcbPlacementEnv(FIXTURE)
    env.reset(seed=42)
    assert env.action_space.n == env.H * env.W


def test_step_returns_five_tuple():
    env = PcbPlacementEnv(FIXTURE)
    obs, _ = env.reset(seed=42)
    # Pick a legal action.
    legal = (obs["action_mask"] > 0).nonzero()[0]
    if len(legal) == 0:
        pytest.skip("no legal actions in fixture")
    result = env.step(int(legal[0]))
    assert len(result) == 5  # (obs, reward, terminated, truncated, info)


def test_step_returns_cost_dict():
    env = PcbPlacementEnv(FIXTURE)
    obs, _ = env.reset(seed=42)
    legal = (obs["action_mask"] > 0).nonzero()[0]
    if len(legal) == 0:
        pytest.skip("no legal actions")
    _, _, _, _, info = env.step(int(legal[0]))
    assert "costs" in info
    costs = info["costs"]
    assert "c_clearance" in costs
    assert "c_hpwl" in costs
    assert "r_tier2" in costs


def test_episode_terminates_when_all_placed():
    env = PcbPlacementEnv(FIXTURE)
    obs, _ = env.reset(seed=42)
    done = False
    steps = 0
    max_steps = 100  # guard against infinite loops
    while not done and steps < max_steps:
        legal = (obs["action_mask"] > 0).nonzero()[0]
        if len(legal) == 0:
            break  # infeasible mask
        obs, _, terminated, truncated, _ = env.step(int(legal[0]))
        done = terminated or truncated
        steps += 1
    assert done, "episode should terminate after placing all components"


def test_reset_reloads_fresh_placement():
    env = PcbPlacementEnv(FIXTURE)
    env.reset(seed=42)
    # Take a step to advance.
    obs, _ = env.reset(seed=42)
    legal = (obs["action_mask"] > 0).nonzero()[0]
    if len(legal) > 0:
        env.step(int(legal[0]))
    # Reset again — should be back to initial state.
    obs2, _ = env.reset(seed=42)
    # The initial design has all components pre-placed (from the KiCad file),
    # but step_idx should be 0 after reset.
    assert env.state.step_idx == 0
