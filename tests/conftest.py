"""Shared pytest fixtures for SCPT-RL test suite.

All fixtures that are reused across multiple test files live here so
individual test files stay focused on their subject matter.

Fixture catalogue:
  two_comp_design   — parsed SCPT IR dict (R1 + C1)
  five_comp_design  — parsed SCPT IR dict (U1 + 2 Rs + 2 Cs + I2C nets)
  smoke_cfg         — SimpleNamespace matching configs/smoke.yaml
  tiny_policy       — SCPTPolicy(d=32, pair_dim=4, n_heads=2, n_layers=1)
  tiny_value_heads  — ValueHeads(d=32, constraint_names=["c_hpwl"])
  fake_env          — FakeEnv stub (no Rust wheel required)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Design JSON fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def two_comp_design() -> dict:
    """SCPT IR dict with two components (R1 + C1) on a 100×100 mm board."""
    with open(FIXTURE_DIR / "two_comp_design.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def five_comp_design() -> dict:
    """SCPT IR dict with five components: MCU (U1) + 2 pull-up Rs + 2 decoupling Cs."""
    with open(FIXTURE_DIR / "five_comp_design.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def minimal_kicad_path() -> Path:
    """Absolute path to the minimal .kicad_pcb file (for Rust-wheel tests)."""
    return FIXTURE_DIR / "minimal.kicad_pcb"


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_cfg() -> SimpleNamespace:
    """Minimal smoke-test config matching configs/smoke.yaml."""
    return SimpleNamespace(
        wandb_project="scpt-rl-smoke",
        wandb_run_name="smoke",
        env=SimpleNamespace(
            board_paths=[],
            grid_resolution_mm=1.0,
            min_spacing_mm=0.2,
            expert_cut_cost=1.0,
            infeasible_penalty=100.0,
        ),
        model=SimpleNamespace(
            d=32,
            pair_dim=4,
            n_heads=2,
            n_layers=1,
        ),
        bc=SimpleNamespace(
            board_paths=[],
            epochs=2,
            lr=1e-3,
        ),
        ppo=SimpleNamespace(
            n_outer_iters=5,
            n_steps_per_iter=16,
            clip_eps=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            lr=1e-3,
            epochs=1,
            minibatch_size=8,
            sigma=1.0,
            constraint_names=["c_hpwl"],
            constraint_budgets={"c_hpwl": 999.0},
            dual_alpha=0.01,
            dual_ema_decay=0.9,
            checkpoint_interval=5,
            log_interval=1,
        ),
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_policy():
    """Minimal SCPTPolicy for fast unit tests (d=32)."""
    from scpt.model.scpt_transformer import SCPTPolicy
    return SCPTPolicy(d=32, pair_dim=4, n_heads=2, n_layers=1)


@pytest.fixture
def tiny_value_heads():
    """Minimal ValueHeads for fast unit tests (d=32, 1 constraint)."""
    from scpt.model.value_heads import ValueHeads
    return ValueHeads(d=32, constraint_names=["c_hpwl"])


@pytest.fixture
def ppo_cfg() -> SimpleNamespace:
    """PPO-EAL training config compatible with tiny_policy + tiny_value_heads."""
    return SimpleNamespace(
        d=32,
        pair_dim=4,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        sigma=1.0,
        constraint_names=["c_hpwl"],
        constraint_budgets={"c_hpwl": 0.5},
        lr=1e-3,
        epochs=1,
        minibatch_size=4,
        dual_alpha=0.01,
        dual_ema_decay=0.9,
    )


# ---------------------------------------------------------------------------
# Fake environment stub (no Rust required)
# ---------------------------------------------------------------------------

class FakeEnv:
    """Minimal deterministic stub env for PPOEALTrainer tests.

    Observation dict has all the keys expected by SCPTPolicy and ValueHeads.
    Episode terminates after ``n_comps`` steps.
    """

    def __init__(self, n_comps: int = 3, grid_cells: int = 16, d: int = 32, pair_dim: int = 4):
        self.n_comps = n_comps
        self.grid_cells = grid_cells
        self.d = d
        self.pair_dim = pair_dim
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return self._obs(), {}

    def step(self, action: int):
        self._step += 1
        terminated = self._step >= self.n_comps
        reward = 0.1 - 0.05 * (action / self.grid_cells)
        costs = {"c_hpwl": 0.1}
        return self._obs(), reward, terminated, False, {"costs": costs}

    def _obs(self) -> dict:
        mask = torch.ones(self.grid_cells)
        mask[0] = 0.0   # first cell always illegal
        mask[-1] = 0.0  # last cell always illegal
        n_placed = max(self._step, 1)
        return {
            "action_mask": mask,
            "grid_xy": torch.randn(self.grid_cells, 2),
            "z_star": torch.randn(self.d),
            "Z_placed": torch.randn(n_placed, self.d),
            "F_pair": torch.randn(n_placed, self.pair_dim),
        }


@pytest.fixture
def fake_env() -> FakeEnv:
    """FakeEnv stub with 3 components and 16 grid cells."""
    return FakeEnv()


@pytest.fixture
def fake_env_factory():
    """Factory fixture: fake_env_factory(n_comps=5, grid_cells=64)."""
    def _factory(n_comps: int = 3, grid_cells: int = 16, d: int = 32, pair_dim: int = 4) -> FakeEnv:
        return FakeEnv(n_comps=n_comps, grid_cells=grid_cells, d=d, pair_dim=pair_dim)
    return _factory
