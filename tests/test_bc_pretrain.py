"""Tests for BC pretraining.

Key invariants:
1. BCDataset builds episodes from design dicts correctly.
2. bc_pretrain reduces the BC loss on a toy dataset.
3. Infeasible expert cells are skipped gracefully.
4. eval_bc_loss returns finite loss on valid dataset.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from scpt.training.bc_pretrain import BCDataset, BCEpisode, bc_pretrain, eval_bc_loss
from scpt.model.scpt_transformer import SCPTPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
PAIR_DIM = 4


def _cfg(**overrides) -> SimpleNamespace:
    c = SimpleNamespace(
        grid_resolution_mm=1.0,
        min_spacing_mm=0.2,
        d=D,
        pair_dim=PAIR_DIM,
        epochs=5,
        lr=1e-2,
    )
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _small_design(n_comps: int = 3, board_w: float = 10.0, board_h: float = 10.0) -> dict:
    """Minimal in-memory design dict for testing (no Rust wheel needed)."""
    components = []
    nets = []
    positions = []
    for i in range(n_comps):
        ref = f"R{i+1}"
        # Place component in a grid layout so expert positions are non-overlapping.
        ex = 1.0 + i * 3.0
        ey = 5.0
        components.append({
            "ref_des": ref,
            "footprint": {
                "pads": [{"local_pos": [0.0, 0.0], "net_name": f"N{i}", "electrical_proxy_confidence": 1.0}],
                "courtyard": {"points": [[ex-1, ey-1], [ex+1, ey-1], [ex+1, ey+1], [ex-1, ey+1]]},
            },
            "value": "1k",
        })
        nets.append({"name": f"N{i}", "role": "signal", "pads": [[i, 0]]})
        positions.append({"component_idx": i, "position": [ex, ey], "rotation_deg": 0.0, "bottom_layer": False})
    return {
        "board": {"bounds": {"x": 0.0, "y": 0.0, "w": board_w, "h": board_h}},
        "components": components,
        "nets": nets,
        "netclasses": [],
        "diff_pairs": [],
        "placement": {
            "positions": positions,
            "placement_order": list(range(n_comps)),
        },
    }


def _make_policy(d: int = D, pair_dim: int = PAIR_DIM) -> SCPTPolicy:
    return SCPTPolicy(d=d, pair_dim=pair_dim, n_heads=2, n_layers=1)


# ---------------------------------------------------------------------------
# BCDataset tests
# ---------------------------------------------------------------------------

def test_dataset_builds_from_dict():
    design = _small_design(n_comps=3)
    dataset = BCDataset(board_paths=[design], cfg=_cfg())
    assert len(dataset) >= 1
    ep = dataset[0]
    assert isinstance(ep, BCEpisode)
    assert len(ep.steps) > 0


def test_dataset_each_step_has_obs_and_action():
    design = _small_design(n_comps=3)
    dataset = BCDataset(board_paths=[design], cfg=_cfg())
    for ep in dataset:
        for step in ep.steps:
            assert "obs" in step
            assert "expert_action" in step
            assert "ref_des" in step


def test_dataset_obs_has_required_keys():
    design = _small_design(n_comps=2)
    dataset = BCDataset(board_paths=[design], cfg=_cfg())
    step = dataset.all_steps()[0]
    obs = step["obs"]
    assert "z_star" in obs
    assert "Z_placed" in obs
    assert "F_pair" in obs
    assert "grid_xy" in obs
    assert "action_mask" in obs


def test_infeasible_cell_skipped():
    """If all cells are pre-occupied (degenerate board), steps should be skipped."""
    # Create a 1x1 board at 1mm resolution → only 1 cell.
    design = _small_design(n_comps=3, board_w=1.0, board_h=1.0)
    cfg = _cfg(grid_resolution_mm=1.0)
    dataset = BCDataset(board_paths=[design], cfg=cfg)
    # With only 1 cell, at most 1 step is legal; the rest are skipped.
    all_steps = dataset.all_steps()
    assert len(all_steps) <= 1


def test_multiple_boards_accumulate():
    designs = [_small_design(n_comps=2), _small_design(n_comps=3)]
    dataset = BCDataset(board_paths=designs, cfg=_cfg())
    assert len(dataset) == 2
    assert len(dataset.all_steps()) >= 4  # 2 + 3 steps


# ---------------------------------------------------------------------------
# bc_pretrain tests
# ---------------------------------------------------------------------------

def test_bc_pretrain_reduces_loss():
    """BC training should strictly decrease loss on a tiny dataset."""
    torch.manual_seed(0)
    design = _small_design(n_comps=3)
    cfg = _cfg(epochs=20, lr=5e-3)
    dataset = BCDataset(board_paths=[design], cfg=cfg)
    if len(dataset.all_steps()) == 0:
        pytest.skip("dataset has no valid steps")

    policy = _make_policy()
    initial_loss = eval_bc_loss(policy, dataset, cfg)
    bc_pretrain(policy, dataset, cfg)
    final_loss = eval_bc_loss(policy, dataset, cfg)

    assert final_loss < initial_loss, (
        f"BC training did not reduce loss: initial={initial_loss:.4f} final={final_loss:.4f}"
    )


def test_bc_pretrain_empty_dataset_no_crash():
    """Empty dataset should log a warning and return without crashing."""
    policy = _make_policy()
    cfg = _cfg()
    # Board with no positions → empty dataset.
    design = _small_design(n_comps=0)
    dataset = BCDataset(board_paths=[design], cfg=cfg)
    bc_pretrain(policy, dataset, cfg)  # Should not raise.


def test_eval_bc_loss_returns_finite():
    design = _small_design(n_comps=3)
    cfg = _cfg()
    dataset = BCDataset(board_paths=[design], cfg=cfg)
    policy = _make_policy()
    loss = eval_bc_loss(policy, dataset, cfg)
    if dataset.all_steps():
        assert torch.isfinite(torch.tensor(loss))
