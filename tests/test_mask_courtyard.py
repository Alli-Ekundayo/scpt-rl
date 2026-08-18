"""Tests for _compute_mask courtyard-based exclusion zone.

These tests bypass pcb_parser (Rust wheel) by injecting a mock module before
pcb_env is imported.  The key regression guarded:
  comp.get('bounds') is ALWAYS None — the IR has no top-level 'bounds' field.
  The mask must use footprint.courtyard.points AABB, not the phantom key.
"""
from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub out pcb_parser before pcb_env is imported
# ---------------------------------------------------------------------------

def _make_pcb_parser_stub():
    m = types.ModuleType("pcb_parser")
    m.load_kicad_pcb = lambda *a, **kw: "{}"
    m.clearance_cost = lambda *a, **kw: 0.0
    m.hpwl = lambda *a, **kw: 0.0
    m.hpwl_incremental = lambda *a, **kw: 0.0
    return m


if "pcb_parser" not in sys.modules:
    sys.modules["pcb_parser"] = _make_pcb_parser_stub()

from scpt.env.pcb_env import EnvConfig, PcbPlacementEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_component(ref_des: str, courtyard_w: float, courtyard_h: float) -> dict:
    """Component with a rectangular courtyard of given dimensions (mm)."""
    hw, hh = courtyard_w / 2.0, courtyard_h / 2.0
    return {
        "ref_des": ref_des,
        "value": "X",
        "netclass_hint": None,
        "footprint": {
            "pads": [],
            "courtyard": {
                "points": [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
            },
            "silkscreen": [],
        },
    }


def _make_design(comps: list[dict], positions: list[dict | None]) -> dict:
    return {
        "board": {
            "bounds": {"x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0},
            "outline": {"points": []},
            "keepouts": [],
        },
        "components": comps,
        "nets": [],
        "netclasses": [],
        "diff_pairs": [],
        "placement": {
            "positions": positions,
            "placement_order": list(range(len(comps))),
        },
    }


def _placed(x: float, y: float, idx: int = 0) -> dict:
    return {"component_idx": idx, "position": [x, y], "rotation_deg": 0.0, "bottom_layer": False}


class _MockEnv:
    """Calls _compute_mask without a real KiCad file or pcb_parser calls."""

    def __init__(self, design: dict, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig(grid_resolution_mm=0.5, min_spacing_mm=0.2)
        bounds = design["board"]["bounds"]
        res = self.cfg.grid_resolution_mm
        self.W = max(1, int(bounds["w"] / res))
        self.H = max(1, int(bounds["h"] / res))
        self.state = SimpleNamespace(
            design=design,
            step_idx=1,  # comp[1] is being placed next
            placement_order=design["placement"]["placement_order"],
        )

    _compute_mask = PcbPlacementEnv._compute_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mask_blocks_courtyard_sized_region():
    """A 4×2 mm component should block a courtyard+margin region, not just 1 cell."""
    res = 0.5
    margin = max(1, math.ceil(0.2 / res))  # = 1 cell

    comp_w, comp_h = 4.0, 2.0
    design = _make_design(
        comps=[_make_component("U1", comp_w, comp_h), _make_component("R1", 1.0, 0.8)],
        positions=[_placed(10.0, 10.0), None],
    )
    env = _MockEnv(design, EnvConfig(grid_resolution_mm=res, min_spacing_mm=0.2))
    mask = env._compute_mask("R1")

    cx = int(10.0 / res)
    cy = int(10.0 / res)

    # Centre must be masked.
    assert mask[cy * env.W + cx] == 0.0, "Centre of placed component must be masked"

    # One cell beyond half-width + margin should be unmasked.
    half_w = max(1, math.ceil(comp_w / (2.0 * res))) + margin  # = 4 + 1 = 5
    far_x = cx + half_w + 1
    if far_x < env.W:
        assert mask[cy * env.W + far_x] == 1.0, "Cell beyond courtyard+margin should be legal"


def test_mask_uses_courtyard_not_phantom_bounds():
    """Regression: the old code used comp.get('bounds') which is always None.

    With a 4 mm wide component at 0.5 mm resolution, cells 2–4 away from the
    placed centre must be masked (half_w = 4+1 = 5 cells).  If the phantom-
    bounds fallback fires instead, only 1 cell is masked and these cells
    would be incorrectly unmasked.
    """
    res = 0.5
    comp_w = 4.0  # half_w = ceil(4/(2*0.5)) + 1 = 4 + 1 = 5 cells
    design = _make_design(
        comps=[_make_component("U1", comp_w, comp_w), _make_component("R1", 1.0, 1.0)],
        positions=[_placed(10.0, 10.0), None],
    )
    env = _MockEnv(design, EnvConfig(grid_resolution_mm=res, min_spacing_mm=0.2))
    mask = env._compute_mask("R1")

    cx = int(10.0 / res)
    cy = int(10.0 / res)

    for offset in range(1, 5):
        assert mask[cy * env.W + (cx + offset)] == 0.0, (
            f"Cell +{offset} from U1 must be masked — "
            "if not, phantom-bounds fallback is active instead of courtyard AABB"
        )


def test_mask_fallback_conservative_when_no_courtyard():
    """Empty courtyard → conservative 2+margin cell fallback (not just 1 cell)."""
    comp_no_cyd = {
        "ref_des": "U1", "value": "X", "netclass_hint": None,
        "footprint": {"pads": [], "courtyard": {"points": []}, "silkscreen": []},
    }
    design = _make_design(
        comps=[comp_no_cyd, _make_component("R1", 1.0, 1.0)],
        positions=[_placed(10.0, 10.0), None],
    )
    res = 0.5
    env = _MockEnv(design, EnvConfig(grid_resolution_mm=res, min_spacing_mm=0.2))
    mask = env._compute_mask("R1")

    cx = int(10.0 / res)
    cy = int(10.0 / res)
    assert mask[cy * env.W + cx] == 0.0, "Centre must always be masked"
    # Fallback = 2 + margin_cells = 3; cell at +1 should be masked too.
    assert mask[cy * env.W + (cx + 1)] == 0.0, "Adjacent cell must be masked in fallback"


def test_clearance_normalised_by_board_area():
    """Source-level check: _compute_costs must divide raw clearance by board area."""
    import inspect
    from scpt.env import pcb_env
    src = inspect.getsource(pcb_env.PcbPlacementEnv._compute_costs)
    assert "board_area" in src, "_compute_costs must divide by board_area"
    assert "raw_clearance" in src, "_compute_costs must store raw_clearance before dividing"
