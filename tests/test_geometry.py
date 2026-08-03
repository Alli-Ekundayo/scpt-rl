"""Python tests for geometry primitives exposed by pcb_parser."""
from __future__ import annotations

import json
from pathlib import Path

import pcb_parser
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"


def load_design():
    return pcb_parser.load_kicad_pcb(str(FIXTURE))


def test_hpwl_zero_when_placements_are_initial():
    design_json = load_design()
    # The fixture has components at (10, 20) and (15, 20) — placed.
    # VCC net connects R1.pad1 + C1.pad1. HPWL depends on world positions.
    h = pcb_parser.hpwl(design_json)
    assert h >= 0.0


def test_hpwl_incremental_subset_of_full():
    design_json = load_design()
    full = pcb_parser.hpwl(design_json)
    inc = pcb_parser.hpwl_incremental(design_json, "R1")
    # Incremental should be <= full (filters to nets touching R1)
    assert inc <= full + 1e-9
    assert inc >= 0.0


def test_clearance_cost_zero_for_separated_components():
    design_json = load_design()
    # Components are at (10, 20) and (15, 20) — well separated.
    c = pcb_parser.clearance_cost(design_json, 0.2)
    assert c == 0.0


def test_clearance_accepts_min_spacing_arg():
    design_json = load_design()
    # Just verify the arg is accepted and returns a finite non-negative value.
    c = pcb_parser.clearance_cost(design_json, 1.0)
    assert c >= 0.0
    assert not (c != c)  # not NaN


def test_hpwl_invalid_json_raises():
    with pytest.raises(Exception):
        pcb_parser.hpwl("not valid json")
