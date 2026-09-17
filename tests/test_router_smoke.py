"""Tests for the pcb_router wheel."""
from __future__ import annotations

import json
from pathlib import Path

import pcb_parser
import pcb_router
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"


def test_route_function_exists():
    assert callable(pcb_router.route)


def test_route_accepts_scpt_design_json():
    design_json = pcb_parser.load_kicad_pcb(str(FIXTURE))
    # Should not raise — even if the routing result isn't serialized back yet
    result = pcb_router.route(design_json)
    assert isinstance(result, str)


def test_route_returns_valid_json():
    design_json = pcb_parser.load_kicad_pcb(str(FIXTURE))
    result = pcb_router.route(design_json)
    # v1: returns the input unchanged; still valid JSON
    parsed = json.loads(result)
    assert "components" in parsed
    assert "nets" in parsed


def test_route_invalid_json_raises():
    with pytest.raises(Exception):
        pcb_router.route("not valid json")


def test_route_enriches_json_with_routes_and_stats():
    design_json = pcb_parser.load_kicad_pcb(str(FIXTURE))
    result = pcb_router.route(design_json, num_iterations=1, grid_scale=2)
    parsed = json.loads(result)
    assert "routes" in parsed
    routes = parsed["routes"]
    assert "segments" in routes
    assert "vias" in routes
    assert "stats" in routes
    stats = routes["stats"]
    assert "completion_rate" in stats
    assert "wirelength_mm" in stats
    assert "num_vias" in stats
    assert stats["completion_rate"] >= 0.0

