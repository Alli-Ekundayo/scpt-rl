"""Round-trip tests: Python -> pcb_parser.load_kicad_pcb -> JSON -> assertions."""
from __future__ import annotations

import json
from pathlib import Path

import pcb_parser

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"


def test_load_kicad_pcb_returns_json_string():
    raw = pcb_parser.load_kicad_pcb(str(FIXTURE))
    assert isinstance(raw, str)
    # Must be valid JSON
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_load_returns_two_components():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    assert len(data["components"]) == 2
    ref_deses = [c["ref_des"] for c in data["components"]]
    assert "R1" in ref_deses
    assert "C1" in ref_deses


def test_component_has_pads():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    r1 = next(c for c in data["components"] if c["ref_des"] == "R1")
    assert len(r1["footprint"]["pads"]) == 2
    assert r1["footprint"]["pads"][0]["shape"] == "rect"


def test_nets_resolve_pad_refs():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    vcc = next(n for n in data["nets"] if n["name"] == "VCC")
    # VCC connects R1.pad1 and C1.pad1 → 2 pad refs
    assert len(vcc["pads"]) == 2
    # Each pad ref is (component_idx, pad_idx)
    for ref in vcc["pads"]:
        assert len(ref) == 2
        assert isinstance(ref[0], int)
        assert isinstance(ref[1], int)


def test_net_role_classification():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    vcc = next(n for n in data["nets"] if n["name"] == "VCC")
    gnd = next(n for n in data["nets"] if n["name"] == "GND")
    sig = next(n for n in data["nets"] if n["name"] == "SIG")
    assert vcc["role"] == "power"
    assert vcc["role_confidence"] == 1.0
    assert gnd["role"] == "ground"
    assert gnd["role_confidence"] == 1.0
    assert sig["role"] == "signal"
    assert sig["role_confidence"] == 0.5  # heuristic-derived


def test_placement_state_populated():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    placements = data["placement"]["positions"]
    assert len(placements) == 2
    # All components start placed (from the KiCad file's (at ...) positions)
    for p in placements:
        assert p is not None
        assert "position" in p
        assert "rotation_deg" in p


def test_placement_order_area_descending():
    data = json.loads(pcb_parser.load_kicad_pcb(str(FIXTURE)))
    order = data["placement"]["placement_order"]
    assert len(order) == 2
    # Deterministic ordering
    assert set(order) == {0, 1}


def test_missing_file_raises():
    import pytest
    with pytest.raises(Exception):  # PyIOError via PyO3
        pcb_parser.load_kicad_pcb("/nonexistent/path.kicad_pcb")
