"""Tests for the scpt.rust_bridge re-exports."""
from __future__ import annotations

from pathlib import Path

from scpt.rust_bridge import parser_client, router_client

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"


def test_parser_client_reexports_load_kicad_pcb():
    assert callable(parser_client.load_kicad_pcb)
    result = parser_client.load_kicad_pcb(str(FIXTURE))
    assert isinstance(result, str)


def test_parser_client_reexports_geometry():
    assert callable(parser_client.hpwl)
    assert callable(parser_client.hpwl_incremental)
    assert callable(parser_client.clearance_cost)


def test_router_client_reexports_route():
    assert callable(router_client.route)
