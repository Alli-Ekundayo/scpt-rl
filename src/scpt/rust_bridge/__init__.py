"""Rust bridge: thin re-export wrappers around the PyO3 wheels.

Per the design spec's discipline: "the bridge is a boundary, not a layer."
No logic, no caching, no transformation. Every function imported here is a
direct re-export from a compiled wheel (`pcb_parser`, `pcb_router`). If you
need to transform or cache, that logic belongs in `env/` or `training/`,
not here.
"""
from . import parser_client, router_client

__all__ = ["parser_client", "router_client"]
