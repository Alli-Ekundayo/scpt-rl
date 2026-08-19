"""Common utility functions for SCPT-RL."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import yaml


def dict_to_ns(d: dict[str, Any]) -> SimpleNamespace:
    """Recursively convert nested dicts into SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def ns_to_dict(ns: SimpleNamespace) -> dict[str, Any]:
    """Recursively convert SimpleNamespace back to plain dict for serialization."""
    out = {}
    for k, v in vars(ns).items():
        out[k] = ns_to_dict(v) if isinstance(v, SimpleNamespace) else v
    return out


def load_cfg(path: str) -> SimpleNamespace:
    """Load a YAML config file into a SimpleNamespace."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return dict_to_ns(raw if raw is not None else {})
