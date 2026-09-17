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


def polygon_area(points: list) -> float:
    """Unsigned area of a simple polygon via the shoelace formula.
    
    Points can be [x, y] lists/tuples or {"x": ..., "y": ...} dicts.
    Returns 0.0 for degenerate polygons (< 3 points).
    """
    if len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = points[i] if isinstance(points[i], (list, tuple)) else (points[i]["x"], points[i]["y"])
        x_j, y_j = points[j] if isinstance(points[j], (list, tuple)) else (points[j]["x"], points[j]["y"])
        s += x_i * y_j - x_j * y_i
    return abs(s) * 0.5


def validate_cfg(cfg: SimpleNamespace, required_paths: list[str]) -> None:
    """Validate that all required dot-separated paths exist in a SimpleNamespace.
    
    Raises AttributeError with a clear message if any field is missing.
    Example: validate_cfg(cfg, ["ppo.clip_eps", "ppo.gamma", "model.d"])
    """
    for path in required_paths:
        obj = cfg
        for part in path.split("."):
            if not hasattr(obj, part):
                raise AttributeError(
                    f"Config missing required field '{path}' "
                    f"(failed at '{part}'). Check your YAML config file."
                )
            obj = getattr(obj, part)
