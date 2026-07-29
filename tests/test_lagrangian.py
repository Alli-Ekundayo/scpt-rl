"""Tests for MomentumDualUpdater."""
from __future__ import annotations

import pytest

from scpt.agent.lagrangian import MomentumDualUpdater


def test_lambda_monotone_under_positive_phi():
    """Consistently violated constraint → lambda climbs monotonically."""
    upd = MomentumDualUpdater(["c_hpwl"], alpha=0.01, ema_decay=0.9)
    vals = []
    for _ in range(10):
        out = upd.update({"c_hpwl": 0.5})
        vals.append(out["c_hpwl"])
    for i in range(len(vals) - 1):
        assert vals[i + 1] >= vals[i], f"lambda should climb: {vals}"


def test_lambda_clamps_to_zero_under_negative_phi():
    """Satisfied constraint → lambda stays at 0, never goes negative."""
    upd = MomentumDualUpdater(["c_hpwl"], alpha=0.01, ema_decay=0.9)
    # Pre-build some positive lambda.
    upd.update({"c_hpwl": 1.0})
    upd.update({"c_hpwl": 1.0})
    # Now consistently violate in the other direction.
    for _ in range(20):
        out = upd.update({"c_hpwl": -1.0})
        assert out["c_hpwl"] >= 0.0, "lambda must stay non-negative"


def test_ema_smoothing():
    """Constant phi_c should cause increasing lambda (via EMA buildup)."""
    upd = MomentumDualUpdater(["c_hpwl"], alpha=0.1, ema_decay=0.9)
    a = upd.update({"c_hpwl": 1.0})["c_hpwl"]
    b = upd.update({"c_hpwl": 1.0})["c_hpwl"]
    c = upd.update({"c_hpwl": 1.0})["c_hpwl"]
    # Each step adds a positive increment. With constant phi and EMA, the
    # increment itself grows (EMA rises toward phi), so lambda should rise
    # at an increasing rate initially.
    assert b > a
    assert c > b


def test_zero_alpha_means_lambda_unchanged():
    upd = MomentumDualUpdater(["c_hpwl"], alpha=0.0, ema_decay=0.9)
    upd.update({"c_hpwl": 0.5})  # build EMA
    out = upd.update({"c_hpwl": 0.5})
    assert out["c_hpwl"] == 0.0


def test_missing_constraint_name_defaults_to_zero():
    """If phi_c_by_name is missing a constraint, treat phi as 0."""
    upd = MomentumDualUpdater(["a", "b"], alpha=0.01, ema_decay=0.9)
    # Only provide "a" — "b" should not blow up or error.
    out = upd.update({"a": 0.5})
    assert out["b"] == 0.0


def test_multiple_constraints_independent():
    upd = MomentumDualUpdater(["a", "b"], alpha=0.01, ema_decay=0.9)
    out = upd.update({"a": 1.0, "b": -1.0})
    assert out["a"] > 0.0
    assert out["b"] == 0.0  # negative phi → clamped


def test_lambdas_property_is_snapshot():
    upd = MomentumDualUpdater(["a"], alpha=0.1, ema_decay=0.9)
    upd.update({"a": 1.0})
    snap1 = upd.lambdas
    upd.update({"a": 1.0})
    snap2 = upd.lambdas
    assert snap1["a"] < snap2["a"]
    # Mutating the snapshot shouldn't affect the updater's state.
    snap1["a"] = 999.0
    assert upd.lambdas["a"] != 999.0
