"""Tests for functional clustering and BC/RL ordering."""
from __future__ import annotations

import pytest

from scpt.training.data import (
    area_descending_cluster_order,
    build_pair_features,
    find_symmetry_pairs,
    functional_clusters,
)


def _make_design(components, nets):
    """Minimal design dict matching SCPT IR shape."""
    return {
        "components": components,
        "nets": nets,
        "board": {"bounds": {"x": 0, "y": 0, "w": 100, "h": 100}, "outline": {}, "keepouts": []},
        "netclasses": {},
        "diff_pairs": [],
        "placement": {"positions": [None] * len(components), "placement_order": []},
    }


def _comp(ref_des, courtyard_points):
    return {
        "ref_des": ref_des,
        "footprint": {"pads": [], "courtyard": {"points": courtyard_points}, "silkscreen": []},
        "value": ref_des,  # reuse ref_des as value for simplicity
        "netclass_hint": None,
    }


def _net(name, role, pads, diff_pair_id=None):
    return {"name": name, "pads": pads, "netclass": "Default", "role": role,
            "role_confidence": 1.0, "diff_pair_id": diff_pair_id}


# ---------------------------------------------------------------------------
# Functional clusters
# ---------------------------------------------------------------------------

def test_components_sharing_signal_net_cluster_together():
    # R1 and C1 share net N1 (signal) → same cluster
    design = _make_design(
        components=[_comp("R1", [(0, 0), (1, 0), (1, 1), (0, 1)]),
                    _comp("C1", [(0, 0), (1, 0), (1, 1), (0, 1)])],
        nets=[_net("N1", "signal", [(0, 0), (1, 0)])],
    )
    ids = functional_clusters(design)
    assert ids[0] == ids[1]


def test_power_net_does_not_merge_clusters():
    # R1—N1—C1 (signal) AND R1—VCC—R2 (power).
    # VCC is Power → excluded. N1 still clusters R1+C1. R2 is unconnected via signal.
    design = _make_design(
        components=[_comp("R1", [(0, 0), (1, 0), (1, 1), (0, 1)]),
                    _comp("C1", [(0, 0), (1, 0), (1, 1), (0, 1)]),
                    _comp("R2", [(0, 0), (1, 0), (1, 1), (0, 1)])],
        nets=[
            _net("N1", "signal", [(0, 0), (1, 0)]),   # R1-C1
            _net("VCC", "power", [(0, 1), (2, 0)]),   # R1-R2 (excluded)
        ],
    )
    ids = functional_clusters(design)
    assert ids[0] == ids[1], "R1 and C1 share signal net"
    assert ids[2] != ids[0], "R2 is isolated (VCC is power, excluded)"


def test_unconnected_components_in_singleton_clusters():
    # 3 components, no nets → all different clusters
    design = _make_design(
        components=[_comp("R1", []), _comp("C1", []), _comp("U1", [])],
        nets=[],
    )
    ids = functional_clusters(design)
    assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# Area-descending cluster-contiguous ordering
# ---------------------------------------------------------------------------

def test_ordering_is_cluster_contiguous():
    # Two clusters: {R1, C1} and {R2}. Larger cluster placed first.
    design = _make_design(
        components=[
            _comp("R1", [(0, 0), (1, 0), (1, 1), (0, 1)]),    # area 1
            _comp("C1", [(0, 0), (2, 0), (2, 2), (0, 2)]),    # area 4
            _comp("R2", [(0, 0), (3, 0), (3, 3), (0, 3)]),    # area 9 (singleton)
        ],
        nets=[_net("N1", "signal", [(0, 0), (1, 0)])],  # R1-C1
    )
    ids = functional_clusters(design)
    order = area_descending_cluster_order(design, ids)
    # Cluster of R1+C1 has total area 5; R2 singleton has area 9.
    # So R2's cluster comes first.
    assert order[0] == 2  # R2 is first
    # R1 and C1 are contiguous.
    assert set(order[1:]) == {0, 1}


def test_tiebreaking_by_ref_des():
    # Two components with identical courtyard area, same cluster → sorted by ref_des.
    design = _make_design(
        components=[
            _comp("R2", [(0, 0), (1, 0), (1, 1), (0, 1)]),
            _comp("R1", [(0, 0), (1, 0), (1, 1), (0, 1)]),
        ],
        nets=[_net("N1", "signal", [(0, 0), (1, 0)])],
    )
    ids = functional_clusters(design)
    order = area_descending_cluster_order(design, ids)
    # R1 < R2 lexicographically → R1 first.
    assert order == [1, 0]  # component index 1 is R1, index 0 is R2


# ---------------------------------------------------------------------------
# Symmetry pairs
# ---------------------------------------------------------------------------

def test_symmetry_pairs_includes_diff_pair_partners():
    # Two components each on a diff-pair net → should be paired.
    design = _make_design(
        components=[
            _comp("R1", []),
            _comp("R2", []),
        ],
        nets=[
            _net("CLK_P", "signal", [(0, 0)], diff_pair_id="CLK_N"),
            _net("CLK_N", "signal", [(1, 0)], diff_pair_id="CLK_P"),
        ],
    )
    ids = functional_clusters(design)
    pairs = find_symmetry_pairs(design, ids)
    # R1 and R2 should be paired.
    assert ("R1", "R2") in pairs or ("R2", "R1") in pairs


def test_symmetry_pairs_includes_matched_rc():
    # R1 + R2 in same cluster → should pair.
    design = _make_design(
        components=[
            _comp("R1", []),
            _comp("R2", []),
            _comp("U1", []),
        ],
        nets=[_net("N1", "signal", [(0, 0), (1, 0)])],  # clusters R1+R2
    )
    ids = functional_clusters(design)
    pairs = find_symmetry_pairs(design, ids)
    assert ("R1", "R2") in pairs


def test_build_pair_features_returns_14_dims():
    design = _make_design(
        components=[
            _comp("U1", []),
            _comp("U2", []),
        ],
        nets=[
            _net("SIG", "signal", [(0, 0), (1, 0)]),
        ],
    )
    design["placement"]["positions"] = [
        {"component_idx": 0, "position": [10.0, 10.0], "rotation_deg": 0.0, "bottom_layer": False},
        {"component_idx": 1, "position": [13.0, 14.0], "rotation_deg": 0.0, "bottom_layer": False},
    ]

    feats = build_pair_features(design, active_idx=0, placed_indices=[1])
    assert feats.shape == (1, 14)
    assert feats[0, 0].item() == 1.0
    assert feats[0, 1].item() == 1.0
    assert feats[0, 9].item() > 0.0
