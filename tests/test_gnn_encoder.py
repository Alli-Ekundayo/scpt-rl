"""Tests for GNN encoder + feature construction."""
from __future__ import annotations

import torch

from scpt.model.gnn_encoder import HeteroPCBEncoder, build_node_features


def _make_design():
    """Minimal design dict for feature construction tests."""
    return {
        "components": [
            {
                "ref_des": "R1",
                "footprint": {
                    "pads": [
                        {"local_pos": [-0.5, 0.0], "net_name": "N1",
                         "electrical_proxy_confidence": 1.0},
                        {"local_pos": [0.5, 0.0], "net_name": "N1",
                         "electrical_proxy_confidence": 1.0},
                    ],
                    "courtyard": {"points": [[-0.8, -0.4], [0.8, -0.4], [0.8, 0.4], [-0.8, 0.4]]},
                    "silkscreen": [],
                },
                "value": "10k",
                "netclass_hint": None,
            },
            {
                "ref_des": "C1",
                "footprint": {
                    "pads": [
                        {"local_pos": [-0.3, 0.0], "net_name": "N1",
                         "electrical_proxy_confidence": 1.0},
                        {"local_pos": [0.3, 0.0], "net_name": "GND",
                         "electrical_proxy_confidence": 1.0},
                    ],
                    "courtyard": {"points": [[-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5]]},
                    "silkscreen": [],
                },
                "value": "100nF",
                "netclass_hint": None,
            },
        ],
        "nets": [
            {"name": "N1", "pads": [[0, 0], [0, 1], [1, 0]], "netclass": "Default",
             "role": "signal", "role_confidence": 0.5, "diff_pair_id": None},
            {"name": "GND", "pads": [[1, 1]], "netclass": "Power",
             "role": "ground", "role_confidence": 1.0, "diff_pair_id": None},
        ],
        "board": {"bounds": {"x": 0, "y": 0, "w": 10, "h": 10}, "outline": {}, "keepouts": []},
        "placement": {
            "positions": [
                {"component_idx": 0, "position": [5.0, 5.0], "rotation_deg": 0.0, "bottom_layer": False},
                None,
            ],
            "placement_order": [1, 0],
        },
    }


# ---------------------------------------------------------------------------
# build_node_features
# ---------------------------------------------------------------------------

def test_build_features_shape():
    d = _make_design()
    feats = build_node_features(d)
    assert "component" in feats
    assert "pad" in feats
    assert "net" in feats
    assert feats["component"].shape == (2, 5)
    assert feats["pad"].shape == (4, 4)
    assert feats["net"].shape == (2, 6)


def test_component_features_include_placed_flag():
    d = _make_design()
    feats = build_node_features(d)
    # Component 0 is placed, component 1 is not
    assert feats["component"][0, 0] == 1.0
    assert feats["component"][1, 0] == 0.0


def test_pad_features_include_local_pos():
    d = _make_design()
    feats = build_node_features(d)
    # First pad of R1: local_pos = (-0.5, 0.0)
    assert abs(feats["pad"][0, 0].item() - (-0.5)) < 1e-6
    assert abs(feats["pad"][0, 1].item() - 0.0) < 1e-6


def test_net_features_role_onehot():
    d = _make_design()
    feats = build_node_features(d)
    # N1 is signal → one-hot index 0
    # GND is ground → one-hot index 2
    n1_feats = feats["net"][0]
    gnd_feats = feats["net"][1]
    assert n1_feats[0] == 1.0 and n1_feats[1] == 0.0  # signal
    assert gnd_feats[2] == 1.0 and gnd_feats[0] == 0.0  # ground


def test_empty_design():
    d = {"components": [], "nets": [], "placement": {"positions": []}}
    feats = build_node_features(d)
    assert feats["component"].shape == (0, 5)
    assert feats["pad"].shape == (0, 4)
    assert feats["net"].shape == (0, 6)


# ---------------------------------------------------------------------------
# HeteroPCBEncoder
# ---------------------------------------------------------------------------

def test_encoder_output_shapes():
    enc = HeteroPCBEncoder(node_dims={"component": 5, "pad": 4, "net": 6}, hidden=32)
    feats = build_node_features(_make_design())
    out = enc(feats)
    assert out["component"].shape == (2, 32)
    assert out["pad"].shape == (4, 32)
    assert out["net"].shape == (2, 32)


def test_encoder_gradients_flow():
    enc = HeteroPCBEncoder(node_dims={"component": 5, "pad": 4}, hidden=16)
    feats = {
        "component": torch.randn(2, 5, requires_grad=True),
        "pad": torch.randn(3, 4, requires_grad=True),
    }
    out = enc(feats)
    loss = out["component"].sum() + out["pad"].sum()
    loss.backward()
    assert feats["component"].grad is not None
    assert feats["pad"].grad is not None


def test_encoder_unknown_node_type_raises():
    enc = HeteroPCBEncoder(node_dims={"component": 5}, hidden=16)
    feats = {"unknown": torch.randn(2, 3)}
    try:
        enc(feats)
        assert False, "should have raised"
    except KeyError:
        pass
