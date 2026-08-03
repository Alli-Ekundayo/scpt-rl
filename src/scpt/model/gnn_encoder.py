"""GNN encoder for PCB designs (v1: MLP stand-in for heterogeneous PyG encoder).

The spec calls for a 3-layer heterogeneous GraphSAGE / HAN-style encoder over
component / pad / net node types. v1 uses a simpler MLP-per-type architecture:
each node type's feature vector is projected via a type-specific linear layer
to the shared hidden dimension.

Why MLP first:
- No torch-geometric dep (keeps the env wheel small and the CI fast).
- Same output shape as the real HAN encoder: dict[str, Tensor] keyed by
  node type, with tensor shape (N_type, hidden).
- Upgrade path is clean: swap `HeteroPCBEncoder` for a real `HeteroConv`
  stack when torch-geometric is added; downstream code (SCPTPolicy,
  ValueHeads) doesn't change because the dict shape is the same.

Feature construction lives outside this module — `build_node_features()` is
a pure function that takes a `PcbDesign` dict and returns feature tensors.
Keeping construction separate from the encoder makes it easy to test the
feature math independently.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from scpt.training.data import build_pair_features


class HeteroPCBEncoder(nn.Module):
    """MLP-per-node-type encoder (v1; real heterogeneous message passing
    is a drop-in replacement — same interface).

    Args:
        node_dims: dict mapping node type name → input feature dimension.
        hidden: shared hidden dimension (output for all node types).
    """
    def __init__(self, node_dims: dict[str, int], hidden: int = 256):
        super().__init__()
        self.node_dims = dict(node_dims)
        self.hidden = hidden
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            for name, in_dim in node_dims.items()
        })

    def forward(self, node_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode each node type independently.

        Args:
            node_features: dict mapping node type name → (N_type, in_dim) tensor.

        Returns:
            dict mapping node type name → (N_type, hidden) tensor.
        """
        out: dict[str, torch.Tensor] = {}
        for name, feats in node_features.items():
            if name not in self.projections:
                raise KeyError(
                    f"node type '{name}' not registered; expected one of "
                    f"{list(self.projections.keys())}"
                )
            out[name] = self.projections[name](feats)
        return out


# ---------------------------------------------------------------------------
# Feature construction (pure function, no model weights)
# ---------------------------------------------------------------------------

def build_node_features(
    design: dict[str, Any],
    hidden_for_placed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build per-node-type feature tensors from a SCPT PcbDesign dict.

    v1 features:
    - component: placed flag (1), courtyard area, pad count, current x/y if placed (else 0/0).
    - pad: local position (2), net_name hash (1 float), electrical_proxy_confidence (1).
    - net: role one-hot (4: signal/power/ground/unknown), pad count, role_confidence.

    The `hidden_for_placed` argument is for the policy: when called after the
    encoder has run, you can pass the per-component z_comp and use it as an
    additional feature for downstream use. v1: ignored (reserved for v2).

    Returns:
        dict with keys "component", "pad", "net" mapping to (N_type, feat_dim) tensors.
    """
    components = design.get("components", [])
    nets = design.get("nets", [])
    placement = design.get("placement", {})
    positions = placement.get("positions", [None] * len(components))

    # --- Component features ---
    comp_feats = []
    total_pads = 0
    for i, comp in enumerate(components):
        placed = positions[i] is not None if i < len(positions) else False
        courtyard_area = _polygon_area(comp.get("footprint", {}).get("courtyard", {}))
        pads = comp.get("footprint", {}).get("pads", [])
        n_pads = len(pads)
        total_pads += n_pads
        if placed:
            x, y = positions[i]["position"]
        else:
            x, y = 0.0, 0.0
        comp_feats.append([1.0 if placed else 0.0, courtyard_area, float(n_pads), x, y])
    comp_tensor = torch.tensor(comp_feats, dtype=torch.float32) if comp_feats else torch.zeros((0, 5))

    # --- Pad features ---
    # Flatten across all components. Each pad has: local_pos (2),
    # net_name hash (1), electrical_proxy_confidence (1) → 4-dim.
    pad_feats = []
    for comp in components:
        for pad in comp.get("footprint", {}).get("pads", []):
            lx, ly = pad.get("local_pos", (0.0, 0.0))
            net_hash = _hash_float(pad.get("net_name", ""))
            conf = pad.get("electrical_proxy_confidence", 0.1)
            pad_feats.append([lx, ly, net_hash, conf])
    pad_tensor = torch.tensor(pad_feats, dtype=torch.float32) if pad_feats else torch.zeros((0, 4))

    # --- Net features ---
    # role one-hot (signal, power, ground, unknown), pad count, confidence → 6-dim.
    role_to_idx = {"signal": 0, "power": 1, "ground": 2, "unknown": 3}
    net_feats = []
    for net in nets:
        role = net.get("role", "unknown")
        one_hot = [0.0, 0.0, 0.0, 0.0]
        idx = role_to_idx.get(role, 3)
        one_hot[idx] = 1.0
        n_pads = len(net.get("pads", []))
        conf = net.get("role_confidence", 0.1)
        net_feats.append(one_hot + [float(n_pads), conf])
    net_tensor = torch.tensor(net_feats, dtype=torch.float32) if net_feats else torch.zeros((0, 6))

    return {
        "component": comp_tensor,
        "pad": pad_tensor,
        "net": net_tensor,
    }


def encode_design(
    design: dict[str, Any],
    encoder: HeteroPCBEncoder,
    active_idx: int,
    placed_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a design into component embeddings for policy/value use.

    Returns:
        z_comp_all: (N, d) component embeddings for the whole design.
        z_star: (d,) embedding for the active component.
        Z_placed: (P, d) embeddings for already placed components.
    """
    device = next(encoder.parameters()).device
    node_features = {
        name: feats.to(device=device)
        for name, feats in build_node_features(design).items()
    }
    encoded = encoder(node_features)
    z_comp_all = encoded["component"]

    if z_comp_all.shape[0] == 0:
        hidden = encoder.hidden
        empty = torch.zeros(hidden, dtype=z_comp_all.dtype, device=z_comp_all.device)
        return z_comp_all, empty, z_comp_all.new_zeros((0, hidden))

    if active_idx < 0 or active_idx >= z_comp_all.shape[0]:
        z_star = torch.zeros(z_comp_all.shape[-1], dtype=z_comp_all.dtype, device=z_comp_all.device)
    else:
        z_star = z_comp_all[active_idx]

    if placed_indices:
        valid = [idx for idx in placed_indices if 0 <= idx < z_comp_all.shape[0]]
        Z_placed = z_comp_all[valid] if valid else z_comp_all.new_zeros((0, z_comp_all.shape[-1]))
    else:
        Z_placed = z_comp_all.new_zeros((0, z_comp_all.shape[-1]))

    return z_comp_all, z_star, Z_placed


def _polygon_area(courtyard: dict[str, Any]) -> float:
    """Unsigned area via shoelace."""
    pts = courtyard.get("points", [])
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        x_i, y_i = pts[i] if isinstance(pts[i], (list, tuple)) else (pts[i]["x"], pts[i]["y"])
        x_j, y_j = pts[j] if isinstance(pts[j], (list, tuple)) else (pts[j]["x"], pts[j]["y"])
        s += x_i * y_j - x_j * y_i
    return abs(s) * 0.5


def _hash_float(s: str) -> float:
    """Deterministic string → float in [0, 1] for feature hashing."""
    return (hash(s) % 10_000) / 10_000.0
