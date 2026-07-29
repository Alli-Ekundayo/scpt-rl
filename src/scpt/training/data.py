"""Union-Find functional clustering and BC/RL placement ordering.

Two jobs:
1. `functional_clusters(design)` — union components that share a non-power,
   non-ground net. Power/ground nets are excluded because they connect
   everything and would collapse the clustering to one blob.
2. `area_descending_cluster_order(design, cluster_ids)` — deterministic
   placement order used by both BC pretraining and RL rollouts. Matching
   the two is what makes BC warm-start transfer.

Both functions operate on the parsed SCPT IR (a dict from `load_kicad_pcb`).
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Weighted quick-union with path compression."""

    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


def functional_clusters(design: dict[str, Any]) -> list[int]:
    """Return cluster id (0..n-1) per component.

    Two components union iff they share a net that is neither Power nor Ground.
    Components that share no qualifying net stay in their own singleton cluster.
    Cluster ids are the find-root of each component, so identical ids mean
    same cluster.
    """
    components = design["components"]
    n = len(components)
    uf = _UnionFind(n)
    for net in design.get("nets", []):
        role = net.get("role", "signal")
        if role in ("power", "ground"):
            continue
        # Union all components referenced by this net's pads.
        comp_indices = {pad_ref[0] for pad_ref in net.get("pads", [])}
        comp_indices_list = list(comp_indices)
        for i in range(1, len(comp_indices_list)):
            uf.union(comp_indices_list[0], comp_indices_list[i])
    return [uf.find(i) for i in range(n)]


# ---------------------------------------------------------------------------
# BC/RL ordering
# ---------------------------------------------------------------------------

def _courtyard_area(component: dict[str, Any]) -> float:
    """Unsigned area of a polygon's point list via the shoelace formula."""
    pts = component.get("footprint", {}).get("courtyard", {}).get("points", [])
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x_i, y_i = pts[i] if isinstance(pts[i], (list, tuple)) else (pts[i]["x"], pts[i]["y"])
        j = (i + 1) % n
        x_j, y_j = pts[j] if isinstance(pts[j], (list, tuple)) else (pts[j]["x"], pts[j]["y"])
        s += x_i * y_j
        s -= x_j * y_i
    return abs(s) * 0.5


def _cluster_total_area(cluster_ids: list[int], component_areas: list[float]) -> dict[int, float]:
    """Sum of courtyard area per cluster."""
    totals: dict[int, float] = {}
    for cid, area in zip(cluster_ids, component_areas):
        totals[cid] = totals.get(cid, 0.0) + area
    return totals


def area_descending_cluster_order(
    design: dict[str, Any],
    cluster_ids: list[int],
) -> list[int]:
    """Cluster-major, area-minor ordering.

    Determinism (same input → same order across runs):
    - Clusters sorted by total courtyard area descending; ties by cluster id ascending.
    - Within a cluster, components sorted by courtyard area descending; ties by
      ref_des ascending (lexicographic).
    """
    components = design["components"]
    comp_areas = [_courtyard_area(c) for c in components]
    cluster_totals = _cluster_total_area(cluster_ids, comp_areas)

    # Build per-cluster component lists.
    per_cluster: dict[int, list[int]] = {}
    for i, cid in enumerate(cluster_ids):
        per_cluster.setdefault(cid, []).append(i)

    # Sort clusters by total area desc, then cid asc.
    sorted_cids = sorted(
        per_cluster.keys(),
        key=lambda cid: (-cluster_totals[cid], cid),
    )

    order: list[int] = []
    for cid in sorted_cids:
        members = per_cluster[cid]
        # Within cluster: area desc, then ref_des asc.
        members.sort(key=lambda i: (-comp_areas[i], components[i]["ref_des"]))
        order.extend(members)
    return order


def find_symmetry_pairs(
    design: dict[str, Any],
    cluster_ids: list[int],
) -> list[tuple[str, str]]:
    """Return pairs of components that should be symmetric:
    - diff-pair partners (from nets with `diff_pair_id` set)
    - matched resistor/capacitor pairs in the same cluster (same value prefix
      + same cluster, e.g. R1+R2, C3+C4)
    """
    seen_pairs: set[tuple[str, str]] = set()

    # Diff pair partners
    net_by_name: dict[str, dict] = {n["name"]: n for n in design.get("nets", [])}
    diff_partners: dict[str, str] = {}
    for net in design.get("nets", []):
        partner = net.get("diff_pair_id")
        if partner:
            diff_partners[net["name"]] = partner

    # Find components on diff-pair nets.
    comp_ref_to_idx: dict[str, int] = {
        c["ref_des"]: i for i, c in enumerate(design["components"])
    }
    comp_idx_to_ref: dict[int, str] = {
        i: c["ref_des"] for i, c in enumerate(design["components"])
    }
    for net_name, partner_name in diff_partners.items():
        if partner_name not in net_by_name:
            continue
        pads_a = {ref for ref, _ in net_by_name[net_name].get("pads", [])}
        pads_b = {ref for ref, _ in net_by_name[partner_name].get("pads", [])}
        # Pair up components on the P-side with components on the N-side.
        # Diff-pair relationships override cluster boundaries — even if R1
        # and R2 are in different functional clusters, if they carry CLK_P
        # and CLK_N respectively they should be mirrored.
        for ci in pads_a:
            for cj in pads_b:
                if ci in comp_idx_to_ref and cj in comp_idx_to_ref:
                    ra, rb = comp_idx_to_ref[ci], comp_idx_to_ref[cj]
                    pair = tuple(sorted((ra, rb)))
                    seen_pairs.add(pair)

    # Matched R/C pairs in same cluster
    from collections import defaultdict
    by_cluster_prefix: dict[tuple[int, str], list[str]] = defaultdict(list)
    for i, comp in enumerate(design["components"]):
        ref = comp["ref_des"]
        # Extract prefix letter(s): R1 → R, C12 → C, U3 → U
        prefix = ""
        for ch in ref:
            if ch.isalpha():
                prefix += ch
            else:
                break
        if prefix in ("R", "C"):  # only resistors + capacitors for v1
            by_cluster_prefix[(cluster_ids[i], prefix)].append(ref)
    for (cid, _), refs in by_cluster_prefix.items():
        refs_sorted = sorted(refs)
        # Pair them up in order: (R1, R2), (R3, R4), etc.
        for i in range(0, len(refs_sorted) - 1, 2):
            pair = tuple(sorted((refs_sorted[i], refs_sorted[i + 1])))
            seen_pairs.add(pair)

    return [tuple(p) for p in sorted(seen_pairs)]
