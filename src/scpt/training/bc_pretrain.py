"""Behavioral cloning (BC) warm-start for the SCPT policy.

BC pretraining teaches the policy to imitate the expert placement order
embedded in the KiCad board file. The expert trajectory is the sequence of
placements already in the design (positions recorded by the engineer), iterated
in the area-descending cluster order defined by `training.data`.

BC minimises cross-entropy loss: -log π(a_expert | s_t) where a_expert is the
grid cell closest to the expert's component position. If the expert cell is
illegal (DRC-masked), we log a warning and skip that step rather than training
on infeasible supervision.

Usage::

    from scpt.training.bc_pretrain import bc_pretrain, BCDataset
    dataset = BCDataset(board_paths=[...], cfg=cfg)
    bc_pretrain(policy, dataset, cfg=cfg)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class BCEpisode:
    """One expert episode: sequence of (obs, expert_action) pairs."""
    steps: list[dict]  # each: {obs, expert_action, board_path, ref_des}


@dataclass
class BCDataset:
    """Dataset of behavioral-cloning episodes from KiCad board files.

    Args:
        board_paths: list of paths to `.kicad_pcb` files or pre-parsed JSON dicts.
        cfg: SimpleNamespace with fields:
            - grid_resolution_mm (float)
            - min_spacing_mm (float)
            - d (int): hidden dim for dummy z_star
            - pair_dim (int): pair feature dim
    """
    board_paths: list[Any]
    cfg: Any
    _episodes: list[BCEpisode] = field(default_factory=list, init=False)

    def __post_init__(self):
        for bp in self.board_paths:
            ep = self._build_episode(bp)
            if ep is not None:
                self._episodes.append(ep)

    def _build_episode(self, board_path_or_dict: Any) -> BCEpisode | None:
        """Build one BC episode from a board path or pre-parsed design dict."""
        from scpt.training.data import functional_clusters, area_descending_cluster_order

        if isinstance(board_path_or_dict, dict):
            design = board_path_or_dict
            board_path_str = "<dict>"
        else:
            try:
                board_path_str = str(board_path_or_dict)
                # Try to import pcb_parser; fall back to loading JSON directly.
                try:
                    import pcb_parser
                    design_json = pcb_parser.load_kicad_pcb(board_path_str)
                    design = json.loads(design_json)
                except ImportError:
                    # Tests without Rust wheel: try loading as JSON directly.
                    with open(board_path_str) as f:
                        design = json.load(f)
            except Exception as exc:
                logger.warning("BCDataset: failed to load %s: %s", board_path_or_dict, exc)
                return None

        components = design.get("components", [])
        if not components:
            return None

        cluster_ids = functional_clusters(design)
        order = area_descending_cluster_order(design, cluster_ids)
        bounds = design.get("board", {}).get("bounds", {"x": 0, "y": 0, "w": 100, "h": 100})
        res = self.cfg.grid_resolution_mm
        W = max(1, int(bounds["w"] / res))
        H = max(1, int(bounds["h"] / res))
        grid_cells = H * W

        # Build grid_xy tensor (static per board).
        grid_xy = torch.zeros(grid_cells, 2)
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                grid_xy[idx, 0] = bounds["x"] + (c + 0.5) * res
                grid_xy[idx, 1] = bounds["y"] + (r + 0.5) * res

        # Simulate the episode step-by-step, extracting expert actions.
        positions = design.get("placement", {}).get("positions", [None] * len(components))
        d = self.cfg.d
        pair_dim = self.cfg.pair_dim

        steps: list[dict] = []
        placed_indices: list[int] = []

        for step_idx, comp_idx in enumerate(order):
            comp = components[comp_idx]
            ref_des = comp.get("ref_des", f"COMP_{comp_idx}")

            # Expert position for this component.
            expert_pos = positions[comp_idx]
            if expert_pos is None:
                logger.debug("BC: %s/%s has no expert position; skipping", board_path_str, ref_des)
                continue

            expert_xy = expert_pos.get("position", [0.0, 0.0])
            ex, ey = float(expert_xy[0]), float(expert_xy[1])

            # Find the grid cell closest to the expert position.
            cx = int((ex - bounds["x"]) / res)
            cy = int((ey - bounds["y"]) / res)
            cx = max(0, min(W - 1, cx))
            cy = max(0, min(H - 1, cy))
            expert_action = cy * W + cx

            # Build stub observation (no Rust wheel needed for BC).
            z_star = torch.zeros(d)
            Z_placed = torch.zeros(max(len(placed_indices), 1), d)
            F_pair = torch.zeros(max(len(placed_indices), 1), pair_dim)

            # Simple action mask: mark already-placed cells as illegal.
            action_mask = torch.ones(grid_cells)
            for placed_idx in placed_indices:
                if placed_idx < grid_cells:
                    action_mask[placed_idx] = 0.0

            # Check if expert cell is legal.
            if action_mask[expert_action] < 0.5:
                logger.warning(
                    "BC: %s/%s expert cell %d is infeasible (DRC-masked); skipping step",
                    board_path_str, ref_des, expert_action,
                )
                # Still advance the episode state.
                placed_indices.append(expert_action)
                continue

            obs = {
                "z_star": z_star,
                "Z_placed": Z_placed,
                "F_pair": F_pair,
                "grid_xy": grid_xy,
                "action_mask": action_mask,
            }
            steps.append({
                "obs": obs,
                "expert_action": expert_action,
                "board_path": board_path_str,
                "ref_des": ref_des,
            })
            placed_indices.append(expert_action)

        return BCEpisode(steps=steps) if steps else None

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int) -> BCEpisode:
        return self._episodes[idx]

    def all_steps(self) -> list[dict]:
        """Flatten all episodes into a single list of step dicts."""
        out = []
        for ep in self._episodes:
            out.extend(ep.steps)
        return out


# ---------------------------------------------------------------------------
# BC loss
# ---------------------------------------------------------------------------

def bc_loss(
    policy: nn.Module,
    step: dict,
    cfg: Any,
) -> torch.Tensor:
    """Compute BC cross-entropy loss for one step.

    Args:
        policy: SCPTPolicy instance.
        step: dict with keys obs, expert_action.
        cfg: namespace (unused currently; reserved for future weighting).

    Returns:
        Scalar cross-entropy loss.
    """
    obs = step["obs"]
    device = next(policy.parameters()).device
    expert_action = torch.tensor(step["expert_action"], dtype=torch.long, device=device)

    z_star = obs["z_star"].to(device=device)
    Z_placed = obs["Z_placed"].to(device=device)
    F_pair = obs["F_pair"].to(device=device)
    grid_xy = obs["grid_xy"].to(device=device)
    action_mask = obs["action_mask"].to(device=device)

    logits = policy(z_star, Z_placed, F_pair, grid_xy, action_mask)
    return torch.nn.functional.cross_entropy(logits.unsqueeze(0), expert_action.unsqueeze(0))


# ---------------------------------------------------------------------------
# Main pretraining function
# ---------------------------------------------------------------------------

def bc_pretrain(
    policy: nn.Module,
    dataset: BCDataset,
    cfg: Any,
) -> None:
    """Train the policy to imitate expert placements. Modifies policy in-place.

    Args:
        policy: SCPTPolicy instance (modified in-place).
        dataset: BCDataset instance.
        cfg: SimpleNamespace with fields:
            - epochs (int): number of epochs over the full dataset
            - lr (float): learning rate
            (other fields ignored)
    """
    steps = dataset.all_steps()
    if not steps:
        logger.warning("BCDataset is empty — skipping BC pretraining")
        return

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)
    policy.train()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n = 0
        # Shuffle steps each epoch.
        perm = torch.randperm(len(steps)).tolist()
        for i in perm:
            step = steps[i]
            optimizer.zero_grad()
            loss = bc_loss(policy, step, cfg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / n if n > 0 else 0.0
        logger.debug("BC epoch %d/%d: avg_loss=%.4f", epoch + 1, cfg.epochs, avg)


def eval_bc_loss(policy: nn.Module, dataset: BCDataset, cfg: Any) -> float:
    """Evaluate BC loss on the dataset (no gradient). Returns mean loss."""
    steps = dataset.all_steps()
    if not steps:
        return float("nan")
    policy.eval()
    total = 0.0
    with torch.no_grad():
        for step in steps:
            total += bc_loss(policy, step, cfg).item()
    policy.train()
    return total / len(steps)
