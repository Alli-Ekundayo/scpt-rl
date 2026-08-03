"""Gymnasium env for SCPT-RL placement.

Wraps the `pcb_parser` Rust wheel (PyO3) to provide:
- `reset()` / `step(action)` / `action_mask` per Gymnasium API
- Tier 0/1/2 cost computation via pcb_parser primitives
- Action space: Discrete(H*W) — flat grid cell index, rotation fixed at 0° for v1

The design state lives in Python as a dict (parsed from the JSON returned by
pcb_parser.load_kicad_pcb). Each step mutates the placement in the dict and
re-serializes to JSON for geometry calls. This JSON hop is a known v1 cost —
a future pass can add a `#[pyclass] PcbDesign` with mutators to avoid it if
profiling says so.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pcb_parser


@dataclass
class EnvConfig:
    """Hyperparameters that don't change per-episode."""
    grid_resolution_mm: float = 0.5
    min_spacing_mm: float = 0.2
    expert_cut_cost: float = 1.0
    infeasible_penalty: float = 100.0
    w_orient: float = 1.0
    w_decap: float = 1.0
    w_therm: float = 1.0
    w_sym: float = 1.0
    decap_radius_mm: float = 2.0


@dataclass
class _EnvState:
    """Mutable per-episode state."""
    design_json: str
    design: dict
    H: int
    W: int
    placement_order: list[int]
    step_idx: int = 0
    placed_count: int = 0


class PcbPlacementEnv(gym.Env):
    """Gymnasium env for SCPT-RL placement.

    Action space: Discrete(H*W) — flat grid cell index.
    Observation: dict with keys matching the spec §4.3 (action_mask, grid_xy,
    placed_comp_indices, etc.). v1 observation is a dict of numpy arrays.
    """
    metadata = {"render_modes": []}

    def __init__(self, board_path: str | Path, cfg: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.board_path = str(board_path)

        # Initial parse.
        self._initial_json = pcb_parser.load_kicad_pcb(self.board_path)
        initial = json.loads(self._initial_json)

        # Compute grid dims from board bounds.
        bounds = initial["board"]["bounds"]
        self.W = max(1, int(bounds["w"] / self.cfg.grid_resolution_mm))
        self.H = max(1, int(bounds["h"] / self.cfg.grid_resolution_mm))
        self.action_space = gym.spaces.Discrete(self.H * self.W)

        # Observation space: dict of flat arrays sized per grid.
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.H * self.W,), dtype=np.float32),
            "grid_xy": gym.spaces.Box(-1e6, 1e6, shape=(self.H * self.W, 2), dtype=np.float32),
            "placed_comp_indices": gym.spaces.Box(0, 10_000, shape=(10_000,), dtype=np.int64),
        })

        # Pre-compute static grid coordinates (don't change across episodes).
        self._grid_xy = np.zeros((self.H * self.W, 2), dtype=np.float32)
        for r in range(self.H):
            for c in range(self.W):
                idx = r * self.W + c
                self._grid_xy[idx, 0] = bounds["x"] + (c + 0.5) * self.cfg.grid_resolution_mm
                self._grid_xy[idx, 1] = bounds["y"] + (r + 0.5) * self.cfg.grid_resolution_mm

        self.state: _EnvState | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Reload fresh JSON so each episode starts from the expert placement.
        design_json = pcb_parser.load_kicad_pcb(self.board_path)
        design = json.loads(design_json)
        placement_order = design["placement"]["placement_order"]
        self.state = _EnvState(
            design_json=design_json,
            design=design,
            H=self.H,
            W=self.W,
            placement_order=placement_order,
            step_idx=0,
            placed_count=0,
        )
        return self._build_obs(), {}

    def step(self, action: int):
        assert self.state is not None, "step() called before reset()"
        st = self.state

        # Action is flat grid index. Convert to world coords.
        cell_y = action // st.W
        cell_x = action % st.W
        bounds = st.design["board"]["bounds"]
        x = bounds["x"] + (cell_x + 0.5) * self.cfg.grid_resolution_mm
        y = bounds["y"] + (cell_y + 0.5) * self.cfg.grid_resolution_mm

        # Active component for this step.
        if st.step_idx >= len(st.placement_order):
            # Already all placed — treat as no-op with zero reward
            return self._build_obs(), 0.0, True, False, {"costs": {}, "done_reason": "already_done"}

        active_idx = st.placement_order[st.step_idx]
        active_ref_des = st.design["components"][active_idx]["ref_des"]

        # Apply placement by mutating the design dict.
        st.design["placement"]["positions"][active_idx] = {
            "component_idx": active_idx,
            "position": [x, y],
            "rotation_deg": 0.0,
            "bottom_layer": False,
        }
        st.placed_count += 1
        st.step_idx += 1

        # Re-serialize for Rust geometry calls.
        st.design_json = json.dumps(st.design)

        # Compute costs.
        costs = self._compute_costs(active_ref_des)

        # Check infeasibility for next step.
        next_active_idx = (
            st.placement_order[st.step_idx] if st.step_idx < len(st.placement_order) else None
        )
        if next_active_idx is not None:
            next_ref_des = st.design["components"][next_active_idx]["ref_des"]
            mask = self._compute_mask(next_ref_des)
            if mask.sum() == 0:
                costs["c_infeasible"] = self.cfg.infeasible_penalty
                return self._build_obs(), -self.cfg.infeasible_penalty, True, False, {
                    "costs": costs, "infeasible": True,
                }

        # Reward: Tier 2 minus constraint costs.
        tier2 = costs.get("r_tier2", 0.0)
        constraint_sum = sum(v for k, v in costs.items() if k.startswith("c_"))
        reward = tier2 - constraint_sum

        terminated = st.placed_count == len(st.placement_order)
        return self._build_obs(), reward, terminated, False, {"costs": costs}

    def current_design(self) -> dict:
        assert self.state is not None, "env not reset"
        return self.state.design

    def current_active_index(self) -> int | None:
        assert self.state is not None, "env not reset"
        if self.state.step_idx >= len(self.state.placement_order):
            return None
        return self.state.placement_order[self.state.step_idx]

    def current_placed_indices(self) -> list[int]:
        assert self.state is not None, "env not reset"
        return list(self.state.placement_order[: self.state.step_idx])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_costs(self, moved_ref_des: str) -> dict[str, float]:
        """Call pcb_parser primitives for the current design state."""
        return {
            "c_clearance": pcb_parser.clearance_cost(self.state.design_json, self.cfg.min_spacing_mm),
            "c_hpwl": pcb_parser.hpwl_incremental(self.state.design_json, moved_ref_des),
            # v1: partition cut not exposed yet — use 0.
            "c_partition": 0.0,
            "b_partition": 1.15 * self.cfg.expert_cut_cost,
            # v1: Tier 2 sub-scores not exposed yet via PyO3 — use 0.
            "r_tier2": 0.0,
        }

    def _compute_mask(self, active_ref_des: str) -> np.ndarray:
        """Coarse overlap/off-board mask.

        v1: marks cells occupied by already-placed components as illegal.
        The clearance / keepout filtering happens via the soft cost instead.
        """
        mask = np.ones(self.H * self.W, dtype=np.float32)
        # Mark cells where another component is placed as illegal.
        for i, p in enumerate(self.state.design["placement"]["positions"]):
            if p is None or i == self.state.placement_order[self.state.step_idx if self.state.step_idx < len(self.state.placement_order) else 0]:
                continue
            pos = p["position"]
            bounds = self.state.design["board"]["bounds"]
            cx = int((pos[0] - bounds["x"]) / self.cfg.grid_resolution_mm)
            cy = int((pos[1] - bounds["y"]) / self.cfg.grid_resolution_mm)
            if 0 <= cx < self.W and 0 <= cy < self.H:
                mask[cy * self.W + cx] = 0.0
        return mask

    def _build_obs(self) -> dict[str, np.ndarray]:
        st = self.state
        # Placed component indices.
        placed = []
        for i, p in enumerate(st.design["placement"]["positions"]):
            if p is not None:
                placed.append(i)
        placed_arr = np.zeros(10_000, dtype=np.int64)
        for j, idx in enumerate(placed):
            if j >= 10_000:
                break
            placed_arr[j] = idx

        # Action mask for the next active component (or all-zero if done).
        if st.step_idx < len(st.placement_order):
            next_idx = st.placement_order[st.step_idx]
            next_ref_des = st.design["components"][next_idx]["ref_des"]
            action_mask = self._compute_mask(next_ref_des)
        else:
            action_mask = np.zeros(self.H * self.W, dtype=np.float32)

        return {
            "action_mask": action_mask,
            "grid_xy": self._grid_xy.copy(),
            "placed_comp_indices": placed_arr,
        }
