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
import math
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
    max_components: int = 10_000
    # Reward shaping weight for the centroid-spread penalty.
    # c_spread is in [0, 1] (0 = perfectly spread, 1 = point-mass).
    # A weight of 0.1 adds a term of at most 0.1 to the magnitude of the
    # per-step reward (which itself is O(−1) from HPWL / board_diag), so the
    # scale is compatible without re-tuning LR or advantage normalisation.
    # NOTE: c_spread is intentionally NOT a Lagrangian constraint — the AL
    # framework's 1/(1−γ) amplification turns consistently-high per-step costs
    # (c_spread ≈ 0.9 early in training) into phi_c values in the hundreds,
    # blowing up the quadratic AL penalty past what clip_grad_norm can catch.
    spread_shaping_weight: float = 0.1


def _get_component_half_extents(comp: dict, res: float, margin_cells: int) -> tuple[int, int]:
    """Derive half-width and half-height (in grid cells) for a component footprint.

    Tries:
    1. Courtyard polygon AABB.
    2. Pad positions bounding box (with 0.5mm assumed pad diameter).
    3. Conservative fallback of 1 cell + margin.
    """
    courtyard_pts = comp.get("footprint", {}).get("courtyard", {}).get("points", [])
    if courtyard_pts:
        xs = [pt[0] for pt in courtyard_pts]
        ys = [pt[1] for pt in courtyard_pts]
        cyd_w = max(xs) - min(xs)
        cyd_h = max(ys) - min(ys)
        half_w = max(1, int(math.ceil(cyd_w / (2.0 * res)))) + margin_cells
        half_h = max(1, int(math.ceil(cyd_h / (2.0 * res)))) + margin_cells
        return half_w, half_h

    pads = comp.get("footprint", {}).get("pads", [])
    pxs = [p["local_pos"][0] for p in pads if "local_pos" in p]
    pys = [p["local_pos"][1] for p in pads if "local_pos" in p]
    if pxs and pys:
        pad_w = (max(pxs) - min(pxs)) + 0.5
        pad_h = (max(pys) - min(pys)) + 0.5
        half_w = max(1, int(math.ceil(pad_w / (2.0 * res)))) + margin_cells
        half_h = max(1, int(math.ceil(pad_h / (2.0 * res)))) + margin_cells
        return half_w, half_h

    half_w = 1 + margin_cells
    half_h = 1 + margin_cells
    return half_w, half_h


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
        if bounds["w"] <= 0.0 or bounds["h"] <= 0.0:
            outline_pts = initial.get("board", {}).get("outline", {}).get("points", [])
            if len(outline_pts) >= 2:
                xs = [p[0] for p in outline_pts]
                ys = [p[1] for p in outline_pts]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                if max_x > min_x and max_y > min_y:
                    bounds = {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}
                    initial["board"]["bounds"] = bounds
                    self._initial_json = json.dumps(initial)
        self.W = max(1, int(bounds["w"] / self.cfg.grid_resolution_mm))
        self.H = max(1, int(bounds["h"] / self.cfg.grid_resolution_mm))
        self.action_space = gym.spaces.Discrete(self.H * self.W)

        # Observation space: dict of flat arrays sized per grid.
        self.observation_space = gym.spaces.Dict({
            "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.H * self.W,), dtype=np.float32),
            "grid_xy": gym.spaces.Box(-1e6, 1e6, shape=(self.H * self.W, 2), dtype=np.float32),
            "placed_comp_indices": gym.spaces.Box(0, self.cfg.max_components, shape=(self.cfg.max_components,), dtype=np.int64),
        })

        # Pre-compute static grid coordinates (don't change across episodes).
        cs = bounds["x"] + (np.arange(self.W) + 0.5) * self.cfg.grid_resolution_mm
        rs = bounds["y"] + (np.arange(self.H) + 0.5) * self.cfg.grid_resolution_mm
        xx, yy = np.meshgrid(cs, rs)
        self._grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        self.state: _EnvState | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        import copy
        # Reload fresh JSON so each episode starts from the expert placement.
        design_json = self._initial_json
        design = copy.deepcopy(json.loads(self._initial_json))
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
        if self.state is None:
            raise RuntimeError("step() called before reset()")
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
        bounds = st.design["board"]["bounds"]
        costs = self._compute_costs(active_ref_des, bounds)

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

        # Reward signal.
        #
        # Primary: negative HPWL normalised by board diagonal.
        # Raw HPWL is in mm (O(500)–O(2000)); dividing by the board diagonal
        # brings the reward to O(−1), commensurate with normalised advantages.
        #
        # Shaping: subtract a spread penalty (c_spread ∈ [0,1]) scaled by
        # spread_shaping_weight (default 0.1).  This counters the HPWL-collapse
        # failure mode where the policy learns to stack everything at one corner.
        # c_spread is NOT a Lagrangian constraint — see EnvConfig.spread_shaping_weight
        # for the full reasoning.  Constraint costs (clearance, partition) are
        # handled exclusively by the PPO-EAL Lagrangian — not subtracted here.
        board_diag = math.sqrt(bounds["w"] ** 2 + bounds["h"] ** 2)
        reward = (
            -costs.get("c_hpwl", 0.0) / max(board_diag, 1.0)
            - self.cfg.spread_shaping_weight * costs.get("c_spread", 0.0)
        )

        terminated = st.placed_count == len(st.placement_order)
        return self._build_obs(), reward, terminated, False, {"costs": costs}

    def current_design(self) -> dict:
        if self.state is None:
            raise RuntimeError("step() called before reset()")
        return self.state.design

    def current_active_index(self) -> int | None:
        if self.state is None:
            raise RuntimeError("step() called before reset()")
        if self.state.step_idx >= len(self.state.placement_order):
            return None
        return self.state.placement_order[self.state.step_idx]

    def current_placed_indices(self) -> list[int]:
        if self.state is None:
            raise RuntimeError("step() called before reset()")
        return list(self.state.placement_order[: self.state.step_idx])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_costs(self, moved_ref_des: str, bounds: dict | None = None) -> dict[str, float]:
        """Call pcb_parser primitives for the current design state.

        Normalisation
        -------------
        * ``c_clearance``: raw output of ``clearance_cost`` is a sum of
          pairwise bounding-box overlap *areas* in mm², which grows as
          O(P²·component_area) and easily reaches millions when components
          are stacked.  Dividing by board area gives a dimensionless value
          in [0, ∞) that is O(1) for a heavily overlapping layout and 0
          for a clean one — commensurate with the reward signal.
        * ``c_hpwl``: raw HPWL is in mm (O(500)–O(2000)).  Normalisation
          happens in ``step()`` where it enters the reward.
        * ``c_spread``: measures how clustered the placed components are.
          It equals ``max(0, 1 - normalised_spread)``, where
          ``normalised_spread = (std_x + std_y) / board_diagonal``.
          Value 0 = perfectly spread (good); value 1 = all components at
          exactly the same point (worst case).  Only meaningful once ≥ 2
          Value 0 = perfectly spread (good); value 1 = all components at
          exactly the same point (worst case).  Only meaningful once ≥ 2
          components are placed; returns 0.0 on the first step (nothing to
          measure yet).  Used as a **reward shaping term** (not a Lagrangian
          constraint) — see ``EnvConfig.spread_shaping_weight``.
          No Rust call needed — pure Python over the positions dict.
        """
        if bounds is None:
            bounds = self.state.design["board"]["bounds"]
        board_area = bounds["w"] * bounds["h"]
        board_diag = math.sqrt(bounds["w"] ** 2 + bounds["h"] ** 2)
        raw_clearance = pcb_parser.clearance_cost(
            self.state.design_json, self.cfg.min_spacing_mm
        )

        # Centroid spread: std of placed component x- and y-positions,
        # normalised by the board diagonal so it is in [0, 1].
        placed_positions = [
            p["position"]
            for p in self.state.design["placement"]["positions"]
            if p is not None
        ]
        if len(placed_positions) >= 2:
            xs = [p[0] for p in placed_positions]
            ys = [p[1] for p in placed_positions]
            spread = (
                float(np.std(xs)) + float(np.std(ys))
            ) / max(board_diag, 1.0)
            c_spread = max(0.0, 1.0 - spread)
        else:
            # Only one component placed — spread is undefined; no penalty yet.
            c_spread = 0.0

        return {
            # Dimensionless: overlap_area_mm2 / board_area_mm2.
            "c_clearance": raw_clearance / max(board_area, 1.0),
            "c_hpwl": pcb_parser.hpwl_incremental(self.state.design_json, moved_ref_des),
            # v1: partition cut not exposed yet — always 0.0; see docstring above.
            "c_partition": 0.0,
            "b_partition": 1.15 * self.cfg.expert_cut_cost,
            # Centroid clustering penalty: 0 = spread, 1 = all at one point.
            "c_spread": c_spread,
            # v1: Tier 2 sub-scores not exposed yet via PyO3 — use 0.
            "r_tier2": 0.0,
        }

    def _compute_mask(self, active_ref_des: str) -> np.ndarray:
        """Overlap + clearance mask over the placement grid.

        For each already-placed component, masks all grid cells covered by
        its bounding box plus a `min_spacing_mm` clearance margin.  Falls
        back to the clearance margin alone when component bounds are not
        available in the design dict.
        """
        mask = np.ones(self.H * self.W, dtype=np.float32)
        mask_2d = mask.reshape(self.H, self.W)
        res = self.cfg.grid_resolution_mm
        margin_cells = max(1, int(math.ceil(self.cfg.min_spacing_mm / res)))
        board_bounds = self.state.design["board"]["bounds"]
        active_order_idx = (
            self.state.step_idx
            if self.state.step_idx < len(self.state.placement_order)
            else 0
        )
        active_comp_idx = self.state.placement_order[active_order_idx]

        # ------------------------------------------------------------------ #
        # Guard 1: active component's own footprint vs. board boundary.       #
        #                                                                      #
        # Zero out the border band where centering the active component       #
        # would push its extent past a board edge.                            #
        # ------------------------------------------------------------------ #
        active_comp = self.state.design["components"][active_comp_idx]
        a_half_w, a_half_h = _get_component_half_extents(active_comp, res, margin_cells)

        # On diminutive boards / test fixtures, clamp boundary exclusion
        # to ensure the active component is not 100% locked out from the center.
        a_half_h = min(a_half_h, max(0, (self.H - 1) // 2))
        a_half_w = min(a_half_w, max(0, (self.W - 1) // 2))

        # Zero out the border band
        if a_half_h > 0:
            mask_2d[:a_half_h, :] = 0.0
            mask_2d[self.H - a_half_h :, :] = 0.0
        if a_half_w > 0:
            mask_2d[:, :a_half_w] = 0.0
            mask_2d[:, self.W - a_half_w :] = 0.0

        for i, p in enumerate(self.state.design["placement"]["positions"]):
            if p is None or i == active_comp_idx:
                continue
            pos = p["position"]
            cx = int((pos[0] - board_bounds["x"]) / res)
            cy = int((pos[1] - board_bounds["y"]) / res)

            # Determine half-extents in grid cells.
            comp = self.state.design["components"][i]
            half_w, half_h = _get_component_half_extents(comp, res, margin_cells)

            # Mask rectangular region around placed component.
            r_min = max(0, cy - half_h)
            r_max = min(self.H, cy + half_h + 1)
            c_min = max(0, cx - half_w)
            c_max = min(self.W, cx + half_w + 1)
            mask_2d[r_min:r_max, c_min:c_max] = 0.0

        return mask_2d.ravel()

    def _build_obs(self) -> dict[str, np.ndarray]:
        st = self.state
        # Placed component indices.
        placed = []
        for i, p in enumerate(st.design["placement"]["positions"]):
            if p is not None:
                placed.append(i)
        max_components = self.cfg.max_components
        placed_arr = np.zeros(max_components, dtype=np.int64)
        for j, idx in enumerate(placed):
            if j >= max_components:
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
