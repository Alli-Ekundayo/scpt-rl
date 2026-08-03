# SCPT-RL Implementation Plan (port/adapt)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SCPT-RL by porting/adapting three existing codebases into a unified workspace — `kicad_parser` (parser), `pcb-rl-eda/crates/pcb-env` (PyO3 env), `PcbRouter/pcb_router_rs/` (router) — and adding the novel SCPT pieces on top (heterogeneous GNN + cross-attention transformer policy, Union-Find clustering, BC pretraining, PPO-EAL agent written fresh per spec).

**Architecture:** Three existing Rust crates ported into `rust/` as PyO3 wheels, plus a thin Python layer for Gymnasium compat and all the novel ML pieces. `kicad_parser` parses `.kicad_pcb` → `BoardOutput`; an adapter converts `BoardOutput` → SCPT's `PcbDesign` IR; PyO3 surfaces IR + geometry to Python. `pcb-env` provides the hot-path PyO3 env (reset/step/action_mask + Tier 0/1/2 costs); a Python shim wraps it as a Gymnasium env. `PcbRouter` consumes an adapter `PcbDesign → KicadPcbDatabase` and drives iterative rip-up/reroute routing. The PPO-EAL agent is written fresh per the spec, referencing `pcb-rl-eda/python/ppo_eal/` for pattern validation but not copying it.

**Tech Stack:** Rust · PyO3 + maturin · `geo` 0.28 + `geo-types` 0.7 · `kiutils-rs` 0.2 (router) · hand-rolled s-expr parser (parser) · Python 3.11 · PyTorch 2.x · torch-geometric · Gymnasium · numpy · wandb · pytest

## Global Constraints

- Keep the SCPT IR (`PcbDesign`, `BoardGeometry`, `Component`, `Footprint`, `Pad`, `Net`, `PlacementState`, `PartitionSpec`, etc.) as the canonical types in `rust/pcb_parser/src/ir/mod.rs`. The existing Task 1-2 IR stays. Adapters wrap external IRs; they don't replace the canonical one.
- Adapters are pure functions in a dedicated module (`src/ir/adapter.rs` or similar) — they convert between external IRs (`kicad_parser::BoardOutput`, `KicadPcbDatabase`) and SCPT's `PcbDesign`. No logic leaks into the adapters.
- Sign convention: `clipped_ppo_surrogate` returns maximize-convention; `total_loss` applies negation **once**. Unit-test guarded.
- Action mask captured per step in rollout buffer; never recomputed during loss pass.
- BC and RL use the same area-descending, cluster-contiguous placement order.
- All-illegal action mask → required handler: log + penalty + terminate.
- Tier 2 weight discipline: any Tier 2 term whose v1 definition depends only on a DoF the action space doesn't expose gets weight 0 until that DoF ships. `orient_score` escapes via position-sensitive definition.
- Router scope: post-training eval only in v1; `router.enabled` in config.
- Rust bridge (`src/scpt/rust_bridge/`) re-exports only — no logic.
- Ported crates keep their own internal types where needed (e.g. router's `KicadPcbDatabase` needs `kiutils-rs` for round-trip writes); adapters bridge to SCPT IR at the boundary.

## Status of existing tasks

- **Task 1:** complete (commit `1f06e31`). Kept.
- **Task 2:** complete (commit `08e9ccc`). Kept. IR types in `rust/pcb_parser/src/ir/mod.rs` are the canonical types.
- **Task 3:** implementer finished (commit `903d25f`) but was a greenfield nom lexer. Since we're porting `kicad_parser` (which has its own hand-rolled s-expr parser), this commit is orphaned work. **Revert commit `903d25f`** before starting the port, since the port will bring its own s-expr parser via `kicad_parser`.

---

## Phase 1 — Prep: revert orphan lexer, vendor `kicad_parser`

### Task 1R: Revert orphan Task 3 commit

**Files:**
- Modify: git history on `feat/scpt-rl-impl`

- [ ] **Step 1: Identify the commit to revert**

```bash
git log --oneline -5
```

Expected: `903d25f feat(parser): nom-based s-expression lexer` is HEAD.

- [ ] **Step 2: Revert the commit**

```bash
git revert --no-commit HEAD    # revert 903d25f's changes but keep them staged for review
git status
git diff --cached --stat        # verify it's just the lexer files
git commit -m "revert: orphan nom lexer (porting kicad_parser instead)"
```

- [ ] **Step 3: Verify clean state**

```bash
cargo test -p pcb_parser    # IR tests (Task 2) should still pass
git log --oneline -5
```

Expected: HEAD is now the revert commit, Task 2's commit `08e9ccc` is intact, `cargo test` passes for IR tests.

- [ ] **Step 4: Update ledger**

```bash
cat >> .superpowers/sdd/2026-07-29-scpt-rl/progress.md <<'EOF'
Task 3 (original greenfield lexer): reverted (commit 903d25f undone). Replaced by port of kicad_parser.
EOF
```

- [ ] **Step 5: Commit (already done via revert — this is a bookkeeping note)**

### Task 4P: Vendor `kicad_parser` into `rust/pcb_parser`

**Files:**
- Create: `rust/pcb_parser/src/kicad_parser_compat/` (vendored source from `~/Projects/kicad_parser/`)
- Modify: `rust/pcb_parser/Cargo.toml` (add `geo`, `geo-types`, `regex`, `thiserror`, `rmp-serde` deps)
- Modify: `rust/pcb_parser/src/lib.rs` (add `pub mod kicad_parser_compat;`)

**Interfaces:**
- Consumes: `~/Projects/kicad_parser/` source tree
- Produces: `pcb_parser::kicad_parser_compat::{parse_pcb_to_output, BoardOutput, SemanticsExtractor, ...}` accessible from within the crate

**Decision: vendor vs. path dep.** We vendor the source into `rust/pcb_parser/src/kicad_parser_compat/` (a subtree) rather than using a Cargo path dep, because:
- We need to add PyO3 bindings to the same crate eventually
- The `kicad_parser` crate has a binary (`main.rs`) we don't want to ship in the wheel
- Vendoring lets us strip the binary and keep only the library code

Alternative considered: path dep `kicad_parser = { path = "../kicad_parser" }` — rejected because we need to extend it with PyO3 bindings, and extending an external crate via path dep means forking it. Vendoring is simpler.

- [ ] **Step 1: Copy the library source**

```bash
# From repo root
mkdir -p rust/pcb_parser/src/kicad_parser_compat
cp ../kicad_parser/src/lib.rs rust/pcb_parser/src/kicad_parser_compat/
cp ../kicad_parser/src/error.rs rust/pcb_parser/src/kicad_parser_compat/
cp -r ../kicad_parser/src/parser rust/pcb_parser/src/kicad_parser_compat/
cp -r ../kicad_parser/src/models rust/pcb_parser/src/kicad_parser_compat/
cp -r ../kicad_parser/src/geometry rust/pcb_parser/src/kicad_parser_compat/
cp -r ../kicad_parser/src/semantics rust/pcb_parser/src/kicad_parser_compat/
```

Don't copy `main.rs` (binary) — we only want the library.

- [ ] **Step 2: Adjust module structure**

The vendored `lib.rs` currently has `pub mod error; pub mod geometry; ...` etc. In the new location, these become submodules of `kicad_parser_compat`. Either:
- (a) Rename the vendored `lib.rs` to `mod.rs` and use `pub mod kicad_parser_compat;` at the top level, OR
- (b) Keep `lib.rs` named and use `#[path = "kicad_parser_compat/lib.rs"] mod kicad_parser_compat;`

Choose (a): rename `lib.rs` → `mod.rs`, add `pub mod kicad_parser_compat;` to `rust/pcb_parser/src/lib.rs`.

- [ ] **Step 3: Add deps to Cargo.toml**

From the original `kicad_parser/Cargo.toml`, copy the relevant library deps (skip `clap` since we dropped the binary):

```toml
geo-types = "0.7"
geo = "0.28"
regex = "1.10"
thiserror = "1.0"
rmp-serde = "1.3"
```

`serde` and `serde_json` are already there from Task 2.

- [ ] **Step 4: Verify it compiles**

```bash
cargo check -p pcb_parser
```

Fix any path/import issues from the vendoring (e.g. `crate::` paths should still work since module structure is preserved).

- [ ] **Step 5: Run tests**

```bash
cargo test -p pcb_parser
```

Expected: original IR tests from Task 2 still pass. Vendored `kicad_parser`'s tests (if any `#[cfg(test)]` modules came along) also pass.

- [ ] **Step 6: Commit**

```bash
git add rust/pcb_parser/
git commit -m "feat(parser): vendor kicad_parser source for PyO3 integration"
```

### Task 5P: PyO3-bind `kicad_parser` + adapter to SCPT IR

**Files:**
- Create: `rust/pcb_parser/src/ir/adapter.rs`
- Modify: `rust/pcb_parser/Cargo.toml` (add `pyo3` features if not present)
- Modify: `rust/pcb_parser/src/lib.rs` (register new `#[pyclass]` and `#[pyfunction]` entries, including `load_kicad_pcb`)
- Create: `tests/test_parser_roundtrip.py`

**Interfaces:**
- Produces: `load_kicad_pcb(path: &str) -> PyResult<PcbDesign>` — loads a KiCad PCB, runs it through the vendored parser, adapts to SCPT IR, returns Python-visible `PcbDesign`.
- Produces: adapter function `adapter::board_output_to_pcb_design(BoardOutput) -> PcbDesign` (pure Rust, not PyO3-exposed directly — called by `load_kicad_pcb`).
- Consumes: vendored `kicad_parser_compat` (Task 4P), SCPT IR (Task 2).

**Adapter mapping** (the non-obvious parts):

| SCPT field | Source | Notes |
|---|---|---|
| `PcbDesign.board.outline` | `BoardOutput.board_geometry.outline_polygon` | Convert `Vec<Point2D>` → SCPT `Polygon` |
| `PcbDesign.board.keepouts` | `BoardOutput.board_geometry.keepout_zones` | Convert `KeepoutZoneRecord` → SCPT `Keepout` |
| `PcbDesign.components[i].footprint.pads` | `ComponentRecord.pins` | Convert `PinRecord` → SCPT `Pad` (map `pad_shape: String` → `PadShape` enum) |
| `PcbDesign.components[i].footprint.courtyard` | `ComponentRecord.courtyard_polygon` | `Vec<Point2D>` → SCPT `Polygon` |
| `Pad.electrical_proxy` | `PinRecord.electrical_type` | Map `ElectricalPinType` → `PinElectricalProxy`; confidence = 1.0 if type is specific, 0.1 if `Unspecified` |
| `Net.role` | `NetRecord.category` | Map `NetCategory` → `NetRole`; confidence = 1.0 |
| `Net.diff_pair_id` | `NetRecord.differential_pair_partner` | Resolve partner name → index |
| `PcbDesign.placement.positions` | `ComponentRecord.position`/`orientation_deg` | Treat existing positions as the "expert" initial placement; populate `PlacementState` |
| `PcbDesign.placement.placement_order` | Area-descending sort over `ComponentRecord.courtyard_polygon` areas | Compute at adapt time |

- [ ] **Step 1: Write failing Python test**

```python
# tests/test_parser_roundtrip.py
from pathlib import Path
import pcb_parser

def test_load_kicad_pcb_returns_pcb_design():
    # Use a fixture .kicad_pcb (create a minimal one as part of this task)
    fixture = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"
    d = pcb_parser.load_kicad_pcb(str(fixture))
    assert len(d.components) >= 1
    assert len(d.nets) >= 1
    assert d.board.bounds.w > 0

def test_adapter_maps_roles():
    fixture = Path(__file__).parent / "fixtures" / "minimal.kicad_pcb"
    d = pcb_parser.load_kicad_pcb(str(fixture))
    # At least one net should have role != Unclassified
    roles = {n.role for n in d.nets}
    assert len(roles) >= 1
```

- [ ] **Step 2: Create minimal fixture**

Create `tests/fixtures/minimal.kicad_pcb` — a valid KiCad 5+ format with 2-3 components and 2-3 nets. Can be hand-written or extracted from a real small board.

- [ ] **Step 3: Run tests, verify fail**

```bash
cargo test -p pcb_parser && pytest tests/test_parser_roundtrip.py -v
```

Expected: `load_kicad_pcb` not found.

- [ ] **Step 4: Implement the adapter**

In `rust/pcb_parser/src/ir/adapter.rs`:

```rust
use crate::kicad_parser_compat::{BoardOutput, /* etc */};
use crate::ir::*;

pub fn board_output_to_pcb_design(bo: BoardOutput) -> PcbDesign {
    let board = BoardGeometry {
        outline: polygon_from_points(&bo.board_geometry.outline_polygon),
        keepouts: bo.board_geometry.keepout_zones.into_iter().map(|kz| Keepout {
            polygon: polygon_from_points(&kz.polygon),
            // ...
        }).collect(),
        bounds: rect_from_bbox(&bo.board_geometry.bounding_box),
    };
    let components: Vec<Component> = bo.components.into_iter().map(|cr| {
        Component {
            ref_des: cr.ref_des,
            footprint: Footprint {
                pads: cr.pins.into_iter().map(|pr| Pad {
                    net_name: pr.net_name,
                    shape: pad_shape_from_str(&pr.pad_shape),
                    local_pos: Vec2(pr.local_position.x, pr.local_position.y),
                    drill: pr.drill_mm.map(|d| d.x),
                    layers: layer_set_from_str(&/* ... */),
                    electrical_proxy: pin_electrical_proxy_from(cr.electrical_type),
                    electrical_proxy_confidence: confidence_for(cr.electrical_type),
                }).collect(),
                courtyard: polygon_from_points(&cr.courtyard_polygon),
                silkscreen: vec![],  // v1: not extracted
            },
            value: cr.value,
            netclass_hint: None,
        }
    }).collect();
    // ... nets, netclasses, diff_pairs, placement_state
    PcbDesign { board, components, nets, netclasses, diff_pairs, placement }
}
```

- [ ] **Step 5: Wire `load_kicad_pcb` PyO3 function**

```rust
#[pyfunction]
pub fn load_kicad_pcb(path: &str) -> PyResult<ir::PcbDesign> {
    let src = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let bo = kicad_parser_compat::parse_pcb_to_output(&src, path, None, None)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(ir::adapter::board_output_to_pcb_design(bo))
}
```

Add `#[pyclass]` to all SCPT IR types (Task 2's types). This is the PyO3 exposure that was deferred from Task 2.

- [ ] **Step 6: Rebuild wheel, run tests, verify pass**

```bash
./scripts/build_rust.sh
pytest tests/test_parser_roundtrip.py -v
```

- [ ] **Step 7: Commit**

```bash
git add rust/pcb_parser/src/ir/adapter.rs rust/pcb_parser/src/lib.rs \
        rust/pcb_parser/Cargo.toml tests/test_parser_roundtrip.py tests/fixtures/
git commit -m "feat(parser): PyO3 load_kicad_pcb + BoardOutput → PcbDesign adapter"
```

### Task 6P: Expose geometry primitives via PyO3

**Files:**
- Modify: `rust/pcb_parser/src/lib.rs`
- Modify: `rust/pcb_parser/src/ir/mod.rs` (or a new `geometry_ffi.rs`)

**Interfaces:**
- Produces: `hpwl`, `hpwl_incremental`, `clearance_cost`, `partition_cut_cost`, `orient_score`, `decap_proximity_score`, `thermal_score`, `symmetry_score` — all PyO3-exposed, consuming `&PcbDesign`.
- Consumes: SCPT IR; vendored `kicad_parser_compat::geometry` primitives for the underlying polygon math.

- [ ] **Step 1: Write failing Python tests**

```python
# tests/test_geometry.py
import pcb_parser

def test_hpwl_zero_when_unplaced():
    d = pcb_parser.load_kicad_pcb(str(FIXTURE))
    assert pcb_parser.hpwl(d) == 0.0  # no placements → 0

def test_clearance_zero_for_single_component():
    d = pcb_parser.load_kicad_pcb(str(FIXTURE))
    # Place one component
    # (via Python attribute access after PyO3 exposure)
    assert pcb_parser.clearance_cost(d, 0.2) == 0.0
```

- [ ] **Step 2: Implement geometry primitives**

Reuse vendored `kicad_parser_compat::geometry::rotate_point`, `polygon_area`, `point_in_polygon`. Implement:
- `hpwl` — iterate nets, sum bbox width+height of placed pads
- `hpwl_incremental` — filter to nets touching moved component
- `clearance_cost` — AABB broadphase + polygon overlap via `geo::Contains`
- `partition_cut_cost` — iterate nets, check partition side per pad's component
- `orient_score`, `decap_proximity_score`, `thermal_score`, `symmetry_score` — per spec §2.4

All operate on `&PcbDesign` (SCPT IR), using the adapter's placement state.

- [ ] **Step 3: Expose via PyO3 `#[pyfunction]`s**

- [ ] **Step 4: Rebuild wheel, run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(parser): PyO3-exposed geometry primitives (HPWL, clearance, partition, Tier 2 scores)"
```

---

## Phase 2 — Env wheel (port `pcb-env`)

### Task 7P: Vendor `pcb-env` as a second wheel

**Files:**
- Create: `rust/pcb_env/` (new maturin crate)
- Modify: `Cargo.toml` workspace root (add `rust/pcb_env` member)
- Modify: `scripts/build_rust.sh` (build all three wheels)

**Interfaces:**
- Produces: `pcb_env` PyO3 module with `PcbEnv` class: `reset()`, `step(ref_des, x, y, rotation_deg)`, `action_mask(active_ref_des)`.
- Consumes: SCPT IR (`PcbDesign`) — the env takes a JSON-encoded `PcbDesign` in its constructor (matching `pcb-rl-eda`'s API) OR a direct PyO3 object.

**Decision: PyO3-only env or Python-shimmed Gymnasium env?** Per the user's choice, we port the PyO3 env from `pcb-rl-eda/crates/pcb-env/` and wrap it in a thin Python Gymnasium shim later (Task 8P). For now, just get the PyO3 wheel working.

- [ ] **Step 1: Copy `pcb-rl-eda/crates/pcb-env/` source**

```bash
mkdir -p rust/pcb_env/src
cp -r ../pcb-rl-eda/crates/pcb-env/src/* rust/pcb_env/src/
# Write Cargo.toml + pyproject.toml modeled on pcb_parser
```

- [ ] **Step 2: Adapt to consume SCPT IR**

The existing `pcb-env` expects `pcb_format::PcbDesign` (a JSON blob). We need to switch it to consume SCPT's `PcbDesign` from `pcb_parser`. Options:
- (a) Add `pcb_parser` as a Cargo path dep of `pcb_env`
- (b) Keep JSON-based interface; SCPT's `PcbDesign` has `to_json()` from Task 2

Choose (b) for minimal porting friction — pass JSON string to `PcbEnv` constructor, it deserializes into SCPT IR. The hot path is still typed Rust after construction.

Update imports: replace `pcb_format::` with `pcb_parser::ir::`. Update constraint calls to use `pcb_parser`'s geometry primitives (Task 6P).

- [ ] **Step 3: Add to workspace**

```toml
# Root Cargo.toml
[workspace]
members = ["rust/pcb_parser", "rust/pcb_router", "rust/pcb_env"]
```

Update `scripts/build_rust.sh` to build all three.

- [ ] **Step 4: Build + smoke test**

```bash
./scripts/build_rust.sh
python -c "import pcb_env; print(pcb_env.PcbEnv)"
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(env): port pcb-env PyO3 wheel adapted to SCPT IR"
```

### Task 8P: Python Gymnasium shim

**Files:**
- Create: `src/scpt/env/pcb_env_shim.py`
- Create: `src/scpt/env/__init__.py`
- Create: `tests/test_pcb_env_shim.py`

**Interfaces:**
- Produces: `PcbPlacementEnvShim(gymnasium.Env)` — wraps `pcb_env.PcbEnv` with Gymnasium's `reset`/`step` signature and observation/action space declarations.

- [ ] **Step 1: Write failing test**

```python
import gymnasium as gym
from scpt.env.pcb_env_shim import PcbPlacementEnvShim

def test_is_gymnasium_env():
    env = PcbPlacementEnvShim(str(FIXTURE), cfg=_cfg())
    assert isinstance(env, gym.Env)

def test_reset_returns_obs_info():
    env = PcbPlacementEnvShim(str(FIXTURE), cfg=_cfg())
    obs, info = env.reset(seed=42)
    assert "action_mask" in obs

def test_step_returns_obs_reward_term_trunc_info():
    env = PcbPlacementEnvShim(str(FIXTURE), cfg=_cfg())
    obs, _ = env.reset(seed=42)
    # Take any legal action
    legal = (obs["action_mask"] > 0).nonzero(as_tuple=True)[0]
    obs2, r, term, trunc, info = env.step(int(legal[0]))
    assert "costs" in info
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement shim**

```python
import gymnasium as gym
import numpy as np
import pcb_env  # PyO3 module

class PcbPlacementEnvShim(gym.Env):
    def __init__(self, board_path, cfg):
        super().__init__()
        self._inner = pcb_env.PcbEnv(
            design_json=open(board_path).read(),  # adapt as needed
            grid_resolution=cfg.grid_resolution_mm,
            min_spacing=cfg.min_spacing_mm,
            expert_cut_cost=cfg.expert_cut_cost,
        )
        self.action_space = gym.spaces.Discrete(self._inner.num_cells())
        # Observation space declared as Dict of generic boxes

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs_json = self._inner.reset()
        return self._decode_obs(obs_json), {}

    def step(self, action: int):
        # Convert flat action → (ref_des, x, y, rotation)
        # Call self._inner.step(...)
        # Return (obs, reward, terminated, truncated, info)
        ...
```

- [ ] **Step 4: Run, verify pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(env): Gymnasium shim around PyO3 pcb_env"
```

---

## Phase 3 — Router wheel (port `PcbRouter`)

### Task 9P: Port `PcbRouter` as a PyO3 wheel

**Files:**
- Copy source from `../PcbRouter/pcb_router_rs/` to `rust/pcb_router/`
- Modify: `rust/pcb_router/Cargo.toml` (add `pyo3` dep, set `crate-type = ["cdylib", "rlib"]`)
- Create: `rust/pcb_router/pyproject.toml` (maturin)
- Modify: `rust/pcb_router/src/lib.rs` (add `#[pymodule]` wrapper + adapter from SCPT IR → `KicadPcbDatabase`)

**Interfaces:**
- Produces: `pcb_router.route(design_json: str) -> PyResult<String>` (returns JSON routing result)
- Consumes: the existing `GridBasedRouter` implementation. The internal `KicadPcbDatabase` stays as-is (it's needed for kiutils round-trip writes).

- [ ] **Step 1: Copy source**

```bash
# Remove stub from Task 1
rm -rf rust/pcb_router/src/*
cp -r ../PcbRouter/pcb_router_rs/src/* rust/pcb_router/src/
cp ../PcbRouter/pcb_router_rs/Cargo.lock rust/pcb_router/ 2>/dev/null || true
cp ../PcbRouter/pcb_router_rs/tests rust/pcb_router/ -r
```

- [ ] **Step 2: Update Cargo.toml for maturin + PyO3**

Keep `kiutils-rs = "0.2.0"` (needed for round-trip). Add PyO3. Set `crate-type = ["cdylib", "rlib"]`.

- [ ] **Step 3: Add PyO3 wrapper**

In `lib.rs`, keep the existing `pub mod` declarations. Add:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn route(design_json: &str) -> PyResult<String> {
    // 1. Parse design_json → SCPT PcbDesign (via pcb_parser)
    // 2. Adapt PcbDesign → KicadPcbDatabase (new adapter function)
    // 3. Construct GridBasedRouter, run route_all
    // 4. Return routing result as JSON
}

#[pymodule]
fn pcb_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(route, m)?)
}
```

- [ ] **Step 4: Write adapter `PcbDesign → KicadPcbDatabase`**

The router's `KicadPcbDatabase` needs:
- `layers`, `netclasses`, `nets`, `instances`, `pads`

From SCPT `PcbDesign`:
- `instances` from `components` (ref_des, position, rotation, layer)
- `pads` from `components[i].footprint.pads` (with world positions computed)
- `nets` from `nets`
- `netclasses` from `netclasses`
- `layers` from a static list of standard KiCad copper layers

This is a straightforward mapping — same shape, different types.

- [ ] **Step 5: Build + smoke test**

```bash
./scripts/build_rust.sh
python -c "import pcb_router; print(pcb_router)"
```

Also run the ported integration tests:

```bash
cargo test -p pcb_router
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(router): port PcbRouter as PyO3 wheel with SCPT IR adapter"
```

---

## Phase 4 — Python rust bridge

### Task 10P: Rust bridge re-exports

**Files:**
- Create: `src/scpt/__init__.py`
- Create: `src/scpt/rust_bridge/__init__.py`
- Create: `src/scpt/rust_bridge/parser_client.py`
- Create: `src/scpt/rust_bridge/router_client.py`
- Create: `src/scpt/rust_bridge/env_client.py`

- [ ] **Step 1: Write the three re-export modules**

```python
# parser_client.py
from pcb_parser import (
    PcbDesign, Net, Component, Pad, NetRole, PinElectricalProxy,
    PartitionSpec, load_kicad_pcb,
    hpwl, hpwl_incremental, clearance_cost, partition_cut_cost,
    orient_score, decap_proximity_score, thermal_score, symmetry_score,
)

# router_client.py
from pcb_router import route, RoutingResult

# env_client.py
from pcb_env import PcbEnv
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from scpt.rust_bridge import parser_client, router_client, env_client"
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(bridge): Python re-export wrappers for the three Rust wheels"
```

---

## Phase 5 — Model (new work)

### Task 11: Heterogeneous PyG encoder

(unchanged from original plan Task 13)

**Files:**
- Create: `src/scpt/model/__init__.py`
- Create: `src/scpt/model/gnn_encoder.py`
- Create: `tests/test_gnn_encoder.py`

- [ ] ... per original plan Task 13

### Task 12: SCPT transformer policy

(unchanged from original plan Task 14)

### Task 13: Value heads

(unchanged from original plan Task 15)

---

## Phase 6 — Agent (fresh, referencing `pcb-rl-eda/python/ppo_eal/`)

### Task 14: Pure PPO loss functions

(unchanged from original plan Task 16 — including the sign-convention test)

### Task 15: Momentum dual updater

(unchanged from original plan Task 17)

### Task 16: PPOEALTrainer

(unchanged from original plan Task 18)

**Note:** `pcb-rl-eda/python/ppo_eal/losses.py` and `lagrangian.py` can be used as a reference for validating the math, but the implementation is fresh per the spec.

---

## Phase 7 — Training (new work)

### Task 17: Union-Find + BC ordering

(unchanged from original plan Task 19)

### Task 18: BC pretraining

(unchanged from original plan Task 20)

### Task 19: Training loop

(unchanged from original plan Task 21)

---

## Phase 8 — Eval + viz + CI

### Task 20: Eval script

(unchanged from original plan Task 24, uses ported router)

### Task 21: Visualization scripts

(unchanged from original plan Task 25)

### Task 22: CI

(modified: CI now builds three wheels)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: false   # no git submodules; sources are vendored
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - uses: dtolnay/rust-toolchain@stable
      - name: Install Python deps
        run: pip install -e . pytest maturin
      - name: Build Rust wheels
        run: ./scripts/build_rust.sh
      - name: Run tests
        run: pytest -v
```

### Task 23: Configs + fixtures

(unchanged from original plan Task 27)

### Task 24: README + .gitignore

(unchanged from original plan Task 28)

---

## Self-Review Checklist

1. **Spec coverage:** All sections 1–9 of the original spec are covered. Parser comes via `kicad_parser` (port). Geometry primitives come via `kicad_parser_compat::geometry` + new SCPT-specific functions. Router ported from `PcbRouter`. Env ported from `pcb-env`. Agent fresh per spec.
2. **Existing code reuse:** `kicad_parser` (parser + geometry + semantics) → Phase 1. `pcb-env` → Phase 2. `PcbRouter` → Phase 3. `ppo_eal` Python code → reference only (Phase 6).
3. **Placeholder scan:** No TBDs.
4. **Type consistency:** SCPT `PcbDesign` is the canonical IR throughout. Adapters are pure functions in dedicated modules. `KicadPcbDatabase` stays router-internal.

---

## Summary: task count

| Phase | Tasks | New vs. ported |
|---|---|---|
| 1 Prep | 1R (revert), 4P, 5P, 6P | 1 revert + 3 ports |
| 2 Env | 7P, 8P | 1 port + 1 new shim |
| 3 Router | 9P | 1 port |
| 4 Bridge | 10P | 1 new |
| 5 Model | 11-13 | 3 new |
| 6 Agent | 14-16 | 3 new |
| 7 Training | 17-19 | 3 new |
| 8 Polish | 20-24 | 5 new |
| **Total** | **24 tasks** (was 28) | **11 ports/adaptations, 13 new** |
