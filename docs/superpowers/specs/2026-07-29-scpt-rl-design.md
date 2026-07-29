# SCPT-RL Design Spec

**Sparse Coupled Placement Transformer** — RL-based PCB auto-placement via
PPO-EAL (exact augmented-Lagrangian constrained PPO). Parses KiCad `.kicad_pcb`
boards, trains a heterogeneous-graph encoder + masked-transformer policy to
place components on a grid, subject to hard DRC (action-masked) and soft
Lagrangian constraints (cut-cost, HPWL, clearance), with weighted quality
rewards (orientation, decap proximity, thermal, symmetry).

**Status**: design approved, pre-implementation.

---

## 1. Repository layout & build system

```
scpt-rl/
├── pyproject.toml                    # Python scpt package (src layout, hatchling)
├── Cargo.toml                        # virtual workspace → rust/*
├── README.md
├── configs/
│   ├── ppo_eal.yaml                  # hyperparameters, constraint budgets, BC schedule
│   └── board/
│       └── default.json              # minimal hand-crafted board for smoke tests
├── rust/
│   ├── pcb_parser/                   # maturin PyO3 wheel #1
│   │   ├── Cargo.toml                #   crate-type = ["cdylib"]
│   │   ├── pyproject.toml            #   build-backend = "maturin"
│   │   └── src/
│   │       ├── lib.rs                #     #[pymodule] + top-level bindings
│   │       ├── kicad/                #     KiCad s-expression parser (nom)
│   │       │   ├── mod.rs
│   │       │   ├── lexer.rs
│   │       │   └── types.rs          #     raw KiCad AST
│   │       ├── ir/                   #     canonical PcbDesign IR
│   │       │   └── mod.rs
│   │       └── geometry/             #     HPWL, clearance, partition cut, tier-2 scores
│   │           └── mod.rs
│   └── pcb_router/                   # maturin PyO3 wheel #2 (ported from PcbRouter)
│       ├── Cargo.toml
│       ├── pyproject.toml
│       └── src/
│           ├── lib.rs                #     #[pymodule] + Router class
│           ├── grid/                 #     BoardGrid, GridCell, SearchGrids
│           ├── router/               #     GridBasedRouter (rip-up/reroute)
│           └── types.rs              #     GlobalParam, Location, etc.
├── src/scpt/
│   ├── __init__.py
│   ├── env/
│   │   ├── pcb_env.py                # Gymnasium env (Python, no PyTorch dep)
│   │   ├── action_mask.py            # hard DRC rasterization (numpy)
│   │   └── constraints.py            # tiered Lagrangian cost orchestration
│   ├── model/
│   │   ├── gnn_encoder.py            # heterogeneous PyG encoder
│   │   ├── scpt_transformer.py       # dense masked attention placement policy
│   │   └── value_heads.py            # 1 reward critic + N constraint critics
│   ├── agent/
│   │   ├── ppo_eal.py                # rollout collection + PPO-EAL loss
│   │   └── lagrangian.py             # momentum dual updater
│   ├── rust_bridge/
│   │   ├── __init__.py
│   │   ├── parser_client.py          # re-exports from pcb_parser wheel
│   │   └── router_client.py          # re-exports from pcb_router wheel
│   └── training/
│       ├── train.py                  # main PPO-EAL loop (multi-board cycling)
│       ├── bc_pretrain.py            # behavioral cloning warm-start
│       └── data.py                   # BoardDataset, Union-Find, placement order
├── data/
│   ├── raw_kicad/                    # .kicad_pcb files (gitignored)
│   ├── processed/                    # parser output IR JSON (gitignored)
│   ├── checkpoints/{policies,critics}/
│   └── logs/                         # JSONL training logs
├── visualizations/
│   ├── README.md
│   ├── plot_training_curves.py
│   ├── plot_board_state.py
│   ├── plot_attention_maps.py
│   ├── plot_cluster_layout.py
│   └── plot_ablation.py
├── scripts/
│   ├── build_rust.sh                 # maturin develop both wheels
│   ├── train.py
│   └── evaluate.py
├── tests/                            # see §9
└── .github/workflows/ci.yml
```

### Build flow

**Developer (one-time + per-Rust-change):**
```bash
./scripts/build_rust.sh               # maturin develop both wheels into active venv
pip install -e .                       # installs scpt package (finds wheels already there)
```

**CI:**
```bash
./scripts/build_rust.sh
pytest -v
```

### Build-system choices

- Top-level `Cargo.toml` is a virtual workspace pointing at both rust crates —
  `cargo check` at root checks both, and they share `target/` + `Cargo.lock`.
- Each rust crate has its **own** `pyproject.toml` (maturin is per-wheel). The
  main Python `pyproject.toml` uses hatchling and declares runtime deps on
  `torch`, `torch-geometric`, `gymnasium`, `numpy`, `pyyaml`, `wandb`,
  `matplotlib`. The two Rust wheels are **not** declared there — they're
  installed by `build_rust.sh` as local maturin wheels.
- `pcb_parser` and `pcb_router` are separately versioned wheels — independently
  releasable, matching the "two rust repo" framing.

---

## 2. Rust IR & parser wheel (`pcb_parser`)

Three jobs: parse KiCad s-expressions → produce a canonical IR → expose
geometry primitives. Python-side consumes the IR natively (no JSON round-trip
on the hot path).

### 2.1 Canonical IR

```rust
pub struct PcbDesign {
    pub board: BoardGeometry,
    pub components: Vec<Component>,
    pub nets: Vec<Net>,
    pub netclasses: HashMap<String, Netclass>,
    pub diff_pairs: Vec<DiffPair>,            // (net_idx_a, net_idx_b)
    pub placement: PlacementState,            // mutable during RL rollout
}

pub struct BoardGeometry {
    pub outline: Polygon,                     // board edge (may be non-rectangular)
    pub keepouts: Vec<Keepout>,               // polygons-with-holes (User.1 etc.)
    pub bounds: Rect,                         // cached AABB
}

pub struct Component {
    pub ref_des: String,                      // "C1", "R12", ...
    pub footprint: Footprint,
    pub value: String,
    pub netclass_hint: Option<String>,
}

pub struct Footprint {
    pub pads: Vec<Pad>,                       // footprint-local coords
    pub courtyard: Polygon,                   // component body exclusion zone
    pub silkscreen: Vec<Segment>,             // visualization only
}

pub struct Pad {
    pub net_name: String,
    pub shape: PadShape,
    pub local_pos: Vec2,
    pub drill: Option<f64>,
    pub layers: LayerSet,
    pub electrical_proxy: PinElectricalProxy, // derived, not authoritative
    pub electrical_proxy_confidence: f32,     // 0.1 | 0.5 | 1.0
}

#[derive(Clone, Copy)]
pub enum PinElectricalProxy {
    Passive, PowerIn, PowerOut, Input, Output, Bidirectional, Unknown
}

pub struct Net {
    pub name: String,
    pub pads: Vec<PadRef>,                    // (component_idx, pad_idx)
    pub netclass: String,
    pub role: NetRole,                        // resolved once at parse time
    pub role_confidence: f32,                 // 0.1 | 0.5 | 1.0
    pub diff_pair_id: Option<u32>,            // _P/_N suffix matching
}

#[derive(Clone, Copy)]
pub enum NetRole { Signal, Power, Ground, Clock, Unclassified }

pub struct PlacementState {
    pub positions: Vec<Option<Placement>>,    // None = unplaced
    pub placement_order: Vec<usize>,          // area-descending BC order, precomputed
}

pub struct Placement {
    pub center: Vec2,
    pub rotation_deg: f64,
    pub layer: Layer,
}

pub enum PartitionSpec {
    GeometricSplit { axis: Axis, at: f64 },
    FunctionalClusters(Vec<ClusterId>),       // from Union-Find output
    Explicit(Vec<(usize, PartitionSide)>),
}

impl PcbDesign {
    pub fn to_json(&self) -> Result<String>;  // opt-in escape hatch for debugging
}
```

### 2.2 Design decisions

- **Pin electrical type**: v1 parses `.kicad_pcb` only (no `.kicad_sch`).
  `electrical_proxy` and `role` are derived from `netclass_hint` + net-name
  heuristics. The `_proxy` suffix and the explicit `confidence` field ensure
  no downstream consumer mistakes these for schematic ground truth. A future
  schematic-ingestion pass can raise confidence to 1.0 without breaking the
  GNN's input shape.
- **Confidence values**: `1.0` from netclass match, `0.5` from net-name
  heuristic, `0.1` when defaulting to `Unclassified`/`Unknown`. Concatenated
  as a scalar alongside the one-hot in GNN node features (Section 3) so the
  model can learn to discount uncertain signals.
- **Diff pairs**: `diff_pair_id` on `Net` populated from `_P`/`_N` suffix
  matching at parse time; top-level `diff_pairs` index for fast iteration.
- **`to_json()`**: opt-in debug escape hatch. Native attribute access is the
  hot path. Used for test fixtures, notebook inspection, bug-report snapshots.
- **Thermal / decap proximity**: explicitly **proxies** in v1.
  `thermal_score` uses netclass="Power" pad density as a heat-source proxy;
  `decap_proximity_score` uses component-value heuristics (capacitors near
  ICs). Both are documented as proxies in Rust doc comments and in the Python
  reward module.
- **Copper zones / via keepouts**: explicitly deferred to post-v1.
  `clearance_cost` v1 uses courtyard polygons only.

### 2.3 KiCad parser (`kicad/`)

`nom`-based s-expression lexer feeding a structural parser:

```
tokenize() → Vec<SExpr>  where SExpr = Atom(String) | List(Vec<SExpr>)
parse_design(sexprs) → Result<PcbDesign>
```

Subset parsed for placement: `general`, `layers`, `net`, `footprint`,
`gr_poly`/`gr_rect`/`gr_line` on User layers (keepouts), board outline / edge
cuts. Routing-only data (`segment`, `via`, `zone` copper pours) is ignored —
placement runs first.

### 2.4 Geometry primitives (PyO3-exposed)

```rust
#[pyfunction] fn hpwl(design: &PcbDesign) -> f64;
#[pyfunction] fn hpwl_incremental(design: &PcbDesign, moved_ref: &str) -> f64;
#[pyfunction] fn clearance_cost(design: &PcbDesign, min_spacing: f64) -> f64;
#[pyfunction] fn partition_cut_cost(design: &PcbDesign, partition: &PartitionSpec) -> f64;
#[pyfunction] fn orient_score(design: &PcbDesign) -> f64;
#[pyfunction] fn decap_proximity_score(design: &PcbDesign, radius: f64) -> f64;
#[pyfunction] fn thermal_score(design: &PcbDesign) -> f64;
#[pyfunction] fn symmetry_score(design: &PcbDesign, pairs: &[(String,String)]) -> f64;
```

Plus `#[pyclass]` exposure of the IR types so Python can read
`design.components`, iterate pads, etc. for GNN feature-tensor construction.

**Performance notes:**

- `hpwl_incremental` recomputes only nets touching `moved_ref` — critical
  because HPWL is called every step.
- `clearance_cost` uses AABB broadphase + SAT narrowphase on courtyard
  polygons. O(n²) pairwise is fine for boards up to a few hundred components.
- All geometry functions are `#[pyfunction]` taking `&PcbDesign` via PyO3 — no
  JSON serialization on the hot path. Python holds the `PcbDesign` as a Python
  object wrapping the Rust struct.

---

## 3. Model: GNN encoder, SCPT transformer, live pair features, Union-Find

### 3.1 Heterogeneous PyG encoder (`model/gnn_encoder.py`)

**Node types:**

| Node type | Features |
|---|---|
| `component` | footprint area, courtyard area, pad count, type-one-hot (IC / passive / connector / electromech — from value heuristics), placed flag, current (x, y) if placed, electrical_proxy histogram of its pads |
| `pad` | shape one-hot, layer one-hot, electrical_proxy one-hot ‖ confidence scalar, net_role one-hot ‖ confidence scalar, local offset from component center |
| `net` | role one-hot ‖ confidence scalar, pad count, live HPWL-so-far, diff_pair flag |

**Edge types:**

| Edge type | Meaning | Feature |
|---|---|---|
| `component → pad` | "contains" | pad-local offset |
| `pad → net` | "joins" | none (structural) |
| `component → component` | spatial k-NN on placed components (k=16) | Euclidean distance, same-cluster flag |
| `net → net` | diff-pair partner link | none (structural) |

```python
class HeteroPCBEncoder(nn.Module):
    """3-layer heterogeneous GraphSAGE / HAN-style message passing."""
    def __init__(self, node_dims, edge_dims, hidden=256, n_layers=3): ...
    def forward(self, data: HeteroData) -> dict[str, Tensor]: ...
```

Graph is rebuilt each step from `PcbDesign` — a few hundred nodes, a few
thousand edges. Cheap.

### 3.2 Union-Find functional clustering (`training/data.py`)

```python
def functional_clusters(design: PcbDesign) -> list[int]:
    """
    Two components union if they share a non-power/non-ground net.
    Power/ground excluded because they connect everything and collapse the
    clustering to one blob.
    """
```

Used for:
1. `PartitionSpec.FunctionalClusters` — drives `partition_cut_cost` when that
   variant is selected.
2. **BC ordering refinement** — within area-descending order, components in
   the same cluster are placed contiguously.
3. GNN edge feature (`same_cluster` flag on `component→component` edges).

Computed once per board at episode start. O(nets × avg_pads).

### 3.3 Live pair feature tensors

For each candidate placement of the active component `c*`, pairwise features
against every *placed* component. "Live" because they change as placements
change during the rollout.

| # | Feature | Type |
|---|---|---|
| 1 | `dx, dy, distance` | 3 × float |
| 2 | `shared_net_count` | int |
| 3 | `shared_net_dominant_role` | one-hot (4) — highest-priority shared net's role over {power, clock, signal, ground}; all-zeros vector if no shared nets OR dominant role is unclassified |
| 4 | `min_clearance_violation` | float (1/distance - 1/min_spacing, clipped) |
| 5 | `same_cluster` | bool |
| 6 | `diff_pair_partner` | bool |
| 7 | `placed_component.area` | float |
| 8 | `orientation_compatibility` | float — directional alignment proxy |
| 9 | `confidence_shared_net` | float — confidence of the dominant shared net's role |

`pair_dim = 14` after one-hot expansion. 9 conceptual features kept to avoid
multiplicative K/V projection cost explosion across `n_heads × n_layers`.

### 3.4 SCPT transformer (`model/scpt_transformer.py`)

```python
class SCPTPolicy(nn.Module):
    """
    Sparse Coupled Placement Transformer.
      Query:    each legal grid cell, embedded as f(z_comp[c*], spatial(cell_xy))
      Key/Value: each placed component, embedded as g(z_comp[c], F_pair[c*, c])
      Attention: dense masked cross-attention (mask = action_mask)
      Output:   logits over all grid cells (illegal cells get -inf upstream)

    Cross-attention (not self-attention) because the two sequences have
    different cardinalities and growth behavior:
      - Grid cells (L): large, roughly fixed per step, shrinking only as cells
        get occupied.
      - Placed components (P): start at 0, grow monotonically across the
        episode.
    Self-attention over a merged sequence would need padding/type-embedding
    tricks and compute attention within the grid-cell block that carries no
    signal. Cross-attention matches the problem asymmetry: "where should c*
    go" is a query-to-context lookup.

    "Sparse" refers to the fact that K/V are only over *placed* components,
    not all N. "Coupled" refers to the live-pair features encoding net
    connectivity between c* and each placed component.
    """
    def __init__(self, d=256, pair_dim=14, n_heads=8, n_layers=4): ...
    def forward(self, z_star, Z_placed, F_pair, grid_xy, action_mask): ...
```

Output is over the full grid (H×W), with illegal cells set to `-inf` *before*
the `Categorical` is constructed — matching the existing `PolicyNetwork`
convention.

### 3.5 Value heads (`model/value_heads.py`)

```python
class ValueHeads(nn.Module):
    """
    1 reward critic + N constraint critics, all reading from the same z_comp.
    Separate linear heads (not shared MLP) — constraint critics shouldn't
    fight the reward critic's representation.
    """
    def __init__(self, d, constraint_names):
        self.reward_critic = nn.Linear(d, 1)
        self.constraint_critics = nn.ModuleDict({
            name: nn.Linear(d, 1) for name in constraint_names
        })
    def forward(self, z_comp):
        g = z_comp.mean(dim=0)    # graph-level readout: mean-pool v1
        out = {"reward": self.reward_critic(g).squeeze(-1)}
        for name, head in self.constraint_critics.items():
            out[name] = head(g).squeeze(-1)
        return out
```

**Readout strategy**: mean-pool over `z_comp` as v1. Diagnostic: log critic
loss split by placement-progress quartile (first-25% / middle-50% /
last-25% of components placed). If early-quartile loss is materially worse,
that's the signal to promote `[GRAPH]` virtual node from deferred to in-flight.

### 3.6 Behavioral cloning pretraining

```python
def bc_pretrain(policy, loader, expert_designs, cfg):
    """
    Warm-start SCPT on expert placements.
    Expert = existing placement in .kicad_pcb (human-authored).
    Ordering: area-descending, cluster-contiguous.
    Loss: cross-entropy over the legal-grid categorical, target = expert cell.
    """
```

Area-descending ordering is the *same* ordering used at RL rollout time —
matching them is what makes BC warm-start transfer.

---

## 4. Environment (`env/`)

### 4.1 `env/action_mask.py` — hard DRC masking

```python
def compute_action_mask(
    design: PcbDesign,
    active_comp_idx: int,
    grid_resolution: float,
    min_spacing: float,
    placed_mask: np.ndarray,              # (H, W) bool — cells occupied
) -> np.ndarray:
    """
    Returns (H, W) float32 mask: 1.0 = legal, 0.0 = illegal.

    Illegal iff:
      (a) outside board outline, OR
      (b) inside a keepout polygon, OR
      (c) placing the active component's courtyard at this cell would overlap
          any placed component's courtyard (after min_spacing expansion), OR
      (d) cell is already occupied.

    (a) and (b) precomputed once per board at reset().
    (c) recomputed each step — numpy-vectorized over placed-component AABBs.
    """
```

**Implementation**: persistent `(H, W)` board/keepout mask computed once at
`reset()`. Per-step, rasterize active component's courtyard + min_spacing
expansion at each candidate cell and test overlap against placed components'
AABBs. Vectorized with numpy broadcasting — no Python loop over cells.

**Why Python and not Rust**: mask rasterization is numpy-native work. Per-step
cost is O(P) where P = number of placed components — fine in numpy for
P ≤ a few hundred. Port to Rust only if profiling demands it.

### 4.2 `env/constraints.py` — tiered Lagrangian orchestration

```python
class ConstraintOrchestrator:
    def step_costs(self, active_comp_idx, x, y, rot) -> dict:
        """
        Geometry primitives (HPWL, clearance, partition cut) → Rust.
        Tier 2 reward terms → Python orchestration of Rust scoring functions.
        """
        costs = {
            "c_clearance": pcb_parser.clearance_cost(self.design, min_spacing),
            "c_partition": pcb_parser.partition_cut_cost(self.design, partition_spec),
            "c_hpwl":      pcb_parser.hpwl_incremental(self.design, moved_ref_des),
        }
        for spec in self.specs:
            if spec.budget > 0:
                costs[f"b_{spec.name}"] = spec.budget
        costs["r_tier2"] = (
            cfg.w_orient * pcb_parser.orient_score(self.design)
            + cfg.w_decap  * pcb_parser.decap_proximity_score(self.design, cfg.decap_radius)
            + cfg.w_therm  * pcb_parser.thermal_score(self.design)
            + cfg.w_sym    * pcb_parser.symmetry_score(self.design, sym_pairs)
        )
        return costs
```

**orient_score definition (v1, position-sensitive)**: for each functional
cluster, measures how regularly components are arranged along board axes —
variance of X/Y coordinates within cluster, alignment of cluster's principal
axis to board edges. All three terms depend on *positions*, not rotation, so
`cfg.w_orient` stays > 0 even with rotation fixed at 0°. **General rule**:
any Tier 2 term whose v1 definition depends only on a DoF the action space
doesn't expose gets its weight set to 0 until that DoF ships.

### 4.3 `env/pcb_env.py` — Gymnasium wrapper

```python
class PcbPlacementEnv(gymnasium.Env):
    """
    Action space: Discrete(H*W) — flat grid cell index.
    Rotation: fixed 0° for v1 (extend to Discrete(4) rotation later).

    Observation (dict):
      - "graph":                HeteroData (PyG)
      - "z_comp_star":          Tensor — GNN embedding of active component
      - "placed_comp_indices":  LongTensor
      - "grid_xy":              Tensor (H*W, 2)
      - "action_mask":          Tensor (H*W,)
      - "live_pair_features":   Tensor (P, pair_dim)
    """
    def __init__(self, board_path, cfg): ...
    def reset(self, *, seed=None, options=None): ...
    def step(self, action: int): ...
    def _build_obs(self) -> dict: ...
```

**Design choices:**
- **Action space `Discrete(H*W)`** — flat grid index. Rotation = 0° for v1.
  Extending to include rotation (Discrete(4) for 0/90/180/270) quadruples the
  effective action space — defer until position-only converges.
- **Observation is a dict, not a rasterized image.** GNN consumes live graph;
  transformer consumes placed-component embeddings + live pair features + grid
  coords. No CNN, no rasterization on the hot path. Treated as a possible
  future diagnostic addition if the transformer appears to miss spatial-density
  signal a raster would give for free.
- **GNN lives in the policy, not the env.** Env produces raw features; policy
  owns the encoder. Clean boundary, env testable without PyTorch.
- **Env/policy discipline**: "Does this require model weights? If yes, it
  belongs in the policy." Union-Find, cluster IDs, live pair features, action
  mask → env (no weights). GNN encoding, SCPT attention, value heads → policy.

### 4.4 Rust ↔ Python boundary

| Python call site | Rust wheel | Function |
|---|---|---|
| `env/pcb_env.py` init | `pcb_parser` | `load_kicad_pcb(path) → PcbDesign` |
| `env/action_mask.py` per step | numpy only | (no Rust call) |
| `env/constraints.py` per step | `pcb_parser` | `hpwl_incremental`, `clearance_cost`, `partition_cut_cost`, `orient_score`, `decap_proximity_score`, `thermal_score`, `symmetry_score` |
| `scripts/evaluate.py` post-RL | `pcb_router` | `route(design) → RoutedDesign` |

---

## 5. Agent (`agent/`): PPO-EAL + momentum dual

### 5.1 `agent/lagrangian.py`

```python
@dataclass
class MomentumDualUpdater:
    constraint_names: list[str]
    alpha: float                # lambda_lr
    ema_decay: float            # lambda_ema_decay
    _lambdas: dict
    _ema: dict

    def update(self, phi_c_by_name: dict) -> dict:
        """
        lambda_i^(t+1) = max(0, lambda_i^(t) + alpha * EMA(phi_c_i^pi))
        EMA smoothed to damp oscillation. Stateful across epochs — one
        instance per training run, not per-update.
        """
```

### 5.2 `agent/ppo_eal.py`

```python
class PPOEALTrainer:
    def collect_rollout(self, env, n_steps) -> RolloutBuffer:
        """
        Records (obs, action, log_prob, reward, done, costs_dict,
        stored_action_mask) per step. stored_action_mask is captured at
        collection time — compute_constraint_gae and constraint_surrogate
        take this stored mask as an explicit argument, never recompute.
        Off-policy masking mismatch would silently bias phi_c_i.
        """

    def compute_gae(self, rewards, values, dones, gamma, gae_lambda) -> Tensor: ...
    def compute_constraint_gae(self, costs_i, values_i, dones, gamma, gae_lambda, stored_mask) -> Tensor: ...

    def ppo_eal_loss(self, batch) -> tuple[Tensor, dict]:
        """
        L = -L_R^{pi_k}(pi)
          + sum_i (sigma/2) * (max{0, lambda_i/sigma + phi_c_i}^2 - lambda_i^2/sigma^2)

        where phi_c_i = J_c_i + (1/(1-gamma)) * E[A_c_i over LEGAL actions only] - b_i

        Returns (loss_to_minimize, diagnostics_dict).
        """

    def update(self, env, n_steps) -> dict:
        """
        1. Collect rollout
        2. For epochs_per_update epochs: sample minibatches, compute and apply
           ppo_eal_loss
        3. Compute phi_c_i estimates for dual update
        4. Update lambdas via MomentumDualUpdater
        5. Return diagnostics
        """
```

### 5.3 Loss functions (pure, independently testable)

```python
def clipped_ppo_surrogate(new_log_probs, old_log_probs, advantages, clip_eps): ...
def constraint_surrogate(j_c_pi_k, constraint_advantages, budget, gamma):
    """
    CRITICAL: advantages must be averaged only over LEGAL actions or phi_c
    gets pulled toward zero by the masked-out probability mass.
    """
def augmented_lagrangian_penalty(phi_c, lam, sigma): ...
def total_loss(reward_surrogate, constraint_phis, lambdas, sigma):
    """Returns loss to MINIMIZE. Applies negation to reward_surrogate here —
    the only place. clipped_ppo_surrogate returns the standard
    maximize-convention value."""
```

### 5.4 Diagnostic logging (built-in v1)

Per update:
- Per-constraint: running mean of `c_i`, `b_i`, `phi_c_i`, `lambda_i`, critic
  MSE loss split by placement-progress quartile
- Per-constraint: `phi_c_i` and `c_i` split by classification confidence
  (netclass-sourced vs. heuristic-sourced vs. unclassified) — catches
  systematic classification bias hiding in an otherwise-converging lambda
  trajectory
- Per-episode: total reward, cost per constraint, episode length, policy
  entropy
- Per-outer-update: lambda trajectory, sigma, policy gradient norm,
  infeasibility rate (steps where action mask was all-illegal, distinct from
  normal completion)

All logged to both `wandb` (if configured) and local JSONL at
`data/logs/<run_id>.jsonl`.

### 5.5 Infeasible-mask handler (required, not optional)

Dense boards WILL hit an all-illegal mask during training. Frame as a
required failure-mode handler:

```python
if action_mask.sum() == 0:
    costs["c_infeasible"] = cfg.infeasible_penalty
    terminated = True
    log("infeasible_step", comp_idx=comp_idx, step=step_idx)
```

`c_infeasible` is a new cost key; whether it gets its own constraint critic
or folds into an existing one is a config choice.

---

## 6. Training loop, BC pretraining, data (`training/`)

### 6.1 `training/data.py`

```python
class BoardDataset(Dataset):
    """
    Each item: pre-parsed PcbDesign + precomputed metadata (cluster_ids,
    placement_order, sym_pairs, partition_spec). PcbDesign is immutable
    across epochs; only its PlacementState mutates during rollouts.
    """

def functional_clusters(design) -> list[int]:
    """Union-Find over components. Two components union if they share a
    non-power/non-ground net."""

def area_descending_cluster_order(design, cluster_ids) -> list[int]:
    """
    Cluster-major, area-minor. Same ordering used at BC AND RL time —
    matching them is what makes BC warm-start transfer.

    Tie-breaking (deterministic, reproducible across runs):
      - Clusters with equal total courtyard area: sorted by cluster_id ascending.
      - Components within a cluster with equal courtyard area: sorted by
        ref_des ascending (lexicographic).
    """

def find_symmetry_pairs(design) -> list[tuple[int, int]]:
    """Diff pairs + matched resistor/capacitor pairs in same cluster."""

def resolve_partition_spec(cfg, cluster_ids) -> PartitionSpec: ...
```

### 6.2 `training/bc_pretrain.py`

```python
def bc_pretrain(policy, dataset, cfg) -> None:
    """
    Warm-start SCPT on expert placements (human-authored .kicad_pcb).
    For each board, for each component in placement_order:
      1. Build obs via env._build_obs() with current partial placement
      2. Target = expert's grid cell
      3. Loss = cross-entropy over legal-grid categorical
      4. If expert cell is itself illegal under grid resolution
         (quantization issue), log as "bc_expert_infeasible" with board_path
         + ref_des and skip
      5. Place component at expert location, advance
    """
```

**Infeasibility handling**: skip + log with per-board + per-component
detail. Summary reports overall rate, per-board rate, top-N components.
If rate > 5% → flag grid_resolution mismatch. If clustered on a few boards
→ flag those for exclusion from BC until resolution is adaptive.

### 6.3 `training/train.py`

```python
def train(cfg) -> None:
    """
    Single-board-per-update, cycling sequentially through dataset.
    Each iteration picks dataset[iter_idx % len(dataset)].
    PcbPlacementEnv instantiated fresh per iteration.

    Policy sees multiple boards across training (generalization pressure)
    without VecEnv parallelization complexity.

    MomentumDualUpdater state persists across iterations — lambda trajectory
    is accumulated across the whole run.
    """
```

**Multi-board cycling, not single-board-forever.** Free to support (just
reload `PcbDesign` at reset with a different `board_path`), avoids rework
when single-board-forever fails to generalize.

### 6.4 `scripts/evaluate.py`

```python
def evaluate(policy, board_path, cfg):
    """
    Greedy placement episode, then invoke pcb_router wheel to route.
    Reports: total HPWL, routed wirelength, via count, bend count,
    clearance violations, Tier 2 scores.
    Router is post-training eval only in v1. Config: router.enabled = "eval_only"
    | "off" | (v2) "reward".
    """
```

**Router scope**: post-training eval only in v1. Wiring into the reward
signal is deferred (cost asymmetry, discontinuous reward signal, conflating
two learning signals before the smoother proxies converge). Documented in
README "Future work" as a planned v2 step.

---

## 7. Configs, data layout, visualizations

### 7.1 `configs/ppo_eal.yaml`

```yaml
constraints:
  - name: clearance
    cost_key: c_clearance
    budget: 0.0
    initial_lambda: 0.0
  - name: partition
    cost_key: c_partition
    budget_key: b_partition
    initial_lambda: 0.0
  - name: hpwl
    cost_key: c_hpwl
    budget_key: b_hpwl
    initial_lambda: 0.0

partition: "functional"

w_orient: 1.0
w_decap: 1.0
w_therm: 1.0
w_sym: 1.0
decap_radius_mm: 2.0

clip_eps: 0.2
gamma: 0.99
gae_lambda: 0.95
lr_policy: 3.0e-4
lr_critics: 1.0e-3
epochs_per_update: 10
minibatch_size: 256
steps_per_update: 2048

sigma: 1.0
lambda_lr: 1.0e-2
lambda_ema_decay: 0.9

grid_resolution_mm: 0.5
min_spacing_mm: 0.2
expert_cut_cost: 1.0
infeasible_penalty: 100.0

board_paths:
  - data/raw_kicad/board_01.kicad_pcb
  - data/raw_kicad/board_02.kicad_pcb

bc:
  enabled: true
  epochs: 5
  lr: 1.0e-3
  checkpoint_path: data/checkpoints/policies/bc_pretrained.pt

train:
  total_iterations: 10000
  checkpoint_interval: 500
  log_jsonl_path: data/logs/{run_id}.jsonl
  wandb_project: scpt-rl
  seed: 42

router:
  enabled: eval_only
```

### 7.2 `configs/board/default.json`

Minimal hand-crafted board (3-4 components, 2-3 nets, no keepouts) for
unit tests and Step-6-style synthetic smoke run.

### 7.3 Data layout

```
data/
├── raw_kicad/              # .kicad_pcb files (gitignored)
├── processed/              # parser IR JSON (gitignored)
├── checkpoints/
│   ├── policies/
│   └── critics/
└── logs/                   # JSONL training logs, one per run
```

### 7.4 `visualizations/`

```
visualizations/
├── README.md
├── plot_training_curves.py       # JSONL → reward, costs, lambdas
├── plot_board_state.py           # PcbDesign → SVG/PNG
├── plot_attention_maps.py        # SCPT attention weights per step
├── plot_cluster_layout.py        # board colored by Union-Find cluster
└── plot_ablation.py              # consumes eval/ablation outputs
```

All scripts take JSONL log path or checkpoint + board path as input, write to
`visualizations/output/`. Pure matplotlib + pandas, no hard wandb dependency.

---

## 8. Rust bridge, build, CI, tests

### 8.1 `src/scpt/rust_bridge/parser_client.py`

Thin re-export wrapper:

```python
from pcb_parser import (
    PcbDesign, Net, Component, Pad, NetRole, PinElectricalProxy,
    PartitionSpec, load_kicad_pcb,
    hpwl, hpwl_incremental, clearance_cost, partition_cut_cost,
    orient_score, decap_proximity_score, thermal_score, symmetry_score,
)
# Deliberately no logic beyond re-export. Caching or transformation belongs
# in env/ or training/, not the bridge. The bridge is a boundary, not a layer.
```

### 8.2 `src/scpt/rust_bridge/router_client.py`

Same pattern:

```python
from pcb_router import Router, RoutingResult, route
```

### 8.3 `scripts/build_rust.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$REPO_ROOT/rust/pcb_parser" && maturin develop --release)
(cd "$REPO_ROOT/rust/pcb_router" && maturin develop --release)
```

### 8.4 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "scpt"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torch-geometric>=2.4",
    "gymnasium>=0.29",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "wandb>=0.15",
    "matplotlib>=3.7",
]

[tool.hatch.build.targets.wheel]
packages = ["src/scpt"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 8.5 Root `Cargo.toml`

```toml
[workspace]
resolver = "2"
members = ["rust/pcb_parser", "rust/pcb_router"]

[profile.release]
opt-level = 3
lto = true
```

### 8.6 `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
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

### 8.7 Tests

```
tests/
├── test_parser_roundtrip.py     # load_kicad_pcb → assert counts
├── test_geometry.py             # hpwl, clearance, partition on hand-built PcbDesign
├── test_action_mask.py          # known-illegal cells come back 0.0
├── test_constraints.py          # orchestrator produces expected cost keys
├── test_ppo_eal_losses.py       # pure-function tests + sign-convention test
├── test_lagrangian.py           # monotone lambda under positive phi_c
├── test_union_find.py           # clustering on hand-built netlist
├── test_bc_ordering.py          # area-descending cluster-contiguous invariant
├── test_router_smoke.py         # route() on tiny placed board
└── fixtures/
    └── minimal.kicad_pcb        # committed sample board
```

Sign-convention test: `test_total_loss_decreases_when_reward_surrogate_increases` —
holds constraint terms fixed, increases `clipped_ppo_surrogate`, asserts
`total_loss` strictly decreases. Catches double-negation regression.

### 8.8 `.gitignore` additions

```
rust/*/target/
*.whl
target/wheels/
data/raw_kicad/
data/processed/
data/logs/
visualizations/output/
.venv/
.env
```

---

## 9. Cross-cutting decisions & invariants

1. **Sign convention**: `clipped_ppo_surrogate` returns the standard
   maximize-convention value. `total_loss` applies the negation once. Unit
   test guards against double-negation.
2. **Mask-instance discipline**: action mask captured per step in rollout
   buffer; never recomputed during loss pass.
3. **Env/policy boundary**: "Does this require model weights? If yes, it
   belongs in the policy."
4. **Confidence propagation**: net role / pin electrical proxy confidence
   flows through GNN node features AND through constraint-cost diagnostic
   logging.
5. **BC/RL ordering parity**: area-descending, cluster-contiguous ordering
   is the same for BC and RL.
6. **Infeasibility handling**: required handler (not optional early
   termination) for all-illegal masks in both RL and BC paths, with distinct
   logging.
7. **Tier 2 weight discipline**: any Tier 2 term whose v1 definition depends
   only on a DoF the action space doesn't expose gets weight 0 until that DoF
   ships. `orient_score` escapes this rule via position-sensitive definition.
8. **Router scope**: post-training eval only in v1; `router.enabled` in
   config makes the scope explicit.
9. **Rust bridge as boundary, not layer**: re-export only; no logic.
10. **Native PyO3 access + JSON escape hatch**: hot path uses attribute
    access; `to_json()` is opt-in for debugging/tests/notebooks.
