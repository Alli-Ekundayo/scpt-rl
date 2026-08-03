# SCPT-RL

**Sparse Coupled Placement Transformer with Reinforcement Learning** — a two-phase
policy-gradient system for automated PCB component placement.

[![CI](https://github.com/your-org/scpt-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/scpt-rl/actions/workflows/ci.yml)

---

## Overview

SCPT-RL trains a cross-attention transformer policy to place PCB components
sequentially on a discrete grid.  Training has two phases:

1. **BC warm-start** — behavioural cloning from expert placements embedded in
   existing KiCad `.kicad_pcb` files.
2. **PPO-EAL fine-tuning** — on-policy PPO with an augmented-Lagrangian penalty
   that enforces hard constraints (clearance, HPWL, partition-cut budget) without
   discarding feasible policy updates.

The Rust crate (`pcb_parser`) provides high-speed DRC, HPWL, and net-topology
primitives via a PyO3 wheel.

```
┌─────────────────────────────────────────────────────┐
│  KiCad .kicad_pcb  →  pcb_parser (Rust/PyO3)       │
│         ↓                    ↓                       │
│   BC Dataset          PcbPlacementEnv (Gymnasium)   │
│         ↓                    ↓                       │
│   bc_pretrain()        PPOEALTrainer.update()        │
│         ↓                    ↓                       │
│   SCPTPolicy  ←  GNN encoder  ←  HeteroPCBEncoder   │
│                               ↓                      │
│                        ValueHeads (reward + λ_i)    │
└─────────────────────────────────────────────────────┘
```

---

## Project Layout

```
scpt-rl/
├── src/scpt/
│   ├── model/
│   │   ├── gnn_encoder.py         # HeteroPCBEncoder + build_node_features()
│   │   ├── scpt_transformer.py    # SCPTPolicy (cross-attention)
│   │   └── value_heads.py         # ValueHeads (reward + constraint critics)
│   ├── agent/
│   │   ├── ppo_eal.py             # PPOEALTrainer, loss functions, RolloutBuffer
│   │   └── lagrangian.py          # MomentumDualUpdater
│   ├── env/
│   │   └── pcb_env.py             # PcbPlacementEnv (Gymnasium)
│   ├── training/
│   │   ├── bc_pretrain.py         # BCDataset + bc_pretrain()
│   │   └── data.py                # functional_clusters(), area_descending_cluster_order()
│   └── rust_bridge/               # Thin re-exports of pcb_parser / pcb_router wheels
├── rust/                          # Rust crates (pcb_parser, pcb_router)
├── scripts/
│   ├── train.py                   # End-to-end training entrypoint
│   ├── eval.py                    # Checkpoint evaluation
│   ├── visualize.py               # Training curve + placement heatmap plots
│   └── build_rust.sh              # Build Rust wheel via maturin
├── configs/
│   ├── default.yaml               # Full-scale training config
│   └── smoke.yaml                 # Tiny CI / dev smoke-test config
├── tests/
│   ├── conftest.py                # Shared fixtures (designs, models, FakeEnv)
│   ├── fixtures/
│   │   ├── minimal.kicad_pcb      # Minimal KiCad file (Rust tests)
│   │   ├── two_comp_design.json   # 2-component SCPT IR (Python tests)
│   │   └── five_comp_design.json  # 5-component SCPT IR (clustering tests)
│   └── test_*.py
└── .github/workflows/ci.yml       # GitHub Actions CI
```

---

## Quick Start

### Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python      | 3.10            |
| Rust        | 1.75 (stable)   |
| PyTorch     | 2.0 (CPU for dev) |
| maturin     | 1.4             |

### 1 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv
```

### 2 — Install the Rust wheel

```bash
bash scripts/build_rust.sh
```

This calls `maturin develop` for each crate under `rust/` and installs the
resulting wheels into the active environment.

### 3 — Install Python dependencies

```bash
uv pip install -e .
```

### 4 — Run the test suite

```bash
# Fast Python-only tests (no Rust wheel required):
pytest tests/ \
  --ignore=tests/test_pcb_env.py \
  --ignore=tests/test_rust_bridge.py \
  --ignore=tests/test_router_smoke.py \
  --ignore=tests/test_parser_roundtrip.py \
  -v

# Full suite (requires Rust wheel):
pytest tests/ -v
```

---

## Training

### Edit the config

Open [`configs/default.yaml`](configs/default.yaml) and set:

```yaml
bc:
  board_paths:
    - /path/to/your/board.kicad_pcb

env:
  board_paths:
    - /path/to/your/board.kicad_pcb
```

### Run training

```bash
# Full run (BC + PPO-EAL, with W&B):
python scripts/train.py --config configs/default.yaml --run-dir runs/exp01

# BC only:
python scripts/train.py --config configs/default.yaml --bc-only --no-wandb

# PPO only (skip BC, e.g. resuming from a checkpoint):
python scripts/train.py --config configs/default.yaml \
  --ppo-only --resume runs/exp01/checkpoints/iter_000500.pt
```

Checkpoints are saved to `<run-dir>/checkpoints/iter_NNNNNN.pt` every
`ppo.checkpoint_interval` outer iterations (default: 100).

---

## Evaluation

```bash
python scripts/eval.py \
  --checkpoint runs/exp01/checkpoints/iter_001000.pt \
  --config configs/default.yaml \
  --boards path/to/board.kicad_pcb \
  --n-episodes 5 \
  --output results/eval_1000.json
```

Reports: mean reward, BC eval loss, constraint costs, per-board breakdown, and
the Lagrangian multiplier values frozen in the checkpoint.

---

## Visualizations

```bash
# Training curves from a JSONL log:
python scripts/visualize.py --log-json runs/exp01/log.jsonl --out-dir plots/exp01

# Placement heatmap (needs checkpoint + board):
python scripts/visualize.py \
  --heatmap \
  --checkpoint runs/exp01/checkpoints/iter_001000.pt \
  --config configs/default.yaml \
  --board path/to/board.kicad_pcb \
  --heatmap-episodes 20 \
  --out-dir plots/exp01
```

Plots produced:
- `reward_curve.png` — episode reward + EMA overlay
- `lambda_curve.png` — Lagrangian multipliers per constraint
- `phi_c_curve.png` — constraint violations with feasibility boundary
- `policy_loss.png` — total PPO-EAL loss
- `bc_loss_curve.png` — BC pretraining loss
- `placement_heatmap.png` — where the policy places components on the grid

---

## Architecture

### SCPTPolicy

Cross-attention transformer: grid cells (queries) attend to placed components
(keys/values). The asymmetry matches the problem structure — "where should
component c\* go" is a query-to-context lookup, not a self-attention problem.
Illegal grid cells receive `−∞` logits before the `Categorical` is constructed.

### PPO-EAL

Standard clipped PPO objective augmented with an augmented-Lagrangian penalty
for constraint satisfaction:

```
L = −L_R(π) + Σᵢ (σ/2) · (max{0, λᵢ/σ + φ_cᵢ}² − (λᵢ/σ)²)
```

Dual variables λᵢ are updated with momentum-smoothed ascent after each PPO
epoch. Action masks are stored in the rollout buffer at collection time and
never recomputed during the loss pass (prevents mask-staleness bugs).

### Rust bridge

`pcb_parser` (PyO3) provides:
- `load_kicad_pcb(path)` → JSON string (SCPT IR)
- `clearance_cost(json, min_spacing)` → float
- `hpwl_incremental(json, ref_des)` → float

The bridge is a boundary, not a layer — no logic, no caching.

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `model.d` | 256 | Hidden dimension |
| `model.pair_dim` | 14 | Live-pair feature dimension |
| `model.n_heads` | 8 | Cross-attention heads |
| `model.n_layers` | 4 | Cross-attention layers |
| `ppo.clip_eps` | 0.2 | PPO clip epsilon |
| `ppo.gamma` | 0.99 | Discount factor |
| `ppo.gae_lambda` | 0.95 | GAE λ |
| `ppo.sigma` | 1.0 | AL penalty coefficient |
| `ppo.dual_alpha` | 0.01 | Dual ascent step size |
| `ppo.dual_ema_decay` | 0.9 | EMA smoothing for φ_c |
| `bc.epochs` | 20 | BC pretraining epochs |
| `env.grid_resolution_mm` | 0.5 | Grid cell size (mm) |

See [`configs/default.yaml`](configs/default.yaml) for the complete reference.

---

## License

MIT — see [`LICENSE`](LICENSE).