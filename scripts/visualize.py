"""SCPT-RL training visualizations.

Plots produced:
  1. reward_curve     — Episode reward vs outer iteration
  2. lambda_curve     — Lagrangian multipliers (λ) over training
  3. phi_c_curve      — Constraint violations (φ_c) over training
  4. policy_loss      — Total PPO-EAL loss over training
  5. bc_loss_curve    — BC pretraining loss per epoch (if BC log provided)
  6. placement_heatmap— Heatmap of where the policy places components on a
                         fixed grid (requires running a rollout)

Usage::

    # From a W&B run history (JSON export):
    python scripts/visualize.py --log-json runs/scpt-rl/log.jsonl --out-dir plots/

    # Minimal: pass rewards + lambdas as positional data:
    python scripts/visualize.py --log-json runs/scpt-rl/log.jsonl

    # Placement heatmap (needs checkpoint + board):
    python scripts/visualize.py --heatmap \\
        --checkpoint runs/scpt-rl/checkpoints/iter_001000.pt \\
        --config configs/default.yaml \\
        --board path/to/board.kicad_pcb

The log file should be a JSONL file with one JSON object per iteration,
containing at minimum: ``iter``, ``ppo/reward_mean``.  Constraint keys
like ``ppo/phi_c/c_hpwl`` and ``ppo/lambda/c_hpwl`` are optional.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger("scpt.visualize")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Shared plot style
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "legend.framealpha": 0.15,
    "legend.edgecolor": "#30363d",
    "font.family": "DejaVu Sans",
}

_PALETTE = [
    "#58a6ff",  # blue
    "#3fb950",  # green
    "#f78166",  # red-orange
    "#d2a8ff",  # purple
    "#ffa657",  # amber
    "#79c0ff",  # light blue
]


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_log(path: str) -> list[dict]:
    """Load a JSONL training log. Each line must be a JSON object."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed log line: %s", e)
    return records


def _extract_series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Extract (iters, values) for a given key from the log records."""
    iters, values = [], []
    for rec in records:
        if key in rec and "iter" in rec:
            try:
                iters.append(int(rec["iter"]))
                values.append(float(rec[key]))
            except (ValueError, TypeError):
                pass
    return iters, values


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_reward_curve(records: list[dict], out_dir: Path) -> Path:
    """Plot episode reward mean vs outer iteration."""
    _apply_style()
    iters, rewards = _extract_series(records, "ppo/reward_mean")
    if not iters:
        logger.warning("No ppo/reward_mean data found in log — skipping reward curve")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, rewards, color=_PALETTE[0], label="Mean Episode Reward")
    _add_ema_line(ax, iters, rewards, color=_PALETTE[1], label="EMA (α=0.05)")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Episode Reward")
    ax.set_title("SCPT-RL — Episode Reward over Training")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "reward_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_lambda_curve(records: list[dict], out_dir: Path) -> Path | None:
    """Plot Lagrangian multipliers (λ) per constraint over training."""
    _apply_style()
    # Discover constraint names from keys like "ppo/lambda/<name>".
    constraint_keys = sorted({
        k.replace("ppo/lambda/", "")
        for rec in records
        for k in rec
        if k.startswith("ppo/lambda/")
    })
    if not constraint_keys:
        logger.warning("No ppo/lambda/* keys found — skipping lambda curve")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(constraint_keys):
        iters, vals = _extract_series(records, f"ppo/lambda/{name}")
        ax.plot(iters, vals, color=_PALETTE[i % len(_PALETTE)], label=f"λ_{name}")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("λ (Lagrange Multiplier)")
    ax.set_title("SCPT-RL — Lagrangian Multipliers over Training")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = out_dir / "lambda_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_phi_c_curve(records: list[dict], out_dir: Path) -> Path | None:
    """Plot constraint violations (φ_c) per constraint over training."""
    _apply_style()
    constraint_keys = sorted({
        k.replace("ppo/phi_c/", "")
        for rec in records
        for k in rec
        if k.startswith("ppo/phi_c/")
    })
    if not constraint_keys:
        logger.warning("No ppo/phi_c/* keys found — skipping phi_c curve")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(constraint_keys):
        iters, vals = _extract_series(records, f"ppo/phi_c/{name}")
        ax.plot(iters, vals, color=_PALETTE[i % len(_PALETTE)], label=f"φ_c({name})")
    ax.axhline(0.0, color="#f78166", linestyle="--", linewidth=1.0, alpha=0.7, label="Feasibility boundary")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("φ_c (Constraint Violation)")
    ax.set_title("SCPT-RL — Constraint Violations over Training")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "phi_c_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_policy_loss(records: list[dict], out_dir: Path) -> Path | None:
    """Plot total PPO-EAL loss over training."""
    _apply_style()
    iters, losses = _extract_series(records, "ppo/policy_loss")
    if not iters:
        logger.warning("No ppo/policy_loss data found — skipping policy loss curve")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iters, losses, color=_PALETTE[3], label="Policy Loss (PPO-EAL)")
    ax.set_xlabel("Outer Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("SCPT-RL — Policy (PPO-EAL) Loss over Training")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "policy_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_bc_loss_curve(records: list[dict], out_dir: Path) -> Path | None:
    """Plot BC pretraining loss, if present in the log."""
    _apply_style()
    iters, losses = _extract_series(records, "bc/eval_loss")
    if not iters:
        return None

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(iters, losses, color=_PALETTE[4], marker="o", markersize=5, label="BC Eval Loss")
    ax.set_xlabel("BC Epoch / Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("SCPT-RL — Behavioural Cloning Pretraining Loss")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "bc_loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def plot_placement_heatmap(
    policy,
    board_path: str,
    cfg: SimpleNamespace,
    out_dir: Path,
    n_episodes: int = 10,
) -> Path | None:
    """Run n_episodes rollouts and heatmap where components get placed."""
    import torch

    _apply_style()

    try:
        from scpt.env.pcb_env import PcbPlacementEnv, EnvConfig
        env_cfg = EnvConfig(
            grid_resolution_mm=cfg.env.grid_resolution_mm,
            min_spacing_mm=cfg.env.min_spacing_mm,
        )
        env = PcbPlacementEnv(board_path, env_cfg)
    except Exception as e:
        logger.warning("Cannot build env for heatmap: %s", e)
        return None

    H, W = env.H, env.W
    counts = np.zeros((H, W), dtype=np.float32)
    d = cfg.model.d
    pair_dim = cfg.model.pair_dim

    policy.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                z_star = obs.get("z_star", torch.zeros(d))
                Z_placed = obs.get("Z_placed", torch.zeros(0, d))
                F_pair = obs.get("F_pair", torch.zeros(0, pair_dim))
                grid_xy = torch.as_tensor(obs["grid_xy"], dtype=torch.float32)
                action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
                logits = policy(z_star, Z_placed, F_pair, grid_xy, action_mask)
                action = int(logits.argmax().item())
                row, col = action // W, action % W
                if 0 <= row < H and 0 <= col < W:
                    counts[row, col] += 1
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        counts,
        origin="upper",
        cmap="hot",
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Placement count")
    ax.set_title(
        f"Component Placement Heatmap\n({n_episodes} episodes · {Path(board_path).name})",
        fontsize=11,
    )
    ax.set_xlabel("Grid Column (X)")
    ax.set_ylabel("Grid Row (Y)")
    fig.tight_layout()
    out = out_dir / "placement_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


# ---------------------------------------------------------------------------
# Helper: EMA overlay
# ---------------------------------------------------------------------------

def _add_ema_line(ax, xs: list, ys: list, color: str, alpha_ema: float = 0.05, **kwargs) -> None:
    """Add an exponential moving average series to an axis."""
    if len(ys) < 2:
        return
    ema = []
    v = ys[0]
    for y in ys:
        v = (1 - alpha_ema) * v + alpha_ema * y
        ema.append(v)
    ax.plot(xs, ema, color=color, linestyle="--", linewidth=1.2, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SCPT-RL training visualizations")
    parser.add_argument("--log-json", default=None, metavar="JSONL",
                        help="JSONL training log file (one JSON object per line)")
    parser.add_argument("--out-dir", default="plots", help="Output directory for PNG files")
    parser.add_argument("--heatmap", action="store_true", help="Generate placement heatmap")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for heatmap")
    parser.add_argument("--config", default=None, help="YAML config for heatmap")
    parser.add_argument("--board", default=None, help="Board path for heatmap")
    parser.add_argument("--heatmap-episodes", type=int, default=10,
                        help="Episodes per heatmap rollout (default: 10)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log-based plots.
    if args.log_json:
        records = load_log(args.log_json)
        logger.info("Loaded %d log records from %s", len(records), args.log_json)
        plot_reward_curve(records, out_dir)
        plot_lambda_curve(records, out_dir)
        plot_phi_c_curve(records, out_dir)
        plot_policy_loss(records, out_dir)
        plot_bc_loss_curve(records, out_dir)
    elif not args.heatmap:
        parser.print_help()
        print("\nProvide --log-json or --heatmap (or both).")
        sys.exit(1)

    # Heatmap.
    if args.heatmap:
        if not (args.checkpoint and args.config and args.board):
            print("--heatmap requires --checkpoint, --config, and --board")
            sys.exit(1)

        import yaml
        from scpt.model.scpt_transformer import SCPTPolicy
        from scpt.model.value_heads import ValueHeads

        def _dict_to_ns(d):
            ns = SimpleNamespace()
            for k, v in d.items():
                setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
            return ns

        with open(args.config) as f:
            cfg = _dict_to_ns(yaml.safe_load(f))

        import torch
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        ckpt_cfg_dict = ckpt.get("cfg", {})
        model_d = ckpt_cfg_dict.get("model", {}).get("d", cfg.model.d)
        model_pair_dim = ckpt_cfg_dict.get("model", {}).get("pair_dim", cfg.model.pair_dim)
        model_n_heads = ckpt_cfg_dict.get("model", {}).get("n_heads", cfg.model.n_heads)
        model_n_layers = ckpt_cfg_dict.get("model", {}).get("n_layers", cfg.model.n_layers)
        constraint_names = ckpt_cfg_dict.get("ppo", {}).get("constraint_names", list(cfg.ppo.constraint_names))

        policy = SCPTPolicy(d=model_d, pair_dim=model_pair_dim, n_heads=model_n_heads, n_layers=model_n_layers)
        policy.load_state_dict(ckpt["policy_state"])

        plot_placement_heatmap(policy, args.board, cfg, out_dir, n_episodes=args.heatmap_episodes)

    logger.info("All plots written to %s/", out_dir)


if __name__ == "__main__":
    main()
