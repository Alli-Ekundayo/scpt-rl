"""SCPT-RL training visualizations.

Plots produced:
  1. reward_curve     — Episode reward vs outer iteration
  2. lambda_curve     — Lagrangian multipliers (λ) over training
  3. phi_c_curve      — Constraint violations (φ_c) over training
  4. j_c_breakdown    — j_c (raw cost) vs c_adv_mean (critic estimate) per constraint
  5. policy_loss      — Total PPO-EAL loss over training
  6. bc_loss_curve    — BC pretraining loss per epoch (if BC log provided)
  7. placement_heatmap— Heatmap of where the policy places components on a
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


def plot_j_c_breakdown(records: list[dict], out_dir: Path) -> Path | None:
    """Plot j_c (raw per-step cost mean) and c_adv_mean (GAE advantage mean)
    separately per constraint.

    phi_c = j_c + (1/(1-gamma)) * c_adv_mean - budget.
    Plotting these two components side-by-side immediately shows whether
    phi_c is driven by real constraint violations (large j_c) or by a
    noisy / unconverged constraint critic (large c_adv_mean with small j_c).
    """
    _apply_style()
    constraint_keys = sorted({
        k.replace("ppo/j_c/", "")
        for rec in records
        for k in rec
        if k.startswith("ppo/j_c/")
    })
    if not constraint_keys:
        logger.warning("No ppo/j_c/* keys found — skipping j_c breakdown plot"
                       " (requires train.py from commit a1b608f+)")
        return None

    n = len(constraint_keys)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.5 * n), squeeze=False)

    for row, name in enumerate(constraint_keys):
        # Left: j_c
        ax_l = axes[row][0]
        iters_j, vals_j = _extract_series(records, f"ppo/j_c/{name}")
        ax_l.plot(iters_j, vals_j, color=_PALETTE[0], label=f"j_c({name})")
        ax_l.axhline(0.0, color="#f78166", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_l.set_xlabel("Outer Iteration")
        ax_l.set_ylabel("Raw cost mean (per step)")
        ax_l.set_title(f"{name} — j_c (raw constraint cost)")
        ax_l.legend()
        ax_l.grid(True, alpha=0.4)

        # Right: c_adv_mean
        ax_r = axes[row][1]
        iters_a, vals_a = _extract_series(records, f"ppo/c_adv_mean/{name}")
        ax_r.plot(iters_a, vals_a, color=_PALETTE[2], label=f"c_adv_mean({name})")
        ax_r.axhline(0.0, color="#f78166", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_r.set_xlabel("Outer Iteration")
        ax_r.set_ylabel("GAE advantage mean")
        ax_r.set_title(f"{name} — c_adv_mean (critic estimate; should→0 as critic converges)")
        ax_r.legend()
        ax_r.grid(True, alpha=0.4)

    fig.suptitle("SCPT-RL — phi_c Decomposition: j_c vs c_adv_mean", y=1.01)
    fig.tight_layout()
    out = out_dir / "j_c_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
    """Plot BC eval loss over training.

    Two series are distinguished:
    - **BC pretraining** (``phase='bc'``): the single post-warm-start eval,
      always at iter 0.  Appears as a star marker.
    - **PPO-phase drift** (``phase='ppo'``): per-checkpoint BC eval recorded
      by train.py at every ``checkpoint_interval`` during PPO.  Appears as a
      solid line so you can watch the policy drift away from (or back toward)
      the expert over the course of RL fine-tuning.

    If only the pretraining point is present the plot still renders, but a
    warning is emitted so a visually identical-across-runs plot is immediately
    distinguishable from genuinely missing data.
    """
    _apply_style()

    # Split records by phase so we can style them separately.
    bc_iters, bc_losses = [], []
    ppo_iters, ppo_losses = [], []
    for rec in records:
        if "bc/eval_loss" not in rec or "iter" not in rec:
            continue
        try:
            it = int(rec["iter"])
            loss = float(rec["bc/eval_loss"])
        except (ValueError, TypeError):
            continue
        if rec.get("phase") == "ppo":
            ppo_iters.append(it)
            ppo_losses.append(loss)
        else:
            # phase == "bc" or absent (legacy single-point entry)
            bc_iters.append(it)
            bc_losses.append(loss)

    if not bc_iters and not ppo_iters:
        return None

    if ppo_iters:
        logger.info(
            "BC loss curve: %d pretraining point(s) + %d PPO-phase drift point(s)",
            len(bc_iters), len(ppo_iters),
        )
    else:
        logger.warning(
            "bc_loss_curve: only the frozen BC-pretraining point is present in the log "
            "(iter=0, loss=%.4f). PPO-phase drift measurements will appear once the next "
            "checkpoint fires. The plot will look identical to previous runs until then.",
            bc_losses[0] if bc_losses else float("nan"),
        )

    fig, ax = plt.subplots(figsize=(9, 4))

    # PPO-phase drift — primary time-series
    if ppo_iters:
        ax.plot(
            ppo_iters, ppo_losses,
            color=_PALETTE[4], marker="o", markersize=4,
            label="BC eval loss (PPO drift, per checkpoint)",
        )
        _add_ema_line(ax, ppo_iters, ppo_losses, color=_PALETTE[1], label="EMA (α=0.05)")

    # BC pretraining anchor — single star
    if bc_iters:
        ax.scatter(
            bc_iters, bc_losses,
            color=_PALETTE[2], marker="*", s=120, zorder=5,
            label=f"BC pretraining eval (iter 0, loss={bc_losses[0]:.4f})",
        )

    ax.set_xlabel("Outer PPO Iteration")
    ax.set_ylabel("Cross-Entropy Loss (BC)")
    ax.set_title("SCPT-RL — BC Eval Loss: pretraining anchor + PPO drift")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    out = out_dir / "bc_loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def _prepare_viz_obs(env, obs: dict, encoder, d: int, pair_dim: int) -> dict:
    """Build complete policy inputs (z_star, Z_placed, F_pair) for visualization rollouts."""
    import torch
    from scpt.model.gnn_encoder import encode_design
    from scpt.training.data import build_pair_features

    prepared = dict(obs)
    design = getattr(env, "state", None).design if getattr(env, "state", None) is not None else None
    active_idx = env.current_active_index() if hasattr(env, "current_active_index") else None
    placed_indices = env.current_placed_indices() if hasattr(env, "current_placed_indices") else []

    if design is not None and encoder is not None and active_idx is not None:
        with torch.no_grad():
            _, z_star, Z_placed = encode_design(design, encoder, active_idx, placed_indices)
        pair_features = build_pair_features(design, active_idx, placed_indices)
        if pair_features.shape[-1] != pair_dim:
            F_pair = torch.zeros(len(placed_indices), pair_dim)
            if pair_features.numel() > 0:
                cols = min(pair_dim, pair_features.shape[-1])
                F_pair[:, :cols] = pair_features[:, :cols]
        else:
            F_pair = pair_features
    else:
        z_star = torch.zeros(d)
        Z_placed = torch.zeros(len(placed_indices), d)
        F_pair = torch.zeros(len(placed_indices), pair_dim)

    prepared["z_star"] = z_star
    prepared["Z_placed"] = Z_placed
    prepared["F_pair"] = F_pair
    return prepared


def plot_placement_heatmap(
    policy,
    board_path: str,
    cfg: SimpleNamespace,
    out_dir: Path,
    n_episodes: int = 10,
    encoder=None,
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
    # Track per-component placement coordinates and single-episode cell collisions
    comp_positions: dict[str, list[tuple[int, int]]] = {}
    same_episode_collisions = 0

    d = cfg.model.d
    pair_dim = cfg.model.pair_dim

    policy.eval()
    if encoder is not None:
        encoder.eval()

    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset()
            obs = _prepare_viz_obs(env, obs, encoder, d, pair_dim)
            done = False
            ep_cell_comps: dict[tuple[int, int], list[str]] = {}

            while not done:
                active_idx = env.current_active_index() if hasattr(env, "current_active_index") else None
                ref_des = (
                    env.state.design["components"][active_idx]["ref_des"]
                    if (active_idx is not None and getattr(env, "state", None) is not None)
                    else f"C{active_idx}"
                )

                z_star = obs["z_star"]
                Z_placed = obs["Z_placed"]
                F_pair = obs["F_pair"]
                grid_xy = torch.as_tensor(obs["grid_xy"], dtype=torch.float32)
                action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
                logits = policy(z_star, Z_placed, F_pair, grid_xy, action_mask)
                action = int(logits.argmax().item())
                row, col = action // W, action % W
                if 0 <= row < H and 0 <= col < W:
                    counts[row, col] += 1
                    comp_positions.setdefault(ref_des, []).append((col, row))
                    ep_cell_comps.setdefault((row, col), []).append(ref_des)

                obs, _, terminated, truncated, _ = env.step(action)
                obs = _prepare_viz_obs(env, obs, encoder, d, pair_dim)
                done = terminated or truncated

            # Check if multiple components in the SAME episode landed in the exact same cell
            for cell, comps in ep_cell_comps.items():
                if len(comps) > 1:
                    same_episode_collisions += (len(comps) - 1)

    # 2-panel visualization: left = density heatmap, right = per-component location map
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    im = ax1.imshow(
        counts,
        origin="upper",
        cmap="hot",
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax1, label="Placement count")
    ax1.set_title(
        f"Placement Density Heatmap\n({n_episodes} ep · {Path(board_path).name})",
        fontsize=11,
    )
    ax1.set_xlabel("Grid Column (X)")
    ax1.set_ylabel("Grid Row (Y)")

    # Right panel: component-specific placement coordinates
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "X", "P"]
    color_palette = _PALETTE

    comp_names = sorted(comp_positions.keys())
    for idx, ref in enumerate(comp_names):
        coords = comp_positions[ref]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        m = markers[idx % len(markers)]
        c = color_palette[idx % len(color_palette)]
        ax2.scatter(xs, ys, label=ref, color=c, marker=m, s=60, alpha=0.85, edgecolors="none")

    ax2.set_xlim(-1, W)
    ax2.set_ylim(H, -1)  # upper origin to match imshow
    ax2.set_xlabel("Grid Column (X)")
    ax2.set_ylabel("Grid Row (Y)")
    ax2.grid(True, alpha=0.3)

    collision_text = (
        f"Intra-episode cell overlaps: {same_episode_collisions}"
        if same_episode_collisions > 0
        else "No intra-episode cell overlaps (100% distinct cell placement)"
    )
    ax2.set_title(
        f"Per-Component Placement Locations\n({collision_text})",
        fontsize=11,
    )
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)

    fig.tight_layout()
    out = out_dir / "placement_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
        plot_j_c_breakdown(records, out_dir)
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

        from scpt.utils import dict_to_ns as _dict_to_ns, load_cfg as _load_cfg
        from scpt.model.scpt_transformer import SCPTPolicy
        from scpt.model.value_heads import ValueHeads

        cfg = _load_cfg(args.config)

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

        encoder = None
        if "encoder_state" in ckpt and ckpt["encoder_state"] is not None:
            from scpt.model.gnn_encoder import HeteroPCBEncoder
            encoder = HeteroPCBEncoder(
                node_dims={"component": 6, "pad": 4, "net": 6},
                hidden=model_d,
            )
            encoder.load_state_dict(ckpt["encoder_state"])
            encoder.eval()

        plot_placement_heatmap(policy, args.board, cfg, out_dir, n_episodes=args.heatmap_episodes, encoder=encoder)

    logger.info("All plots written to %s/", out_dir)


if __name__ == "__main__":
    main()
