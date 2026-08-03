"""SCPT-RL training entrypoint.

Two-phase training loop:
  Phase 1 — BC warm-start: imitate expert placements from KiCad board files.
  Phase 2 — PPO-EAL fine-tuning: policy-gradient with augmented-Lagrangian
             constraint enforcement (clearance, HPWL, partition-cut budget).

Usage::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --no-wandb

Checkpoints are written to <run_dir>/checkpoints/ every `checkpoint_interval`
outer iterations and at the end of training.

W&B is optional — if ``wandb`` is unavailable or ``--no-wandb`` is passed,
metrics are only printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scpt.train")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg(path: str) -> SimpleNamespace:
    """Load a YAML config file into a SimpleNamespace (dot-access)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert dicts → SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def _ns_to_dict(ns: SimpleNamespace) -> dict:
    """Convert SimpleNamespace back to plain dict for serialisation."""
    out = {}
    for k, v in vars(ns).items():
        out[k] = _ns_to_dict(v) if isinstance(v, SimpleNamespace) else v
    return out


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_models(cfg: SimpleNamespace):
    """Instantiate policy and value heads from config."""
    from scpt.model.gnn_encoder import HeteroPCBEncoder
    from scpt.model.scpt_transformer import SCPTPolicy
    from scpt.model.value_heads import ValueHeads

    encoder = HeteroPCBEncoder(
        node_dims={"component": 5, "pad": 4, "net": 6},
        hidden=cfg.model.d,
    )
    policy = SCPTPolicy(
        d=cfg.model.d,
        pair_dim=cfg.model.pair_dim,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
    )
    value_heads = ValueHeads(
        d=cfg.model.d,
        constraint_names=list(cfg.ppo.constraint_names),
    )
    logger.info(
        "Encoder: %d params | Policy: %d params | ValueHeads: %d params",
        sum(p.numel() for p in encoder.parameters()),
        sum(p.numel() for p in policy.parameters()),
        sum(p.numel() for p in value_heads.parameters()),
    )
    return encoder, policy, value_heads


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_dir: Path,
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    value_heads: torch.nn.Module,
    dual_updater,
    outer_iter: int,
    cfg: SimpleNamespace,
) -> Path:
    """Save a checkpoint. Returns the path written."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"iter_{outer_iter:06d}.pt"
    torch.save(
        {
            "outer_iter": outer_iter,
            "encoder_state": encoder.state_dict(),
            "policy_state": policy.state_dict(),
            "value_heads_state": value_heads.state_dict(),
            "lambdas": dual_updater.lambdas,
            "ema": dual_updater._ema,
            "cfg": _ns_to_dict(cfg),
        },
        path,
    )
    logger.info("Checkpoint → %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    encoder: torch.nn.Module,
    policy: torch.nn.Module,
    value_heads: torch.nn.Module,
    dual_updater,
) -> int:
    """Load weights from a checkpoint. Returns the outer_iter it was saved at."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder_state" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state"])
    policy.load_state_dict(ckpt["policy_state"])
    value_heads.load_state_dict(ckpt["value_heads_state"])
    for k, v in ckpt.get("lambdas", {}).items():
        dual_updater._lambdas[k] = v
    for k, v in ckpt.get("ema", {}).items():
        dual_updater._ema[k] = v
    outer_iter = ckpt.get("outer_iter", 0)
    logger.info("Loaded checkpoint from %s (outer_iter=%d)", path, outer_iter)
    return outer_iter


# ---------------------------------------------------------------------------
# Phase 1: BC warm-start
# ---------------------------------------------------------------------------

def run_bc_phase(policy: torch.nn.Module, cfg: SimpleNamespace, use_wandb: bool, log_path: Path | None = None) -> None:
    """Behavioural-cloning warm-start phase."""
    from scpt.training.bc_pretrain import BCDataset, bc_pretrain, eval_bc_loss

    board_paths = list(cfg.bc.board_paths) if hasattr(cfg.bc, "board_paths") else []
    if not board_paths:
        logger.warning("bc.board_paths is empty — skipping BC phase")
        return

    bc_cfg = SimpleNamespace(
        grid_resolution_mm=cfg.env.grid_resolution_mm,
        min_spacing_mm=cfg.env.min_spacing_mm,
        d=cfg.model.d,
        pair_dim=cfg.model.pair_dim,
        epochs=cfg.bc.epochs,
        lr=cfg.bc.lr,
    )

    logger.info("Building BC dataset from %d board(s)…", len(board_paths))
    dataset = BCDataset(board_paths=board_paths, cfg=bc_cfg)
    logger.info("BC dataset: %d episodes, %d steps", len(dataset), len(dataset.all_steps()))

    bc_pretrain(policy, dataset, bc_cfg)

    val_loss = eval_bc_loss(policy, dataset, bc_cfg)
    logger.info("BC complete. eval_loss=%.4f", val_loss)
    if log_path is not None:
        _append_jsonl(log_path, {"iter": 0, "bc/eval_loss": val_loss, "phase": "bc"})
    if use_wandb:
        try:
            import wandb
            wandb.log({"bc/eval_loss": val_loss, "phase": "bc_done"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 2: PPO-EAL loop
# ---------------------------------------------------------------------------

def run_ppo_phase(
    encoder,
    policy: torch.nn.Module,
    value_heads: torch.nn.Module,
    cfg: SimpleNamespace,
    run_dir: Path,
    use_wandb: bool,
    log_path: Path | None = None,
    resume_from: str | None = None,
) -> None:
    """PPO-EAL fine-tuning phase."""
    from scpt.agent.ppo_eal import PPOEALTrainer

    board_paths = list(cfg.env.board_paths) if hasattr(cfg.env, "board_paths") else []
    if not board_paths:
        logger.warning("env.board_paths is empty — skipping PPO-EAL phase")
        return

    ppo_cfg = SimpleNamespace(
        d=cfg.model.d,
        pair_dim=cfg.model.pair_dim,
        clip_eps=cfg.ppo.clip_eps,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        sigma=cfg.ppo.sigma,
        constraint_names=list(cfg.ppo.constraint_names),
        constraint_budgets=_ns_to_dict(cfg.ppo.constraint_budgets),
        lr=cfg.ppo.lr,
        epochs=cfg.ppo.epochs,
        minibatch_size=cfg.ppo.minibatch_size,
        dual_alpha=cfg.ppo.dual_alpha,
        dual_ema_decay=cfg.ppo.dual_ema_decay,
    )

    trainer = PPOEALTrainer(policy, value_heads, ppo_cfg, encoder=encoder)
    start_iter = 0

    if resume_from:
        start_iter = load_checkpoint(resume_from, encoder, policy, value_heads, trainer.dual_updater)

    # Build envs (one per board for now; multi-env can be added later).
    envs = _build_envs(board_paths, cfg)
    if not envs:
        logger.error("No environments could be constructed — aborting PPO phase")
        return

    n_iters = cfg.ppo.n_outer_iters
    n_steps = cfg.ppo.n_steps_per_iter
    ckpt_interval = getattr(cfg.ppo, "checkpoint_interval", 100)
    log_interval = getattr(cfg.ppo, "log_interval", 10)

    logger.info("PPO-EAL: %d outer iters × %d steps/iter", n_iters, n_steps)

    for outer_iter in range(start_iter, n_iters):
        t0 = time.perf_counter()

        # Round-robin over envs each outer iter.
        env = envs[outer_iter % len(envs)]
        diag = trainer.update(env, n_steps=n_steps)

        elapsed = time.perf_counter() - t0
        reward_mean = diag.get("reward_mean", float("nan"))
        phi_c = diag.get("phi_c", {})
        lambdas = diag.get("lambdas", {})

        if (outer_iter + 1) % log_interval == 0 or outer_iter == 0:
            phi_str = "  ".join(f"{k}={v:.3f}" for k, v in phi_c.items())
            lam_str = "  ".join(f"λ_{k}={v:.4f}" for k, v in lambdas.items())
            logger.info(
                "[%05d/%d] reward=%.4f  %s  %s  (%.2fs)",
                outer_iter + 1, n_iters, reward_mean, phi_str, lam_str, elapsed,
            )

        if log_path is not None:
            record = {
                "iter": outer_iter + 1,
                "ppo/reward_mean": reward_mean,
                "ppo/policy_loss": diag.get("policy_loss", float("nan")),
            }
            for k, v in phi_c.items():
                record[f"ppo/phi_c/{k}"] = v
            for k, v in lambdas.items():
                record[f"ppo/lambda/{k}"] = v
            _append_jsonl(log_path, record)

        if use_wandb:
            _wandb_log_iter(outer_iter, diag)

        if (outer_iter + 1) % ckpt_interval == 0:
            save_checkpoint(run_dir, encoder, policy, value_heads, trainer.dual_updater, outer_iter + 1, cfg)

    # Final checkpoint.
    save_checkpoint(run_dir, encoder, policy, value_heads, trainer.dual_updater, n_iters, cfg)
    logger.info("PPO-EAL complete.")


def _build_envs(board_paths: list[str], cfg: SimpleNamespace):
    """Try to build PcbPlacementEnv instances; fall back gracefully if Rust wheel missing."""
    try:
        from scpt.env.pcb_env import PcbPlacementEnv, EnvConfig
        env_cfg = EnvConfig(
            grid_resolution_mm=cfg.env.grid_resolution_mm,
            min_spacing_mm=cfg.env.min_spacing_mm,
            expert_cut_cost=getattr(cfg.env, "expert_cut_cost", 1.0),
            infeasible_penalty=getattr(cfg.env, "infeasible_penalty", 100.0),
        )
        envs = []
        for p in board_paths:
            try:
                envs.append(PcbPlacementEnv(p, env_cfg))
                logger.info("Env ready: %s", p)
            except Exception as e:
                logger.warning("Could not build env for %s: %s", p, e)
        return envs
    except ImportError as e:
        logger.error("pcb_parser not available: %s — cannot build envs", e)
        return []


def _wandb_log_iter(outer_iter: int, diag: dict) -> None:
    try:
        import wandb
        log = {
            "ppo/reward_mean": diag.get("reward_mean", float("nan")),
            "ppo/policy_loss": diag.get("policy_loss", float("nan")),
            "iter": outer_iter + 1,
        }
        for k, v in diag.get("phi_c", {}).items():
            log[f"ppo/phi_c/{k}"] = v
        for k, v in diag.get("lambdas", {}).items():
            log[f"ppo/lambda/{k}"] = v
        wandb.log(log, step=outer_iter + 1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SCPT-RL training loop")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-dir", default="runs/scpt-rl", help="Directory for logs + checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--bc-only", action="store_true", help="Run BC phase only")
    parser.add_argument("--ppo-only", action="store_true", help="Run PPO phase only (skip BC)")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = _select_device()
    logger.info("Using device: %s", device)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=getattr(cfg, "wandb_project", "scpt-rl"),
                name=getattr(cfg, "wandb_run_name", None),
                config=_ns_to_dict(cfg),
            )
            logger.info("W&B run: %s", wandb.run.url if wandb.run else "(offline)")
        except Exception as e:
            logger.warning("W&B init failed (%s) — proceeding without it", e)
            use_wandb = False

    encoder, policy, value_heads = build_models(cfg)
    encoder.to(device)
    policy.to(device)
    value_heads.to(device)

    if not args.ppo_only:
        logger.info("=== Phase 1: BC warm-start ===")
        run_bc_phase(policy, cfg, use_wandb, log_path=log_path)

    if not args.bc_only:
        logger.info("=== Phase 2: PPO-EAL fine-tuning ===")
        run_ppo_phase(encoder, policy, value_heads, cfg, run_dir, use_wandb, log_path=log_path, resume_from=args.resume)

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
