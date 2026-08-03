"""SCPT-RL evaluation script.

Evaluates a trained checkpoint against a set of KiCad board files and reports:
  - Mean episode reward
  - Constraint violations (phi_c per constraint)
  - Lambda (dual variable) values at checkpoint time
  - BC eval loss (how far the policy drifted from the expert)
  - Per-board statistics (components placed, reward, costs)

Usage::

    python scripts/eval.py --checkpoint runs/scpt-rl/checkpoints/iter_001000.pt \\
                           --config configs/default.yaml \\
                           --boards path/to/board.kicad_pcb [...]

    # Output as JSON:
    python scripts/eval.py --checkpoint ... --output results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scpt.eval")


# ---------------------------------------------------------------------------
# Config helpers (mirrors train.py)
# ---------------------------------------------------------------------------

def _load_cfg(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_ns(raw)


def _dict_to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model construction + checkpoint loading
# ---------------------------------------------------------------------------

def _build_and_load(ckpt_path: str, cfg: SimpleNamespace):
    """Construct models and load weights from checkpoint.

    Returns (encoder, policy, value_heads, lambdas, constraint_names, model_d, model_pair_dim).
    """
    from scpt.model.gnn_encoder import HeteroPCBEncoder
    from scpt.model.scpt_transformer import SCPTPolicy
    from scpt.model.value_heads import ValueHeads

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Prefer config baked into the checkpoint over the CLI config.
    if "cfg" in ckpt and ckpt["cfg"]:
        ckpt_cfg_dict = ckpt["cfg"]
        model_d = ckpt_cfg_dict.get("model", {}).get("d", cfg.model.d)
        model_pair_dim = ckpt_cfg_dict.get("model", {}).get("pair_dim", cfg.model.pair_dim)
        model_n_heads = ckpt_cfg_dict.get("model", {}).get("n_heads", cfg.model.n_heads)
        model_n_layers = ckpt_cfg_dict.get("model", {}).get("n_layers", cfg.model.n_layers)
        constraint_names = ckpt_cfg_dict.get("ppo", {}).get("constraint_names", list(cfg.ppo.constraint_names))
    else:
        model_d = cfg.model.d
        model_pair_dim = cfg.model.pair_dim
        model_n_heads = cfg.model.n_heads
        model_n_layers = cfg.model.n_layers
        constraint_names = list(cfg.ppo.constraint_names)

    device = _select_device()
    encoder = HeteroPCBEncoder(
        node_dims={"component": 5, "pad": 4, "net": 6},
        hidden=model_d,
    ).to(device)
    policy = SCPTPolicy(d=model_d, pair_dim=model_pair_dim, n_heads=model_n_heads, n_layers=model_n_layers).to(device)
    value_heads = ValueHeads(d=model_d, constraint_names=constraint_names).to(device)

    if "encoder_state" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state"])
    policy.load_state_dict(ckpt["policy_state"])
    value_heads.load_state_dict(ckpt["value_heads_state"])

    lambdas = ckpt.get("lambdas", {})
    outer_iter = ckpt.get("outer_iter", "?")
    logger.info("Loaded checkpoint (outer_iter=%s)", outer_iter)
    return encoder, policy, value_heads, lambdas, constraint_names, model_d, model_pair_dim


# ---------------------------------------------------------------------------
# Rollout evaluation
# ---------------------------------------------------------------------------

def _prepare_obs(env, obs: dict, encoder, d: int, pair_dim: int) -> dict:
    from scpt.model.gnn_encoder import encode_design
    from scpt.training.data import build_pair_features

    device = next(encoder.parameters()).device
    prepared = {
        "grid_xy": torch.as_tensor(obs["grid_xy"], dtype=torch.float32, device=device),
        "action_mask": torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device),
    }
    design = env.current_design() if hasattr(env, "current_design") else None
    if design is None:
        prepared["z_star"] = torch.zeros(d, device=device)
        prepared["Z_placed"] = torch.zeros(0, d, device=device)
        prepared["F_pair"] = torch.zeros(0, pair_dim, device=device)
        return prepared

    active_idx = env.current_active_index() if hasattr(env, "current_active_index") else None
    placed_indices = list(env.current_placed_indices()) if hasattr(env, "current_placed_indices") else []
    if active_idx is None:
        prepared["z_star"] = torch.zeros(d, device=device)
        prepared["Z_placed"] = torch.zeros(0, d, device=device)
        prepared["F_pair"] = torch.zeros(0, pair_dim, device=device)
        return prepared

    _, z_star, Z_placed = encode_design(design, encoder, active_idx, placed_indices)
    pair_features = build_pair_features(design, active_idx, placed_indices)
    if pair_features.shape[-1] != pair_dim:
        F_pair = torch.zeros(len(placed_indices), pair_dim, device=device)
        if pair_features.numel() > 0:
            F_pair[:, : min(pair_dim, pair_features.shape[-1])] = pair_features[:, : min(pair_dim, pair_features.shape[-1])]
    else:
        F_pair = pair_features

    prepared["z_star"] = z_star
    prepared["Z_placed"] = Z_placed
    prepared["F_pair"] = F_pair
    return prepared


@torch.no_grad()
def _sample_action(policy, obs: dict) -> int:
    """Greedy-sample action from the policy (eval mode)."""
    logits = policy(obs["z_star"], obs["Z_placed"], obs["F_pair"], obs["grid_xy"], obs["action_mask"])
    # Greedy (argmax) for evaluation — no sampling noise.
    action = int(logits.argmax().item())
    return action


def eval_episode(policy, encoder, env, d: int, pair_dim: int) -> dict[str, Any]:
    """Run one full episode. Returns per-episode stats."""
    obs, _ = env.reset()
    obs = _prepare_obs(env, obs, encoder, d, pair_dim)
    done = False
    total_reward = 0.0
    total_costs: dict[str, float] = {}
    steps = 0

    while not done:
        action = _sample_action(policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = _prepare_obs(env, obs, encoder, d, pair_dim)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1
        for k, v in info.get("costs", {}).items():
            total_costs[k] = total_costs.get(k, 0.0) + float(v)

    return {
        "reward": total_reward,
        "steps": steps,
        "costs": total_costs,
        "terminated_normally": not info.get("infeasible", False),
    }


def eval_bc(policy, board_paths: list[str], cfg: SimpleNamespace, model_d: int, model_pair_dim: int) -> float | None:
    """Compute BC eval loss. Returns None if not possible (e.g., no boards)."""
    try:
        from scpt.training.bc_pretrain import BCDataset, eval_bc_loss

        bc_cfg = SimpleNamespace(
            grid_resolution_mm=cfg.env.grid_resolution_mm,
            min_spacing_mm=cfg.env.min_spacing_mm,
            d=model_d,
            pair_dim=model_pair_dim,
            epochs=1,
            lr=1e-3,
        )
        dataset = BCDataset(board_paths=board_paths, cfg=bc_cfg)
        if len(dataset) == 0:
            return None
        return eval_bc_loss(policy, dataset, bc_cfg)
    except Exception as e:
        logger.warning("BC eval failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="SCPT-RL evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--boards", nargs="+", default=[], metavar="PATH",
        help="KiCad .kicad_pcb board paths to evaluate on",
    )
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    parser.add_argument("--n-episodes", type=int, default=1, help="Episodes per board")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    encoder, policy, value_heads, ckpt_lambdas, constraint_names, model_d, model_pair_dim = _build_and_load(args.checkpoint, cfg)
    policy.eval()
    value_heads.eval()
    encoder.eval()

    d = model_d
    pair_dim = model_pair_dim

    # BC eval loss.
    bc_loss = None
    if args.boards:
        bc_loss = eval_bc(policy, args.boards, cfg, model_d, model_pair_dim)
        if bc_loss is not None:
            logger.info("BC eval loss: %.4f", bc_loss)

    # RL episode rollouts.
    board_results = []
    all_rewards: list[float] = []
    all_costs: dict[str, list[float]] = {}

    if args.boards:
        try:
            from scpt.env.pcb_env import PcbPlacementEnv, EnvConfig
            env_cfg = EnvConfig(
                grid_resolution_mm=cfg.env.grid_resolution_mm,
                min_spacing_mm=cfg.env.min_spacing_mm,
            )
            for board_path in args.boards:
                board_ep_rewards = []
                board_ep_costs: dict[str, list[float]] = {}
                try:
                    env = PcbPlacementEnv(board_path, env_cfg)
                    for ep_idx in range(args.n_episodes):
                        ep = eval_episode(policy, encoder, env, d, pair_dim)
                        board_ep_rewards.append(ep["reward"])
                        for k, v in ep["costs"].items():
                            board_ep_costs.setdefault(k, []).append(v)
                            all_costs.setdefault(k, []).append(v)
                        all_rewards.append(ep["reward"])
                        logger.info(
                            "  %s ep%d: reward=%.3f steps=%d%s",
                            Path(board_path).name, ep_idx,
                            ep["reward"], ep["steps"],
                            "" if ep["terminated_normally"] else " [INFEASIBLE]",
                        )
                    board_results.append({
                        "board": board_path,
                        "mean_reward": sum(board_ep_rewards) / len(board_ep_rewards),
                        "n_episodes": len(board_ep_rewards),
                        "mean_costs": {k: sum(v) / len(v) for k, v in board_ep_costs.items()},
                    })
                except Exception as e:
                    logger.warning("Failed to evaluate %s: %s", board_path, e)
                    board_results.append({"board": board_path, "error": str(e)})
        except ImportError as e:
            logger.warning("pcb_parser not available (%s) — skipping RL eval", e)

    # Aggregate.
    results = {
        "checkpoint": args.checkpoint,
        "n_boards": len(args.boards),
        "n_episodes_total": len(all_rewards),
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else None,
        "bc_eval_loss": bc_loss,
        "checkpoint_lambdas": ckpt_lambdas,
        "constraint_names": constraint_names,
        "mean_costs": {k: sum(v) / len(v) for k, v in all_costs.items()},
        "per_board": board_results,
    }

    _print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results written to %s", out_path)

    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("SCPT-RL Evaluation Summary")
    print("=" * 60)
    print(f"  Checkpoint   : {results['checkpoint']}")
    if results["mean_reward"] is not None:
        print(f"  Mean reward  : {results['mean_reward']:.4f}  (n={results['n_episodes_total']})")
    if results["bc_eval_loss"] is not None:
        print(f"  BC eval loss : {results['bc_eval_loss']:.4f}")
    if results["mean_costs"]:
        print("  Mean costs   :")
        for k, v in results["mean_costs"].items():
            print(f"    {k}: {v:.4f}")
    if results["checkpoint_lambdas"]:
        print("  Lambdas (at checkpoint):")
        for k, v in results["checkpoint_lambdas"].items():
            print(f"    λ_{k}: {v:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
