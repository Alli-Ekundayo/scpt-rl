#!/usr/bin/env python3
"""Bake a trained SCPT-RL policy's placement decisions into a real KiCad .kicad_pcb file.

Overview
--------
1. Load the source board via pcb_parser (same Rust wheel the env uses) to get
   the canonical JSON IR with bounds, components, placement_order, and per-
   component expert positions.
2. Optionally run a single greedy rollout of the trained policy on the board
   (requires --checkpoint + --config).  If no checkpoint is given the script
   instead uses the *expert* placement baked into the IR (useful for sanity-
   checking the pipeline with a known-good result).
3. For each component, convert the chosen grid cell index back to world-space
   mm coordinates using the *identical* formula pcb_env.py uses:
       x = bounds.x + (cell_x + 0.5) * grid_resolution_mm
       y = bounds.y + (cell_y + 0.5) * grid_resolution_mm
4. Rewrite the footprint ``(at x y [rot])`` s-expression in the source
   .kicad_pcb using a line-by-line state machine that matches only the
   top-level footprint ``at`` token (not the per-pad / per-text ``at`` tokens
   which are in relative coords and must not change).
5. Write the patched board to --out-board.

Usage (expert placement passthrough — no checkpoint needed)::

    python scripts/bake_placement.py \\
        --board dataset/base_raw/Driverino-Shield.kicad_pcb \\
        --out-board /tmp/driverino_placed.kicad_pcb

Usage (run trained policy first)::

    python scripts/bake_placement.py \\
        --board dataset/base_raw/Driverino-Shield.kicad_pcb \\
        --checkpoint runs/scpt-rl/checkpoints/iter_001000.pt \\
        --config configs/default.yaml \\
        --out-board /tmp/driverino_placed.kicad_pcb \\
        --grid-resolution-mm 0.5

The output file is a valid KiCad 8 .kicad_pcb that ``kicad-cli pcb …`` can
consume directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scpt.bake")

# ---------------------------------------------------------------------------
# KiCad s-expression patching
# ---------------------------------------------------------------------------

def _collect_footprint_blocks(lines: list[str]) -> list[dict]:
    """First pass: scan lines and return one descriptor per footprint block.

    Handles both KiCad 8 format (``(footprint …)`` + ``(property "Reference" "…")``)
    and KiCad 5 format (``(module …)`` + ``(fp_text reference REF …)``).

    Each descriptor::

        {
          "ref_des":     str | None,
          "at_line_idx": int | None,   # line index of the placement (at …) token
          "at_indent":   str,
        }
    """
    blocks: list[dict] = []
    depth = 0
    in_fp = False
    fp_open_depth = 0
    cur: dict = {}

    for i, line in enumerate(lines):
        opens       = line.count("(")
        closes      = line.count(")")
        depth_after = depth + opens - closes

        stripped = line.lstrip()

        if not in_fp:
            # KiCad 8: "(footprint …" at depth 1
            # KiCad 5: "(module …"   at depth 1
            if (stripped.startswith("(footprint ") or stripped.startswith("(module ")) and depth == 1:
                in_fp         = True
                fp_open_depth = depth  # == 1
                cur = {"ref_des": None, "at_line_idx": None, "at_indent": ""}
        else:
            # ── Capture placement `(at …)` ─────────────────────────────────
            # Must be the *first* (at …) at depth == fp_open_depth + 1,
            # i.e. a direct child of the footprint/module block.
            if cur["at_line_idx"] is None and stripped.startswith("(at ") and depth == fp_open_depth + 1:
                cur["at_line_idx"] = i
                cur["at_indent"]   = line[: len(line) - len(stripped)]

            # ── Capture Reference designator ───────────────────────────────
            if cur["ref_des"] is None:
                # KiCad 8: (property "Reference" "REF")
                m8 = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', line)
                if m8:
                    cur["ref_des"] = m8.group(1)
                else:
                    # KiCad 5: (fp_text reference REF (at …) …)
                    # The ref_des is the token immediately after "reference".
                    m5 = re.search(r'\(fp_text\s+reference\s+(\S+)', line)
                    if m5:
                        cur["ref_des"] = m5.group(1).strip('"')

            # ── End of footprint/module block ──────────────────────────────
            if depth_after == fp_open_depth:
                in_fp = False
                blocks.append(cur)
                cur = {}

        depth = depth_after

    return blocks


def _patch_footprint_at(
    pcb_text: str,
    ref_to_xy: dict[str, tuple[float, float]],
    ref_to_rot: dict[str, float] | None = None,
) -> tuple[str, int]:
    """Replace footprint (at x y [rot]) tokens in *pcb_text* for each ref in ref_to_xy.

    Uses a two-pass approach so the Reference property (which comes *after*
    the at-token in KiCad 8 files) is always known before patching.

    Returns (patched_text, n_patched).
    """
    if ref_to_rot is None:
        ref_to_rot = {}

    lines = pcb_text.split("\n")

    # Pass 1: collect footprint block metadata.
    blocks = _collect_footprint_blocks(lines)

    # Build a dict: line_idx → (ref_des, x, y, rot)
    patch_map: dict[int, tuple[str, float, float, float | None]] = {}
    for block in blocks:
        ref = block["ref_des"]
        idx = block["at_line_idx"]
        if ref is not None and idx is not None and ref in ref_to_xy:
            x, y = ref_to_xy[ref]
            rot  = ref_to_rot.get(ref, None)
            patch_map[idx] = (ref, x, y, rot)

    # Pass 2: apply patches.
    n_patched = 0
    result: list[str] = []
    for i, line in enumerate(lines):
        if i in patch_map:
            ref, x, y, rot = patch_map[i]
            indent = line[: len(line) - len(line.lstrip())]
            if rot is not None and rot != 0.0:
                line = f"{indent}(at {x:.4f} {y:.4f} {rot:.4f})"
            else:
                line = f"{indent}(at {x:.4f} {y:.4f})"
            n_patched += 1
            log.debug("Patched %-10s → (%.4f, %.4f)", ref, x, y)
        result.append(line)

    return "\n".join(result), n_patched



# ---------------------------------------------------------------------------
# Policy rollout (optional — needs checkpoint)
# ---------------------------------------------------------------------------

def _run_policy_rollout(
    board_path: str,
    ckpt_path: str,
    config_path: str,
    grid_resolution_mm: float,
) -> dict[str, tuple[float, float]]:
    """Run one greedy rollout of the trained policy.

    Returns a dict mapping ref_des → (x_mm, y_mm) for every placed component.
    """
    import torch
    from scpt.utils import load_cfg
    from scpt.env.pcb_env import PcbPlacementEnv, EnvConfig
    from scpt.model.gnn_encoder import HeteroPCBEncoder, encode_design
    from scpt.model.scpt_transformer import SCPTPolicy
    from scpt.training.data import build_pair_features

    cfg       = load_cfg(config_path)
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg  = ckpt.get("cfg", {})
    model_d      = ckpt_cfg.get("model", {}).get("d",        cfg.model.d)
    model_pair   = ckpt_cfg.get("model", {}).get("pair_dim", cfg.model.pair_dim)
    model_heads  = ckpt_cfg.get("model", {}).get("n_heads",  cfg.model.n_heads)
    model_layers = ckpt_cfg.get("model", {}).get("n_layers", cfg.model.n_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HeteroPCBEncoder(
        node_dims={"component": 5, "pad": 4, "net": 6}, hidden=model_d
    ).to(device)
    policy = SCPTPolicy(
        d=model_d, pair_dim=model_pair, n_heads=model_heads, n_layers=model_layers
    ).to(device)
    if "encoder_state" in ckpt and ckpt["encoder_state"] is not None:
        encoder.load_state_dict(ckpt["encoder_state"])
    policy.load_state_dict(ckpt["policy_state"])
    encoder.eval()
    policy.eval()

    env_cfg = EnvConfig(
        grid_resolution_mm=grid_resolution_mm,
        min_spacing_mm=cfg.env.min_spacing_mm,
    )
    env = PcbPlacementEnv(board_path, env_cfg)
    obs, _ = env.reset()
    done    = False
    placements: dict[str, tuple[float, float]] = {}

    with torch.no_grad():
        while not done:
            active_idx  = env.current_active_index()
            if active_idx is None:
                break
            placed_idx  = list(env.current_placed_indices())
            design      = env.current_design()

            _, z_star, Z_placed = encode_design(design, encoder, active_idx, placed_idx)
            pair_features = build_pair_features(design, active_idx, placed_idx)
            if pair_features.shape[-1] != model_pair:
                F = torch.zeros(len(placed_idx), model_pair, device=device)
                if pair_features.numel() > 0:
                    c = min(model_pair, pair_features.shape[-1])
                    F[:, :c] = pair_features[:, :c].to(device)
            else:
                F = pair_features.to(device)

            grid_xy  = torch.as_tensor(obs["grid_xy"],     dtype=torch.float32, device=device)
            act_mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
            logits   = policy(z_star.to(device), Z_placed.to(device), F, grid_xy, act_mask)
            action   = int(logits.argmax().item())

            # ── Grid cell → mm (same formula as pcb_env.py step()) ──────────
            bounds = design["board"]["bounds"]
            cell_x = action % env.W
            cell_y = action // env.W
            x = bounds["x"] + (cell_x + 0.5) * grid_resolution_mm
            y = bounds["y"] + (cell_y + 0.5) * grid_resolution_mm
            ref = design["components"][active_idx]["ref_des"]
            placements[ref] = (x, y)
            log.info("  %-10s → cell (%d,%d) = (%.3f, %.3f) mm", ref, cell_x, cell_y, x, y)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    log.info("Policy rollout complete — placed %d components", len(placements))
    return placements


# ---------------------------------------------------------------------------
# Expert placement passthrough
# ---------------------------------------------------------------------------

def _extract_expert_placement(design: dict) -> dict[str, tuple[float, float]]:
    """Read the original expert positions from the IR JSON for passthrough mode."""
    placements: dict[str, tuple[float, float]] = {}
    for idx, pos in enumerate(design["placement"]["positions"]):
        if pos is None:
            continue
        ref  = design["components"][idx]["ref_des"]
        x, y = pos["position"]
        placements[ref] = (float(x), float(y))
    return placements


def _patch_via_pcbnew(
    board_path: str,
    out_path: str,
    ref_to_xy: dict[str, tuple[float, float]],
    ref_to_rot: dict[str, float] | None = None,
) -> int:
    """Use kicad-python's pcbnew module to load, mutate, and save the board.

    Iterates footprints by reference designator, calls SetPosition() / SetOrientation(),
    and saves the board cleanly. Raises an Exception if pcbnew cannot load the board
    (e.g. KiCad version mismatch), allowing callers to fall back to s-expression patching.
    """
    import sys
    for p in ["/usr/lib/python3/dist-packages", "/usr/lib/kicad/lib/python3/dist-packages"]:
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)
    import pcbnew

    board = pcbnew.LoadBoard(str(board_path))
    n_patched = 0
    ref_to_rot = ref_to_rot or {}

    for fp in board.GetFootprints():
        ref = fp.GetReference()
        if ref in ref_to_xy:
            x, y = ref_to_xy[ref]
            pos = pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
            fp.SetPosition(pos)
            rot = ref_to_rot.get(ref, 0.0)
            if hasattr(pcbnew, "EDA_ANGLE"):
                fp.SetOrientation(pcbnew.EDA_ANGLE(rot, pcbnew.DEGREES_T))
            elif hasattr(fp, "SetOrientationDegrees"):
                fp.SetOrientationDegrees(rot)
            n_patched += 1

    pcbnew.SaveBoard(str(out_path), board)
    return n_patched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bake SCPT-RL policy placement into a .kicad_pcb file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--board",     required=True, help="Source .kicad_pcb file")
    parser.add_argument("--out-board", required=True, help="Output .kicad_pcb file path")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Trained .pt checkpoint (omit for expert-passthrough mode)",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML config path (required when --checkpoint is given)",
    )
    parser.add_argument(
        "--grid-resolution-mm", type=float, default=0.5,
        help="Grid cell size in mm; must match training (default: 0.5)",
    )
    args = parser.parse_args(argv)

    board_path = args.board
    out_path   = Path(args.out_board)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse board with pcb_parser ──────────────────────────────────────
    try:
        import pcb_parser
    except ImportError:
        log.error(
            "pcb_parser Rust wheel not installed.\n"
            "  Build it:  cd rust/pcb_parser && maturin develop --release"
        )
        sys.exit(1)

    design     = json.loads(pcb_parser.load_kicad_pcb(board_path))
    board_name = Path(board_path).stem
    n_comp     = len(design["components"])
    bounds     = design["board"]["bounds"]
    log.info(
        "Loaded %s — %d components | bounds (%.1f, %.1f) %.0f × %.0f mm",
        board_name, n_comp, bounds["x"], bounds["y"], bounds["w"], bounds["h"],
    )

    # ── 2. Obtain placements ─────────────────────────────────────────────────
    if args.checkpoint:
        if not args.config:
            parser.error("--config is required when --checkpoint is given")
        log.info("Running policy greedy rollout from checkpoint: %s", args.checkpoint)
        ref_to_xy = _run_policy_rollout(
            board_path, args.checkpoint, args.config, args.grid_resolution_mm
        )
    else:
        log.info("No checkpoint supplied — using expert (original) placement as passthrough")
        ref_to_xy = _extract_expert_placement(design)

    log.info(
        "Resolved placements for %d / %d components", len(ref_to_xy), n_comp
    )

    # ── 3. Apply placements: pcbnew first, fallback to s-expression patcher ──
    patched_by_pcbnew = False
    try:
        n_patched = _patch_via_pcbnew(board_path, str(out_path), ref_to_xy)
        log.info("Successfully baked placement via pcbnew Python API (%d footprints)", n_patched)
        patched_by_pcbnew = True
    except Exception as exc:
        log.info("pcbnew skipped (%s) — using s-expression patcher", exc)

    if not patched_by_pcbnew:
        with open(board_path, "r", encoding="utf-8") as f:
            pcb_text = f.read()

        patched, n_patched = _patch_footprint_at(pcb_text, ref_to_xy)
        log.info("Patched %d footprint (at …) tokens via s-expression rewriting", n_patched)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(patched)

    log.info("Written → %s", out_path)
    log.info(
        "Suggested next steps:\n"
        "  2D SVG :  kicad-cli pcb export svg --board-only %s -o /tmp/board_2d/\n"
        "  3D+Video: python scripts/render_kicad.py --board %s --out-dir /tmp/render/",
        out_path, out_path,
    )


if __name__ == "__main__":
    main()
