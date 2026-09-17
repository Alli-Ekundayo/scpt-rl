#!/usr/bin/env python3
"""Render a KiCad .kicad_pcb board to 2D SVG and a 3D spinning orbit video.

Step 2 + 3 of the SCPT-RL placement visualisation pipeline.

Produces
--------
  <out-dir>/board_2d.svg          — 2D copper+silk SVG (via kicad-cli)
  <out-dir>/frames/frame_NNN.png  — one 3D raytraced frame per azimuth step
  <out-dir>/orbit.mp4             — smooth 3D spin (all azimuths stitched)
  <out-dir>/splitscreen.mp4       — 2D left | 3D-spin right side-by-side

Requirements
------------
  - kicad-cli  (KiCad ≥8; on Ubuntu: apt install kicad)
  - ffmpeg

Usage::

    python scripts/render_kicad.py \\
        --board /tmp/driverino_placed.kicad_pcb \\
        --out-dir /tmp/render/ \\
        --width 1280 --height 1280 \\
        --azimuth-start 0 --azimuth-end 360 --azimuth-step 5 \\
        --framerate 30

    # Quick test (just 2D, no 3D):
    python scripts/render_kicad.py --board ... --out-dir ... --no-3d

    # Just compose split-screen from existing frames:
    python scripts/render_kicad.py --board ... --out-dir ... --no-2d --no-3d --splitscreen-only
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scpt.render")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(tool: str) -> str:
    """Return full path of *tool* or abort with a helpful message."""
    path = shutil.which(tool)
    if path is None:
        log.error(
            "'%s' not found on PATH.\n"
            "  kicad-cli: sudo apt install kicad   (or brew install kicad on macOS)\n"
            "  ffmpeg:    sudo apt install ffmpeg",
            tool,
        )
        sys.exit(1)
    return path


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log.debug("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        log.error("Command failed (exit %d):\n  %s\nstdout:\n%s\nstderr:\n%s",
                  result.returncode, " ".join(str(c) for c in cmd),
                  result.stdout, result.stderr)
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# 2-D export (SVG)
# ---------------------------------------------------------------------------

def render_2d(
    board: Path,
    out_dir: Path,
    kicad_cli: str,
    width: int,
) -> Path:
    """Export 2D board-only SVG using kicad-cli.

    Handles both KiCad 8 (``--board-only``) and KiCad 7
    (``--exclude-drawing-sheet --page-size-mode 2``) flag differences.

    Returns path to the produced SVG file.
    """
    svg_path = out_dir / "board_2d.svg"

    # Probe which flags this version supports.
    probe = subprocess.run(
        [kicad_cli, "pcb", "export", "svg", "--help"],
        capture_output=True, text=True,
    )
    help_text = probe.stdout + probe.stderr
    has_board_only = "--board-only" in help_text

    cmd = [kicad_cli, "pcb", "export", "svg"]
    if has_board_only:
        cmd.append("--board-only")
    else:
        # KiCad 7 equivalent.
        cmd += ["--exclude-drawing-sheet", "--page-size-mode", "2"]
    cmd += [
        "--layers", "F.Cu,B.Cu,F.SilkS,B.SilkS,Edge.Cuts",
        "-o", str(svg_path),
        str(board),
    ]
    log.info("Exporting 2D SVG …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(
            "kicad-cli 2D export failed (%s) — falling back to direct 2D ratsnest renderer",
            result.stderr.strip().splitlines()[-1] if result.stderr else "error",
        )
        return render_2d_ratsnest(board, out_dir, width)

    log.info("  → %s", svg_path)
    return svg_path


def render_2d_ratsnest(board: Path, out_dir: Path, width: int) -> Path:
    """Generate 2D board visualization with copper pads and airwires / ratsnest."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import pcb_parser

    png_path = out_dir / "board_2d.png"
    design = json.loads(pcb_parser.load_kicad_pcb(str(board)))
    bounds = design["board"]["bounds"]
    bw, bh = max(bounds["w"], 20.0), max(bounds["h"], 20.0)

    dpi = 120
    fig_w = width / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * (bh / bw)), dpi=dpi, facecolor="#0d1117")
    ax.set_facecolor("#112211")

    # Board substrate
    board_rect = Rectangle(
        (bounds["x"], bounds["y"]), bw, bh,
        linewidth=1.5, edgecolor="#2d6a4f", facecolor="#163820", zorder=1,
    )
    ax.add_patch(board_rect)

    # Precompute pad world positions
    pad_positions = {}
    for c_idx, pos in enumerate(design["placement"]["positions"]):
        if pos is None:
            continue
        cx, cy = pos["position"]
        comp = design["components"][c_idx]
        fp = comp.get("footprint", {})
        pads = fp.get("pads", [])
        for p_idx, pad in enumerate(pads):
            ppos = pad.get("pos", [0, 0])
            pad_positions[(c_idx, p_idx)] = (cx + ppos[0], cy + ppos[1])
        ax.text(cx, cy, comp["ref_des"], color="#e0e0e0", fontsize=5, ha="center", va="center", zorder=4)

    # Ratsnest airwires
    for net in design.get("nets", []):
        net_pads = [pad_positions[tuple(p)] for p in net.get("pads", []) if tuple(p) in pad_positions]
        if len(net_pads) > 1:
            for i in range(len(net_pads) - 1):
                p1, p2 = net_pads[i], net_pads[i + 1]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ffd166", alpha=0.45, linewidth=0.7, zorder=2)

    # Pads
    if pad_positions:
        pad_xs = [pos[0] for pos in pad_positions.values()]
        pad_ys = [pos[1] for pos in pad_positions.values()]
        ax.scatter(pad_xs, pad_ys, s=5, color="#e76f51", zorder=3, edgecolors="none")

    margin = max(bw, bh) * 0.05
    ax.set_xlim(bounds["x"] - margin, bounds["x"] + bw + margin)
    ax.set_ylim(bounds["y"] - margin, bounds["y"] + bh + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    log.info("Rendered 2D board with ratsnest → %s", png_path)
    return png_path



# ---------------------------------------------------------------------------
# KiCad version detection
# ---------------------------------------------------------------------------

def _kicad_has_render_cmd(kicad_cli: str) -> bool:
    """Return True if this kicad-cli supports 'pcb render' (KiCad ≥8)."""
    probe = subprocess.run(
        [kicad_cli, "pcb", "--help"],
        capture_output=True, text=True,
    )
    return "render" in (probe.stdout + probe.stderr)


# ---------------------------------------------------------------------------
# 3-D rendering via trimesh (KiCad 7 fallback: STEP → PNG frames)
# ---------------------------------------------------------------------------

def _export_step(board: Path, out_dir: Path, kicad_cli: str) -> Path | None:
    """Export the board to a STEP file using kicad-cli pcb export step."""
    step_path = out_dir / "board.step"
    cmd = [kicad_cli, "pcb", "export", "step", "-o", str(step_path), str(board)]
    log.info("Exporting STEP model (KiCad 7 fallback) …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("STEP export failed:\n%s", result.stderr)
        return None
    log.info("  → %s", step_path)
    return step_path


def render_3d_frames_trimesh(
    board: Path,
    frames_dir: Path,
    kicad_cli: str,
    out_dir: Path,
    width: int,
    height: int,
    az_start: int,
    az_end: int,
    az_step: int,
    elevation: int,
) -> list[Path]:
    """KiCad 7 fallback: export STEP then render orbit frames with trimesh.

    Uses ``trimesh`` for STEP loading and ``pyrender`` for headless rendering
    if available; falls back to a matplotlib-based isometric wireframe render
    if pyrender is not installed.
    """
    import math
    import numpy as np

    step_path = _export_step(board, out_dir, kicad_cli)
    frames_dir.mkdir(parents=True, exist_ok=True)
    azimuths = list(range(az_start, az_end, az_step))
    paths: list[Path] = []

    if step_path is None:
        log.info("STEP export unavailable — rendering 3D board directly from PCB geometry")
        _render_with_matplotlib(None, board, frames_dir, azimuths, width, height, elevation, paths)
        log.info("  → %d frames in %s", len(paths), frames_dir)
        return paths

    try:

        import trimesh
        log.info("Loading STEP model with trimesh …")
        scene = trimesh.load(str(step_path), force="scene")
        if isinstance(scene, trimesh.Trimesh):
            scene = trimesh.Scene(scene)

        try:
            import pyrender
            _render_with_pyrender(scene, frames_dir, azimuths, width, height, elevation, paths)
        except ImportError:
            log.info("Rendering populated 3D board with matplotlib renderer …")
            _render_with_matplotlib(scene, board, frames_dir, azimuths, width, height, elevation, paths)

    except Exception as exc:
        log.warning("STEP-based 3D load failed (%s) — using direct PCB geometry renderer", exc)
        _render_with_matplotlib(None, board, frames_dir, azimuths, width, height, elevation, paths)

    log.info("  → %d frames in %s", len(paths), frames_dir)
    return paths


def _render_with_pyrender(scene, frames_dir, azimuths, width, height, elevation, paths):
    """Render orbit frames with pyrender (headless, no display required)."""
    import math
    import numpy as np
    import pyrender
    from PIL import Image

    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")  # headless

    # Convert trimesh Scene to pyrender Scene.
    pr_scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.3, 0.3, 0.3])

    # Compute scene bounding sphere for camera placement.
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    radius = np.linalg.norm(bounds[1] - bounds[0]) / 2.0
    cam_dist = radius * 3.5

    camera = pyrender.PerspectiveCamera(yfov=math.radians(30))

    # Directional light from above.
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

    renderer = pyrender.OffscreenRenderer(width, height)
    el_rad = math.radians(elevation)

    log.info("Rendering %d frames with pyrender …", len(azimuths))
    for i, az in enumerate(azimuths):
        az_rad = math.radians(az)
        cx = center[0] + cam_dist * math.cos(el_rad) * math.sin(az_rad)
        cy = center[1] + cam_dist * math.cos(el_rad) * math.cos(az_rad)
        cz = center[2] + cam_dist * math.sin(el_rad)
        eye = np.array([cx, cy, cz])

        forward = center - eye
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up2 = np.cross(right, forward)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up2
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = eye

        cam_node   = pr_scene.add(camera, pose=cam_pose)
        light_node = pr_scene.add(light,  pose=cam_pose)
        color, _ = renderer.render(pr_scene)
        pr_scene.remove_node(cam_node)
        pr_scene.remove_node(light_node)

        frame_path = frames_dir / f"frame_{i:03d}.png"
        Image.fromarray(color).save(frame_path)
        paths.append(frame_path)
        if (i + 1) % 10 == 0:
            log.info("  %d / %d frames", i + 1, len(azimuths))

    renderer.delete()


def _render_with_matplotlib(scene, board_path: Path, frames_dir: Path, azimuths, width, height, elevation, paths):
    """Render 3D orbit frames with PCB substrate + placed components."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#161b22")

    verts_list = []
    faces_list = []
    if scene is not None:
        verts_list = [geom.vertices for geom in scene.geometry.values() if hasattr(geom, "vertices")]
        offset = 0
        for geom in scene.geometry.values():
            if hasattr(geom, "faces"):
                faces_list.append(geom.faces + offset)
                offset += len(geom.vertices)

    if verts_list:
        verts = np.vstack(verts_list)
        faces = np.vstack(faces_list) if faces_list else np.array([]).reshape(0, 3)
        if np.max(verts) < 5.0 and np.max(verts) > 0.001:
            verts = verts * 1000.0  # convert meters to mm
        poly_sub = Poly3DCollection(verts[faces], alpha=0.95, facecolor="#1b4332", edgecolor="#2d6a4f", linewidths=0.3)
        ax.add_collection3d(poly_sub)
        center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        scale = max(np.max(verts.max(axis=0) - verts.min(axis=0)), 10.0)
    else:
        center = np.array([50.0, 50.0, 0.0])
        scale = 60.0

    # Add 3D component boxes from board placement
    try:
        import pcb_parser, json
        design = json.loads(pcb_parser.load_kicad_pcb(str(board_path)))
        bounds = design["board"]["bounds"]
        bw, bh = max(bounds["w"], 20.0), max(bounds["h"], 20.0)

        if not verts_list:
            sub = trimesh.creation.box(extents=[bw, bh, 1.6])
            sub_verts = sub.vertices + np.array([bounds["x"] + bw/2, bounds["y"] + bh/2, -0.8])
            poly_sub = Poly3DCollection(sub_verts[sub.faces], facecolor="#1b4332", edgecolor="#2d6a4f", alpha=0.95, linewidths=0.5)
            ax.add_collection3d(poly_sub)
            center = np.array([bounds["x"] + bw/2, bounds["y"] + bh/2, 0.0])
            scale = max(bw, bh) * 1.3

        for idx, p in enumerate(design["placement"]["positions"]):
            if p is None:
                continue
            x, y = p["position"]
            ref = design["components"][idx]["ref_des"]
            if ref.startswith("U"):
                w, h, depth = 6.0, 5.0, 1.8
                color = "#111111"
            elif ref.startswith(("J", "P")):
                w, h, depth = 8.0, 4.0, 3.5
                color = "#2b2d42"
            else:
                w, h, depth = 2.0, 1.2, 0.8
                color = "#212529"

            cbox = trimesh.creation.box(extents=[w, h, depth])
            cverts = cbox.vertices + np.array([x, y, depth/2])
            poly_c = Poly3DCollection(cverts[cbox.faces], facecolor=color, edgecolor="#6c757d", alpha=0.95, linewidths=0.3)
            ax.add_collection3d(poly_c)
            ax.text(x, y, depth + 0.3, ref, color="#e0e0e0", fontsize=4.5, ha="center", va="center")
    except Exception as exc:
        log.warning("Could not add components to 3D view: %s", exc)

    ax.set_xlim(center[0] - scale * 0.6, center[0] + scale * 0.6)
    ax.set_ylim(center[1] - scale * 0.6, center[1] + scale * 0.6)
    ax.set_zlim(-scale * 0.3, scale * 0.3)
    ax.set_axis_off()

    log.info("Rendering %d orbit frames …", len(azimuths))
    for i, az in enumerate(azimuths):
        ax.view_init(elev=elevation, azim=az)
        frame_path = frames_dir / f"frame_{i:03d}.png"
        fig.savefig(frame_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=dpi)
        paths.append(frame_path)
        if (i + 1) % 10 == 0:
            log.info("  %d / %d frames rendered", i + 1, len(azimuths))

    plt.close(fig)



# ---------------------------------------------------------------------------
# 3-D rendering via kicad-cli pcb render (KiCad ≥8 native raytracer)
# ---------------------------------------------------------------------------

def render_3d_frames(
    board: Path,
    frames_dir: Path,
    kicad_cli: str,
    width: int,
    height: int,
    az_start: int,
    az_end: int,
    az_step: int,
    elevation: int,
    side: str,
) -> list[Path]:
    """Render one PNG per azimuth using ``kicad-cli pcb render``.

    kicad-cli 8 syntax::

        kicad-cli pcb render \\
            --side iso \\
            --width W --height H \\
            --floor no \\
            --background transparent \\
            --zoom 1.0 \\
            --pan 0 0 \\
            --rotate 0 <elevation> <azimuth> \\
            -o out.png board.kicad_pcb

    The ``--rotate`` flag takes ``<x> <y> <z>`` Euler angles (degrees).
    For an orbit animation we sweep the *z* component (azimuth) while holding
    *x* (tilt / elevation) fixed.

    Returns the list of produced PNG paths in frame order.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    azimuths = range(az_start, az_end, az_step)
    paths: list[Path] = []

    # Check whether this kicad-cli supports --rotate (KiCad ≥8.0.3) or uses
    # older --azimuth / --elevation flags.
    probe = subprocess.run(
        [kicad_cli, "pcb", "render", "--help"],
        capture_output=True, text=True,
    )
    help_text = probe.stdout + probe.stderr
    has_rotate_flag   = "--rotate" in help_text
    has_azimuth_flag  = "--azimuth" in help_text
    has_elevation_flag = "--elevation" in help_text

    log.info(
        "Rendering %d frames (%d°–%d° az, step %d°)…",
        len(azimuths), az_start, az_end - az_step, az_step,
    )
    for i, az in enumerate(azimuths):
        frame_path = frames_dir / f"frame_{i:03d}.png"
        cmd: list[str] = [
            kicad_cli, "pcb", "render",
            "--side", side,
            "--width",  str(width),
            "--height", str(height),
            "-o", str(frame_path),
        ]
        if has_rotate_flag:
            cmd += ["--rotate", str(elevation), "0", str(az)]
        else:
            if has_azimuth_flag:
                cmd += ["--azimuth", str(az)]
            if has_elevation_flag:
                cmd += ["--elevation", str(elevation)]

        cmd.append(str(board))
        _run(cmd)
        paths.append(frame_path)
        if (i + 1) % 10 == 0:
            log.info("  %d / %d frames rendered", i + 1, len(azimuths))

    log.info("  → %d frames in %s", len(paths), frames_dir)
    return paths


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def frames_to_video(
    frames_dir: Path,
    out_path: Path,
    framerate: int,
    ffmpeg: str,
) -> Path:
    """Stitch PNG frames into an MP4 with the given framerate."""
    pattern = str(frames_dir / "frame_%03d.png")
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(framerate),
        "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # ensure even dims
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    log.info("Stitching orbit video …")
    _run(cmd)
    log.info("  → %s", out_path)
    return out_path


def svg_to_png(svg_path: Path, png_path: Path, width: int, ffmpeg: str) -> Path:
    """Convert SVG → PNG using ffmpeg, rsvg-convert, or inkscape."""
    # Try ffmpeg directly (uses librsvg when built with it, works out of the box)
    cmd = [
        ffmpeg, "-y",
        "-i", str(svg_path),
        "-update", "1",
        "-vf", f"scale={width}:{width}:force_original_aspect_ratio=decrease,pad={width}:{width}:(ow-iw)/2:(oh-ih)/2:color=black",
        str(png_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and png_path.exists():
        log.info("Converted SVG → PNG via ffmpeg: %s", png_path)
        return png_path

    # Fallback: rsvg-convert
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        cmd = [rsvg, "-w", str(width), "-o", str(png_path), str(svg_path)]
        log.info("Converting SVG → PNG via rsvg-convert …")
        _run(cmd)
        return png_path

    # Fallback: Inkscape
    inkscape = shutil.which("inkscape")
    if inkscape:
        cmd = [inkscape, f"--export-width={width}", f"--export-filename={png_path}", str(svg_path)]
        log.info("Converting SVG → PNG via inkscape …")
        _run(cmd)
        return png_path

    # Fallback: cairosvg Python package
    try:
        import cairosvg
        log.info("Converting SVG → PNG via cairosvg …")
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=width)
        return png_path
    except ImportError:
        pass

    log.warning("Cannot convert SVG→PNG. Split-screen will use a blank left panel.")
    return png_path



def _get_video_duration(video_path: Path) -> float | None:
    """Extract video duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    res = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return float(res.stdout.strip())
    except (ValueError, TypeError):
        return None


def compose_splitscreen(
    left_png: Path | None,
    orbit_mp4: Path,
    out_path: Path,
    ffmpeg: str,
    width: int,
    height: int,
) -> Path:
    """Combine a static 2D PNG (left) with the orbit video (right) side-by-side.

    If *left_png* does not exist, a blank black panel is used on the left.
    """
    if left_png is None or not left_png.exists():
        # Generate a blank panel.
        left_png = out_path.parent / "_blank_left.png"
        cmd_blank = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", f"color=black:size={width}x{height}:rate=1",
            "-frames:v", "1",
            str(left_png),
        ]
        _run(cmd_blank)

    # Scale both inputs to the same height before hstack.
    filter_complex = (
        f"[0:v]scale=-2:{height}[left];"
        f"[1:v]scale=-2:{height}[right];"
        "[left][right]hstack=inputs=2[out]"
    )
    duration = _get_video_duration(orbit_mp4)
    cmd = [
        ffmpeg, "-y",
        "-loop", "1", "-i", str(left_png),
        "-i", str(orbit_mp4),
        "-filter_complex", filter_complex,
        "-map", "[out]",
    ]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    else:
        cmd += ["-shortest"]

    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    log.info("Composing split-screen …")
    _run(cmd)
    log.info("  → %s", out_path)
    return out_path



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render SCPT-RL placed board to 2D SVG + 3D orbit video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--board",   required=True, help="Path to .kicad_pcb (e.g. output of bake_placement.py)")
    parser.add_argument("--out-dir", required=True, help="Output directory for all render artefacts")
    parser.add_argument("--width",   type=int, default=1280, help="Frame width in pixels (default 1280)")
    parser.add_argument("--height",  type=int, default=1280, help="Frame height in pixels (default 1280)")
    parser.add_argument("--azimuth-start", type=int, default=0,   help="Start azimuth in degrees (default 0)")
    parser.add_argument("--azimuth-end",   type=int, default=360, help="End azimuth in degrees exclusive (default 360)")
    parser.add_argument("--azimuth-step",  type=int, default=5,   help="Azimuth step in degrees (default 5)")
    parser.add_argument("--elevation",     type=int, default=30,  help="Camera elevation in degrees (default 30)")
    parser.add_argument("--side",   default="iso", choices=["iso", "top", "bottom", "front", "back", "left", "right"],
                        help="3D render side/view preset (default iso)")
    parser.add_argument("--framerate", type=int, default=30, help="Output video framerate (default 30)")
    parser.add_argument("--no-2d",  action="store_true", help="Skip 2D SVG export")
    parser.add_argument("--no-3d",  action="store_true", help="Skip 3D frame rendering")
    parser.add_argument("--splitscreen-only", action="store_true",
                        help="Skip 2D+3D render; just compose split-screen from existing files")
    parser.add_argument("--rollout", action="store_true",
                        help="Generate dynamic step-by-step placement rollout & progressive routing animation matching video_2026-09-16_13-16-27.mp4")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration in seconds for rollout animation (default 15.0)")
    args = parser.parse_args(argv)

    board    = Path(args.board)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rollout:
        from scripts.render_rollout_animation import render_rollout_video
        render_rollout_video(board, out_dir, fps=args.framerate, duration=args.duration)
        return

    kicad_cli = _require("kicad-cli")
    ffmpeg    = _require("ffmpeg")

    frames_dir  = out_dir / "frames"
    svg_path    = out_dir / "board_2d.svg"
    left_png    = out_dir / "board_2d.png"
    orbit_mp4   = out_dir / "orbit.mp4"
    split_mp4   = out_dir / "splitscreen.mp4"

    # ── 2D ──────────────────────────────────────────────────────────────────
    if not args.no_2d and not args.splitscreen_only:
        render_2d(board, out_dir, kicad_cli, args.width)
        if svg_path.exists():
            svg_to_png(svg_path, left_png, args.width, ffmpeg)

    # ── 3D frames ────────────────────────────────────────────────────────────
    if not args.no_3d and not args.splitscreen_only:
        has_render = _kicad_has_render_cmd(kicad_cli)
        if has_render:
            log.info("Using KiCad ≥8 native 'pcb render' for 3D frames")
            render_3d_frames(
                board, frames_dir, kicad_cli,
                args.width, args.height,
                args.azimuth_start, args.azimuth_end, args.azimuth_step,
                args.elevation, args.side,
            )
        else:
            log.info("KiCad 7 detected — using STEP + trimesh/matplotlib fallback for 3D frames")
            render_3d_frames_trimesh(
                board, frames_dir, kicad_cli, out_dir,
                args.width, args.height,
                args.azimuth_start, args.azimuth_end, args.azimuth_step,
                args.elevation,
            )
        if any(frames_dir.iterdir()) if frames_dir.exists() else False:
            frames_to_video(frames_dir, orbit_mp4, args.framerate, ffmpeg)

    # ── Split-screen ─────────────────────────────────────────────────────────
    if orbit_mp4.exists():
        compose_splitscreen(
            left_png if left_png.exists() else None,
            orbit_mp4,
            split_mp4,
            ffmpeg,
            args.width,
            args.height,
        )
    else:
        log.warning("orbit.mp4 not found — skipping split-screen composition")

    log.info("All render artefacts written to: %s", out_dir)
    _print_summary(out_dir, svg_path, orbit_mp4, split_mp4)


def _print_summary(out_dir: Path, svg: Path, orbit: Path, split: Path) -> None:
    print("\n" + "=" * 62)
    print("SCPT-RL Render Summary")
    print("=" * 62)
    for label, path in [("2D SVG", svg), ("Orbit MP4", orbit), ("Split-screen", split)]:
        status = "✓" if path.exists() else "✗ (not produced)"
        size   = f" ({path.stat().st_size // 1024} KB)" if path.exists() else ""
        print(f"  {label:<14}: {path.name}{size}  {status}")
    print(f"\n  Output dir: {out_dir}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
