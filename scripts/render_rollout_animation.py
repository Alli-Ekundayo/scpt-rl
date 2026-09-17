#!/usr/bin/env python3
"""SCPT-RL Dynamic Placement Rollout & Routing Animation Renderer.

Produces a 1280x640 split-screen video matching the visual language and dynamics of
video_2026-09-16_13-16-27.mp4:

Left Panel (640x640):
  - 2D layout on white background with board outline and corner mounting holes.
  - Unplaced components neatly organized outside the board (right & bottom margins).
  - Footprints rendered with individual copper pads, courtyard outlines, and refdes text.
  - Dynamic Minimum Spanning Tree (MST) cyan-blue airwires (ratsnest) connecting pins.
  - Step-by-step placement rollout: components fly from margins onto the board.
  - Dynamic camera transition: smoothly zooms in to frame the board as placement completes.
  - Progressive routing phase: copper tracks (F.Cu red/orange, B.Cu blue) and vias
    appear net-by-net as ratsnest lines dissolve.

Right Panel (640x640):
  - Photorealistic 3D OpenGL view in fixed isometric perspective.
  - Forest green PCB substrate with soft floor shadow on clean white background.
  - Unplaced 3D component packages resting on the white floor outside the board.
  - Dynamic arc flight: components lift off the floor and land onto the PCB.
  - Placed 3D components (electrolytic cans, QFPs, headers, terminals, passives)
    with metallic leads, pins, and pads.
  - Surface copper tracks and vias appear during routing phase.

Usage:
  python scripts/render_rollout_animation.py \
      --board dataset/base_raw/Driverino-Shield.kicad_pcb \
      --out-dir renders/driverino_rollout/ \
      --fps 30 \
      --duration 15.0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Configure headless OpenGL for pyrender
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pyrender
    import trimesh
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False

try:
    import scipy.sparse.csgraph as csgraph
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scpt.rollout_anim")


# ---------------------------------------------------------------------------
# Board Data Loading & Assembly Ordering
# ---------------------------------------------------------------------------

def load_board_data(board_path: str | Path) -> dict[str, Any]:
    """Load board IR using pcb_parser with regex extraction for tracks and vias."""
    import pcb_parser

    board_path = Path(board_path)
    design = json.loads(pcb_parser.load_kicad_pcb(str(board_path)))

    # Check if router-generated routes already exist in the design IR
    if "routes" in design and (design["routes"].get("segments") or design["routes"].get("vias")):
        segments = [
            {
                "start": (float(s["start"][0]), float(s["start"][1])),
                "end": (float(s["end"][0]), float(s["end"][1])),
                "width": float(s["width"]),
                "layer": s["layer"],
                "net": int(s["net"]),
            }
            for s in design["routes"].get("segments", [])
        ]
        vias = [
            {
                "pos": (float(v["at"][0]), float(v["at"][1])),
                "size": float(v["size"]),
                "drill": float(v["drill"]),
                "net": int(v["net"]),
            }
            for v in design["routes"].get("vias", [])
        ]
    else:
        # Raw text extraction for copper segments and vias
        raw_text = board_path.read_text(encoding="utf-8", errors="ignore")

        # Extract segments: (segment (start X Y) (end X Y) (width W) (layer L) (net N))
        seg_pattern = re.compile(
            r'\(segment\s+\(start\s+([\d\.-]+)\s+([\d\.-]+)\)\s+\(end\s+([\d\.-]+)\s+([\d\.-]+)\)'
            r'\s+\(width\s+([\d\.-]+)\)\s+\(layer\s+\"?([^\s\"]+)\"?\)\s+\(net\s+(\d+)\)'
        )
        segments = []
        for m in seg_pattern.finditer(raw_text):
            x1, y1, x2, y2, w, layer, net = m.groups()
            segments.append({
                "start": (float(x1), float(y1)),
                "end": (float(x2), float(y2)),
                "width": float(w),
                "layer": layer,
                "net": int(net),
            })

        # Extract vias: (via (at X Y) (size S) (drill D) (layers L1 L2) (net N))
        via_pattern = re.compile(
            r'\(via\s+(?:\([^\)]+\)\s+)*\(at\s+([\d\.-]+)\s+([\d\.-]+)\)\s+\(size\s+([\d\.-]+)\)'
            r'(?:\s+\(drill\s+([\d\.-]+)\))?(?:\s+\(layers\s+[^\)]+\))?\s+\(net\s+(\d+)\)'
        )
        vias = []
        for m in via_pattern.finditer(raw_text):
            vx, vy, sz, dr, net = m.groups()
            vias.append({
                "pos": (float(vx), float(vy)),
                "size": float(sz),
                "drill": float(dr) if dr else float(sz) * 0.5,
                "net": int(net),
            })

    design["segments"] = segments
    design["vias"] = vias

    # Assembly Order:
    # 1. Chip passives (R, C, D)
    # 2. ICs (U), inductors (L), fuses (F), jumpers (JP)
    # 3. Electrolytic can capacitors (C1, C2)
    # 4. Headers and terminal connectors (P)
    comps = design["components"]

    def comp_priority(item: tuple[int, dict]) -> tuple[int, str]:
        idx, c = item
        ref = c.get("ref_des", "")
        val = c.get("value", "").lower()
        if ref.startswith(("R", "D")):
            return (0, ref)
        elif ref.startswith("C"):
            if "uf" in val or "elect" in val or ref in ["C1", "C2"]:
                return (2, ref)
            return (0, ref)
        elif ref.startswith(("U", "L", "F", "JP")):
            return (1, ref)
        elif ref.startswith("P"):
            return (3, ref)
        return (4, ref)

    indexed_comps = list(enumerate(comps))
    indexed_comps.sort(key=comp_priority)
    design["assembly_order"] = [idx for idx, _ in indexed_comps]

    return design


# ---------------------------------------------------------------------------
# Staging Slot Assignment (Outside the Board)
# ---------------------------------------------------------------------------

def compute_unplaced_slots(
    components: list[dict],
    bounds: dict[str, float],
) -> dict[int, tuple[float, float]]:
    """Organize unplaced components neatly into Right Margin and Bottom Margin."""
    bx = bounds["x"]
    by = bounds["y"]
    bw = max(bounds["w"], 20.0)
    bh = max(bounds["h"], 20.0)

    unplaced_slots = {}

    large_comps = []
    passive_comps = []

    for i, c in enumerate(components):
        ref = c.get("ref_des", "")
        val = c.get("value", "").lower()
        if ref.startswith(("P", "U", "L", "JP")) or (ref.startswith("C") and ("uf" in val or ref in ["C1", "C2"])):
            large_comps.append(i)
        else:
            passive_comps.append(i)

    # Right Margin: 2 columns for large components
    for idx_in_large, comp_idx in enumerate(large_comps):
        col = idx_in_large % 2
        row = idx_in_large // 2
        x = bx + bw + 9.0 + col * 15.0
        y = by + 2.0 + row * 7.5
        unplaced_slots[comp_idx] = (x, y)

    # Bottom Margin: 8 columns for chip passives
    for idx_in_pass, comp_idx in enumerate(passive_comps):
        col = idx_in_pass % 8
        row = idx_in_pass // 8
        x = bx + 4.0 + col * 8.5
        y = by + bh + 7.0 + row * 5.2
        unplaced_slots[comp_idx] = (x, y)

    return unplaced_slots


# ---------------------------------------------------------------------------
# 2D Panel Renderer (PIL) with Dynamic Zoom & Vector Graphics
# ---------------------------------------------------------------------------

class Renderer2D:
    def __init__(self, design: dict[str, Any], unplaced_slots: dict[int, tuple[float, float]], size: int = 640):
        self.design = design
        self.unplaced_slots = unplaced_slots
        self.size = size

        self.bounds = design["board"]["bounds"]
        self.bx = self.bounds["x"]
        self.by = self.bounds["y"]
        self.bw = max(self.bounds["w"], 20.0)
        self.bh = max(self.bounds["h"], 20.0)

        # Precompute pad offsets per component
        self.comp_pads = []
        for c in design["components"]:
            pads = []
            fp = c.get("footprint", {})
            for p in fp.get("pads", []):
                pads.append({
                    "pos": p.get("pos", [0.0, 0.0]),
                    "size": p.get("size", [1.2, 1.2]),
                    "shape": p.get("shape", "rect"),
                    "layers": p.get("layers", ["F.Cu"]),
                })
            if not pads:
                pads = [
                    {"pos": [-0.8, 0.0], "size": [0.8, 1.2], "shape": "rect", "layers": ["F.Cu"]},
                    {"pos": [0.8, 0.0], "size": [0.8, 1.2], "shape": "rect", "layers": ["F.Cu"]},
                ]
            self.comp_pads.append(pads)

    def get_transform(self, zoom_progress: float) -> tuple[float, float, float]:
        """Compute scale and origin based on zoom_progress."""
        # Wide viewport (board + margins)
        w_wide = self.bw * 1.72
        h_wide = self.bh * 1.72
        scale_wide = min((self.size - 40) / w_wide, (self.size - 40) / h_wide)
        min_x_wide = self.bx - 4.0
        min_y_wide = self.by - 4.0

        # Tight board viewport (centered on board with padding)
        scale_tight = min((self.size - 50) / self.bw, (self.size - 50) / self.bh)
        pad_x = ((self.size - 50) / scale_tight - self.bw) / 2.0
        pad_y = ((self.size - 50) / scale_tight - self.bh) / 2.0
        min_x_tight = self.bx - pad_x - 2.0
        min_y_tight = self.by - pad_y - 2.0

        ease = 0.5 - 0.5 * math.cos(math.pi * zoom_progress)
        scale = scale_wide + (scale_tight - scale_wide) * ease
        min_x = min_x_wide + (min_x_tight - min_x_wide) * ease
        min_y = min_y_wide + (min_y_tight - min_y_wide) * ease

        return scale, min_x, min_y

    def world_to_px(self, x: float, y: float, scale: float, min_x: float, min_y: float) -> tuple[int, int]:
        px = 20 + (x - min_x) * scale
        py = 20 + (y - min_y) * scale
        return int(px), int(py)

    def render_frame(
        self,
        placed_indices: set[int],
        interpolating: dict[int, tuple[float, float]] | None = None,
        zoom_progress: float = 0.0,
        routed_fraction: float = 0.0,
    ) -> Image.Image:
        """Render a single 2D layout frame."""
        img = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        scale, min_x, min_y = self.get_transform(zoom_progress)

        # 1. Board substrate outline
        p0 = self.world_to_px(self.bx, self.by, scale, min_x, min_y)
        p1 = self.world_to_px(self.bx + self.bw, self.by + self.bh, scale, min_x, min_y)
        draw.rectangle([p0, p1], fill=(248, 248, 250), outline=(130, 130, 135), width=2)

        # 2. Corner mounting holes
        for hx, hy in [
            (self.bx + 3.5, self.by + 3.5),
            (self.bx + self.bw - 3.5, self.by + 3.5),
            (self.bx + 3.5, self.by + self.bh - 3.5),
            (self.bx + self.bw - 3.5, self.by + self.bh - 3.5),
        ]:
            cx, cy = self.world_to_px(hx, hy, scale, min_x, min_y)
            r_out = max(int(1.8 * scale), 3)
            r_in = max(int(1.1 * scale), 2)
            draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=(210, 160, 45), outline=(110, 80, 20), width=1)
            draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(30, 30, 30))

        # 3. Component positions
        current_positions: dict[int, tuple[float, float]] = {}
        for i, comp in enumerate(self.design["components"]):
            if interpolating and i in interpolating:
                current_positions[i] = interpolating[i]
            elif i in placed_indices:
                pos = self.design["placement"]["positions"][i]
                if pos is not None:
                    current_positions[i] = tuple(pos["position"])
                else:
                    current_positions[i] = self.unplaced_slots[i]
            else:
                current_positions[i] = self.unplaced_slots[i]

        # 4. Routed copper traces
        routed_nets = set()
        if routed_fraction > 0.0:
            segments = self.design.get("segments", [])
            n_show_segs = int(len(segments) * min(routed_fraction, 1.0))
            for seg in segments[:n_show_segs]:
                routed_nets.add(seg["net"])
                sx, sy = self.world_to_px(*seg["start"], scale, min_x, min_y)
                ex, ey = self.world_to_px(*seg["end"], scale, min_x, min_y)
                color = (214, 60, 38) if "F.Cu" in seg["layer"] else (38, 105, 218)
                w = max(int(seg["width"] * scale * 0.9), 2)
                draw.line([(sx, sy), (ex, ey)], fill=color, width=w)

            # Vias
            vias = self.design.get("vias", [])
            n_show_vias = int(len(vias) * min(routed_fraction, 1.0))
            for via in vias[:n_show_vias]:
                vx, vy = self.world_to_px(*via["pos"], scale, min_x, min_y)
                vr = max(int(via["size"] * scale * 0.45), 2)
                draw.ellipse([vx - vr, vy - vr, vx + vr, vy + vr], fill=(215, 165, 45), outline=(40, 40, 40), width=1)

        # 5. Component footprints, pads, and refdes text
        pad_world_coords: dict[tuple[int, int], tuple[float, float]] = {}
        for i, comp in enumerate(self.design["components"]):
            cx, cy = current_positions[i]
            pads = self.comp_pads[i]
            ref = comp["ref_des"]

            # Courtyard bounding box
            if len(pads) > 2:
                xs = [p["pos"][0] for p in pads]
                ys = [p["pos"][1] for p in pads]
                c_min_x = cx + min(xs) - 0.6
                c_max_x = cx + max(xs) + 0.6
                c_min_y = cy + min(ys) - 0.6
                c_max_y = cy + max(ys) + 0.6
                bx0, by0 = self.world_to_px(c_min_x, c_min_y, scale, min_x, min_y)
                bx1, by1 = self.world_to_px(c_max_x, c_max_y, scale, min_x, min_y)
                draw.rectangle([bx0, by0, bx1, by1], outline=(175, 175, 180), width=1)

            # Draw individual pads
            for p_idx, pad in enumerate(pads):
                ppos = pad["pos"]
                psize = pad["size"]
                wx = cx + ppos[0]
                wy = cy + ppos[1]
                pad_world_coords[(i, p_idx)] = (wx, wy)

                px0, py0 = self.world_to_px(wx - psize[0] / 2.0, wy - psize[1] / 2.0, scale, min_x, min_y)
                px1, py1 = self.world_to_px(wx + psize[0] / 2.0, wy + psize[1] / 2.0, scale, min_x, min_y)

                pad_color = (196, 92, 48) if "F.Cu" in pad["layers"] else (58, 128, 218)
                draw.rectangle([px0, py0, max(px1, px0 + 1), max(py1, py0 + 1)], fill=pad_color, outline=(130, 58, 28))

            # Reference text
            tx, ty = self.world_to_px(cx, cy, scale, min_x, min_y)
            draw.text((tx - len(ref) * 2, ty - 6), ref, fill=(35, 35, 35))

        # 6. Dynamic MST Ratsnest airwires
        for net_idx, net in enumerate(self.design.get("nets", [])):
            if net_idx in routed_nets:
                continue

            net_pads = [pad_world_coords[tuple(p)] for p in net.get("pads", []) if tuple(p) in pad_world_coords]
            if len(net_pads) > 1:
                pts = np.array(net_pads)
                if len(pts) == 2:
                    pA = self.world_to_px(pts[0][0], pts[0][1], scale, min_x, min_y)
                    pB = self.world_to_px(pts[1][0], pts[1][1], scale, min_x, min_y)
                    draw.line([pA, pB], fill=(114, 184, 232), width=1)
                elif HAS_SCIPY:
                    dist_matrix = squareform(pdist(pts))
                    mst = csgraph.minimum_spanning_tree(dist_matrix).toarray()
                    edges = np.argwhere(mst > 0)
                    for iA, iB in edges:
                        pA = self.world_to_px(pts[iA][0], pts[iA][1], scale, min_x, min_y)
                        pB = self.world_to_px(pts[iB][0], pts[iB][1], scale, min_x, min_y)
                        draw.line([pA, pB], fill=(114, 184, 232), width=1)

        return img


# ---------------------------------------------------------------------------
# 3D Panel Renderer (pyrender + EGL) with Shading, Arc Flight & Models
# ---------------------------------------------------------------------------

class Renderer3D:
    def __init__(self, design: dict[str, Any], unplaced_slots: dict[int, tuple[float, float]], size: int = 640):
        self.design = design
        self.unplaced_slots = unplaced_slots
        self.size = size

        self.bounds = design["board"]["bounds"]
        self.bw = max(self.bounds["w"], 20.0)
        self.bh = max(self.bounds["h"], 20.0)
        self.center_x = self.bounds["x"] + self.bw / 2.0
        self.center_y = self.bounds["y"] + self.bh / 2.0

        self.pcb_th = 1.6

        # Check & initialize hardware OpenGL (pyrender) or fall back to software 3D
        self.use_pyrender = False
        if HAS_PYRENDER:
            try:
                # Patch PyRender EGL device query for Docker / Kaggle containers
                try:
                    import pyrender.platforms.egl as pyrender_egl
                    orig_query = pyrender_egl.query_devices
                    def safe_query_devices():
                        devs = orig_query()
                        return devs if devs else [pyrender_egl.EGLDevice(None)]
                    pyrender_egl.query_devices = safe_query_devices
                except Exception:
                    pass

                # Probe offscreen renderer
                probe = pyrender.OffscreenRenderer(16, 16)
                probe.delete()
                self.use_pyrender = True
                log.info("3D Backend: Hardware OpenGL (pyrender/EGL) active")
            except Exception as e:
                log.warning("Hardware OpenGL unavailable (%s) — using high-performance Software 3D renderer", e)
                self.use_pyrender = False
        else:
            log.info("PyRender not installed — using high-performance Software 3D renderer")

        self._init_materials()
        self._prebuild_component_meshes()

    def _init_materials(self):
        # Base RGB colors
        self.color_map = {
            "substrate": (28, 110, 52),
            "silver": (225, 225, 230),
            "ic": (28, 28, 28),
            "cap": (165, 115, 65),
            "res": (32, 32, 32),
            "copper": (225, 175, 60),
            "shadow": (230, 230, 234),
            "terminal": (30, 102, 184),
            "led_orange": (245, 140, 25),
            "led_blue": (50, 140, 245),
            "led_green": (40, 215, 65),
        }

        if HAS_PYRENDER:
            self.sub_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.08, 0.33, 0.15, 1.0], roughnessFactor=0.35, metallicFactor=0.05
            )
            self.silver_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.88, 0.88, 0.90, 1.0], roughnessFactor=0.15, metallicFactor=0.92
            )
            self.ic_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.11, 0.11, 0.11, 1.0], roughnessFactor=0.55, metallicFactor=0.08
            )
            self.cap_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.65, 0.45, 0.25, 1.0], roughnessFactor=0.45, metallicFactor=0.05
            )
            self.res_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.12, 0.12, 0.12, 1.0], roughnessFactor=0.42, metallicFactor=0.08
            )
            self.copper_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.88, 0.68, 0.24, 1.0], roughnessFactor=0.22, metallicFactor=0.85
            )
            self.shadow_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.72, 0.72, 0.72, 0.45], roughnessFactor=1.0
            )
            self.terminal_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.12, 0.40, 0.72, 1.0], roughnessFactor=0.4, metallicFactor=0.1
            )
            self.led_orange_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.95, 0.55, 0.10, 1.0], roughnessFactor=0.25, metallicFactor=0.2
            )
            self.led_blue_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.20, 0.55, 0.95, 1.0], roughnessFactor=0.25, metallicFactor=0.2
            )
            self.led_green_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.15, 0.85, 0.25, 1.0], roughnessFactor=0.25, metallicFactor=0.2
            )
        else:
            self.sub_mat = "substrate"
            self.silver_mat = "silver"
            self.ic_mat = "ic"
            self.cap_mat = "cap"
            self.res_mat = "res"
            self.copper_mat = "copper"
            self.shadow_mat = "shadow"
            self.terminal_mat = "terminal"
            self.led_orange_mat = "led_orange"
            self.led_blue_mat = "led_blue"
            self.led_green_mat = "led_green"

    def _prebuild_component_meshes(self):
        """Construct detailed 3D geometric models for each component."""
        self.comp_models = []
        for comp in self.design["components"]:
            ref = comp["ref_des"]
            val = comp.get("value", "").lower()
            fp = comp.get("footprint", {})

            parts = []
            if ref.startswith("C") and ("uf" in val or "elect" in val or ref in ["C1", "C2"]):
                # Electrolytic Can Capacitor: silver cylinder + black vent top + black base
                can_r = 4.2 if ref == "C2" else 3.0
                can_h = 8.0 if ref == "C2" else 6.0
                can = trimesh.creation.cylinder(radius=can_r, height=can_h)
                can.apply_translation([0, 0, can_h / 2.0])
                parts.append((can, self.silver_mat))
                # Top black vent notch
                top = trimesh.creation.cylinder(radius=can_r * 0.96, height=0.2)
                top.apply_translation([0, 0, can_h + 0.1])
                parts.append((top, self.ic_mat))
                # Square plastic base
                base = trimesh.creation.box(extents=[can_r * 2.1, can_r * 2.1, 0.6])
                base.apply_translation([0, 0, 0.3])
                parts.append((base, self.ic_mat))

            elif ref.startswith("U"):
                # QFP / QFN Microcontroller
                body_sz = 7.5
                body = trimesh.creation.box(extents=[body_sz, body_sz, 1.2])
                body.apply_translation([0, 0, 0.6])
                parts.append((body, self.ic_mat))
                # Silver pins/leads along all 4 sides
                for side in range(4):
                    for p in np.linspace(-body_sz * 0.38, body_sz * 0.38, 7):
                        lead = trimesh.creation.box(extents=[0.35, 1.4, 0.15])
                        lead.apply_translation([0, 0, 0.1])
                        if side == 0:
                            lead.apply_translation([p, body_sz / 2.0 + 0.6, 0])
                        elif side == 1:
                            lead.apply_translation([p, -body_sz / 2.0 - 0.6, 0])
                        elif side == 2:
                            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
                            lead.apply_transform(rot)
                            lead.apply_translation([body_sz / 2.0 + 0.6, p, 0])
                        elif side == 3:
                            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
                            lead.apply_transform(rot)
                            lead.apply_translation([-body_sz / 2.0 - 0.6, p, 0])
                        parts.append((lead, self.silver_mat))

            elif ref.startswith("P") and any(k in val for k in ["screw", "terminal", "blue", "conn_01x02", "conn_01x03"]):
                # Terminal Block (Screw connector): Blue block + silver top screw cylinders
                n_term = 3 if "03" in val else 2
                tb_w = 5.08 * n_term
                tb_d = 7.5
                tb_h = 10.0
                body = trimesh.creation.box(extents=[tb_w, tb_d, tb_h])
                body.apply_translation([0, 0, tb_h / 2.0])
                parts.append((body, self.terminal_mat))
                # Screw cylinder indentations
                for pin_i in range(n_term):
                    off_x = (pin_i - (n_term - 1) / 2.0) * 5.08
                    screw = trimesh.creation.cylinder(radius=1.3, height=0.4)
                    screw.apply_translation([off_x, 0, tb_h + 0.05])
                    parts.append((screw, self.silver_mat))

            elif ref.startswith("P"):
                # Pin Header: Black base plastic strip + tall silver vertical pins
                n_pins = 8 if ref in ["P1", "P2", "P3", "P4"] else 6
                strip_len = n_pins * 2.54
                base = trimesh.creation.box(extents=[strip_len, 2.4, 2.5])
                base.apply_translation([0, 0, 1.25])
                parts.append((base, self.ic_mat))
                for pin_i in range(n_pins):
                    off_x = (pin_i - (n_pins - 1) / 2.0) * 2.54
                    pin = trimesh.creation.box(extents=[0.64, 0.64, 8.5])
                    pin.apply_translation([off_x, 0, 4.25])
                    parts.append((pin, self.silver_mat))

            elif ref.startswith("D"):
                # Diode / LED: Colored package
                lens_color = self.led_orange_mat if ref == "D1" else (self.led_blue_mat if ref == "D2" else self.led_green_mat)
                body = trimesh.creation.box(extents=[2.0, 1.2, 1.0])
                body.apply_translation([0, 0, 0.5])
                parts.append((body, lens_color))
                for end in [-0.85, 0.85]:
                    term = trimesh.creation.box(extents=[0.4, 1.2, 1.0])
                    term.apply_translation([end, 0, 0.5])
                    parts.append((term, self.silver_mat))

            elif ref.startswith("L"):
                # Inductor: Charcoal/dark gray wirewound body
                body = trimesh.creation.box(extents=[4.5, 4.5, 3.2])
                body.apply_translation([0, 0, 1.6])
                parts.append((body, self.res_mat))

            else:
                # Chip Resistor (0805/0603) or Ceramic Cap (0805)
                chip_l, chip_w, chip_h = (2.0, 1.2, 0.8) if ref.startswith("R") else (2.0, 1.25, 0.9)
                m_color = self.res_mat if ref.startswith("R") else self.cap_mat
                body = trimesh.creation.box(extents=[chip_l * 0.55, chip_w, chip_h])
                body.apply_translation([0, 0, chip_h / 2.0])
                parts.append((body, m_color))
                for end in [-chip_l * 0.4, chip_l * 0.4]:
                    term = trimesh.creation.box(extents=[chip_l * 0.25, chip_w, chip_h])
                    term.apply_translation([end, 0, chip_h / 2.0])
                    parts.append((term, self.silver_mat))

            self.comp_models.append(parts)

    def get_camera_pose(self, zoom_progress: float):
        """Compute camera pose with smooth easing from wide staging view to tight board view."""
        # Wide view: offset slightly towards staging margins
        dist_wide = max(self.bw, self.bh) * 2.75
        target_wide = np.array([14.0, -5.0, 0.0])

        # Tight view: centered directly on PCB
        dist_tight = max(self.bw, self.bh) * 2.05
        target_tight = np.array([0.0, 0.0, 0.0])

        ease = 0.5 - 0.5 * math.cos(math.pi * zoom_progress)
        cam_dist = dist_wide + (dist_tight - dist_wide) * ease
        target = target_wide + (target_tight - target_wide) * ease

        elev = np.radians(36)
        azim = np.radians(45)
        cam_x = target[0] + cam_dist * np.cos(elev) * np.sin(azim)
        cam_y = target[1] - cam_dist * np.cos(elev) * np.cos(azim)
        cam_z = target[2] + cam_dist * np.sin(elev)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up_cam = np.cross(right, forward)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up_cam
        cam_pose[:3, 2] = -forward
        cam_pose[:3, 3] = cam_pos

        cam = pyrender.PerspectiveCamera(yfov=np.radians(36)) if HAS_PYRENDER else None
        return cam_pose, cam

    def _render_pyrender(
        self,
        placed_indices: set[int],
        interpolating: dict[int, tuple[float, float, float]] | None = None,
        zoom_progress: float = 0.0,
        routed_fraction: float = 0.0,
    ) -> Image.Image:
        """Render frame using hardware OpenGL / pyrender."""
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.50, 0.50, 0.50])

        # 1. PCB Substrate
        sub_mesh = trimesh.creation.box(extents=[self.bw, self.bh, self.pcb_th])
        sub_mesh.apply_translation([0, 0, self.pcb_th / 2.0])
        scene.add(pyrender.Mesh.from_trimesh(sub_mesh, material=self.sub_mat))

        # 2. Soft floor shadow underneath PCB
        shadow_mesh = trimesh.creation.box(extents=[self.bw * 1.05, self.bh * 1.05, 0.05])
        shadow_mesh.apply_translation([2.0, -2.0, -0.05])
        scene.add(pyrender.Mesh.from_trimesh(shadow_mesh, material=self.shadow_mat))

        # 3. Corner mounting holes
        for hx, hy in [
            (-self.bw / 2 + 3.5, -self.bh / 2 + 3.5),
            (self.bw / 2 - 3.5, -self.bh / 2 + 3.5),
            (-self.bw / 2 + 3.5, self.bh / 2 - 3.5),
            (self.bw / 2 - 3.5, self.bh / 2 - 3.5),
        ]:
            ring = trimesh.creation.cylinder(radius=3.2 / 2, height=self.pcb_th + 0.05)
            ring.apply_translation([hx, hy, self.pcb_th / 2.0])
            scene.add(pyrender.Mesh.from_trimesh(ring, material=self.copper_mat))

        # 4. Routed copper traces & vias in 3D
        if routed_fraction > 0.0:
            segments = self.design.get("segments", [])
            n_show_segs = int(len(segments) * min(routed_fraction, 1.0))
            for seg in segments[:n_show_segs]:
                if "F.Cu" in seg.get("layer", ""):
                    sx, sy = seg["start"]
                    ex, ey = seg["end"]
                    rx1, ry1 = sx - self.center_x, -(sy - self.center_y)
                    rx2, ry2 = ex - self.center_x, -(ey - self.center_y)
                    length = math.hypot(rx2 - rx1, ry2 - ry1)
                    if length > 0.1:
                        trace = trimesh.creation.box(extents=[length, max(seg["width"], 0.4), 0.05])
                        ang = math.atan2(ry2 - ry1, rx2 - rx1)
                        rot_mat = trimesh.transformations.rotation_matrix(ang, [0, 0, 1])
                        trace.apply_transform(rot_mat)
                        trace.apply_translation([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0, self.pcb_th + 0.025])
                        scene.add(pyrender.Mesh.from_trimesh(trace, material=self.copper_mat))

            vias = self.design.get("vias", [])
            n_show_vias = int(len(vias) * min(routed_fraction, 1.0))
            for via in vias[:n_show_vias]:
                vx, vy = via["pos"]
                rx, ry = vx - self.center_x, -(vy - self.center_y)
                via_mesh = trimesh.creation.cylinder(radius=0.45, height=0.06)
                via_mesh.apply_translation([rx, ry, self.pcb_th + 0.03])
                scene.add(pyrender.Mesh.from_trimesh(via_mesh, material=self.copper_mat))

        # 5. Add 3D components
        for i, comp in enumerate(self.design["components"]):
            is_placed = i in placed_indices
            is_interp = interpolating and i in interpolating

            if is_interp:
                cx, cy, cz = interpolating[i]
            elif is_placed:
                pos = self.design["placement"]["positions"][i]
                if pos is not None:
                    cx, cy = pos["position"]
                    cz = self.pcb_th
                else:
                    cx, cy = self.unplaced_slots[i]
                    cz = 0.0
            else:
                cx, cy = self.unplaced_slots[i]
                cz = 0.0

            rel_x = cx - self.center_x
            rel_y = -(cy - self.center_y)

            parts = self.comp_models[i]
            for mesh_def, mat in parts:
                m_copy = mesh_def.copy()
                m_copy.apply_translation([rel_x, rel_y, cz])
                scene.add(pyrender.Mesh.from_trimesh(m_copy, material=mat))

        # 6. Camera & Lights
        cam_pose, cam = self.get_camera_pose(zoom_progress)
        scene.add(cam, pose=cam_pose)

        light1 = pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.2)
        scene.add(light1, pose=cam_pose)
        light2 = pyrender.DirectionalLight(color=[0.90, 0.95, 1.0], intensity=1.8)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [-cam_pose[0, 3], -cam_pose[1, 3], cam_pose[2, 3] * 0.8]
        scene.add(light2, pose=light2_pose)

        renderer = pyrender.OffscreenRenderer(self.size, self.size)
        color, _ = renderer.render(scene)
        renderer.delete()

        return Image.fromarray(color)

    def _render_software(
        self,
        placed_indices: set[int],
        interpolating: dict[int, tuple[float, float, float]] | None = None,
        zoom_progress: float = 0.0,
        routed_fraction: float = 0.0,
    ) -> Image.Image:
        """High-performance Software 3D renderer with shading and depth sorting (zero external GL dependencies)."""
        img = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Camera math matching get_camera_pose
        ease = 0.5 - 0.5 * math.cos(math.pi * zoom_progress)
        dist_wide = max(self.bw, self.bh) * 2.75
        target_wide = np.array([14.0, -5.0, 0.0])
        dist_tight = max(self.bw, self.bh) * 2.05
        target_tight = np.array([0.0, 0.0, 0.0])

        cam_dist = dist_wide + (dist_tight - dist_wide) * ease
        target = target_wide + (target_tight - target_wide) * ease

        elev = np.radians(36)
        azim = np.radians(45)
        cam_x = target[0] + cam_dist * np.cos(elev) * np.sin(azim)
        cam_y = target[1] - cam_dist * np.cos(elev) * np.cos(azim)
        cam_z = target[2] + cam_dist * np.sin(elev)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up_cam = np.cross(right, forward)

        fov = np.radians(36)
        f_len = 0.5 * self.size / np.tan(fov / 2.0)
        cx_scr, cy_scr = self.size / 2.0, self.size / 2.0

        def project_pts(pts_world):
            rel_v = pts_world - cam_pos
            xc = rel_v @ right
            yc = rel_v @ up_cam
            zc = rel_v @ forward
            zc_clamped = np.maximum(zc, 1e-3)
            u = cx_scr + f_len * (xc / zc_clamped)
            v = cy_scr - f_len * (yc / zc_clamped)
            return np.column_stack([u, v]), zc

        L = np.array([0.4, -0.6, 0.8])
        L = L / np.linalg.norm(L)

        # 1. Contact shadow on floor at z = 0
        sh_w, sh_h = self.bw + 8.0, self.bh + 8.0
        sh_corners = np.array([
            [-sh_w / 2.0, -sh_h / 2.0, 0.0],
            [ sh_w / 2.0, -sh_h / 2.0, 0.0],
            [ sh_w / 2.0,  sh_h / 2.0, 0.0],
            [-sh_w / 2.0,  sh_h / 2.0, 0.0],
        ])
        poly_sh, _ = project_pts(sh_corners)
        draw.polygon([tuple(p) for p in poly_sh], fill=(230, 230, 234))

        # 2. PCB Substrate
        pcb_mesh = trimesh.creation.box(extents=[self.bw, self.bh, self.pcb_th])
        pcb_mesh.apply_translation([0, 0, self.pcb_th / 2.0])
        v_w = pcb_mesh.vertices
        uv, zc = project_pts(v_w)
        for f_i, face in enumerate(pcb_mesh.faces):
            norm = pcb_mesh.face_normals[f_i]
            v0 = v_w[face[0]]
            if np.dot(norm, cam_pos - v0) <= 0.0:
                continue
            dot = max(0.0, float(np.dot(norm, L)))
            shade = 0.45 + 0.55 * dot
            rgb = tuple(int(min(255, c * shade)) for c in self.color_map["substrate"])
            draw.polygon([tuple(uv[idx]) for idx in face], fill=rgb)

        # 3. Corner mounting holes
        for hx, hy in [
            (-self.bw / 2 + 3.5, -self.bh / 2 + 3.5),
            (self.bw / 2 - 3.5, -self.bh / 2 + 3.5),
            (-self.bw / 2 + 3.5, self.bh / 2 - 3.5),
            (self.bw / 2 - 3.5, self.bh / 2 - 3.5),
        ]:
            pts_hole = np.array([
                [hx + 1.8 * np.cos(t), hy + 1.8 * np.sin(t), self.pcb_th + 0.01]
                for t in np.linspace(0, 2 * np.pi, 16)
            ])
            proj_hole, _ = project_pts(pts_hole)
            draw.polygon([tuple(p) for p in proj_hole], fill=self.color_map["copper"])
            pts_inner = np.array([
                [hx + 1.0 * np.cos(t), hy + 1.0 * np.sin(t), self.pcb_th + 0.02]
                for t in np.linspace(0, 2 * np.pi, 16)
            ])
            proj_inner, _ = project_pts(pts_inner)
            draw.polygon([tuple(p) for p in proj_inner], fill=(20, 20, 20))

        # 4. Routed copper traces & vias in 3D
        if routed_fraction > 0.0:
            segments = self.design.get("segments", [])
            n_show_segs = int(len(segments) * min(routed_fraction, 1.0))
            for seg in segments[:n_show_segs]:
                if "F.Cu" in seg.get("layer", ""):
                    sx, sy = seg["start"]
                    ex, ey = seg["end"]
                    p1 = np.array([sx - self.center_x, -(sy - self.center_y), self.pcb_th + 0.015])
                    p2 = np.array([ex - self.center_x, -(ey - self.center_y), self.pcb_th + 0.015])
                    pts_proj, _ = project_pts(np.array([p1, p2]))
                    draw.line([tuple(pts_proj[0]), tuple(pts_proj[1])], fill=self.color_map["copper"], width=2)

            vias = self.design.get("vias", [])
            n_show_vias = int(len(vias) * min(routed_fraction, 1.0))
            for via in vias[:n_show_vias]:
                vx, vy = via["pos"]
                vp = np.array([vx - self.center_x, -(vy - self.center_y), self.pcb_th + 0.02])
                v_proj, _ = project_pts(np.array([vp]))
                px, py = v_proj[0]
                draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=self.color_map["copper"], outline=(40, 30, 15))

        # 5. Component meshes (sorted back-to-front by depth)
        comp_faces = []
        for i, comp in enumerate(self.design["components"]):
            is_placed = i in placed_indices
            is_interp = interpolating and i in interpolating

            if is_interp:
                cx, cy, cz = interpolating[i]
            elif is_placed:
                pos = self.design["placement"]["positions"][i]
                if pos is not None:
                    cx, cy = pos["position"]
                    cz = self.pcb_th
                else:
                    cx, cy = self.unplaced_slots[i]
                    cz = 0.0
            else:
                cx, cy = self.unplaced_slots[i]
                cz = 0.0

            rel_x = cx - self.center_x
            rel_y = -(cy - self.center_y)

            parts = self.comp_models[i]
            for mesh_def, mat_spec in parts:
                if isinstance(mat_spec, str):
                    base_rgb = self.color_map.get(mat_spec, (180, 180, 180))
                else:
                    c_fac = getattr(mat_spec, "baseColorFactor", [0.7, 0.7, 0.7])
                    base_rgb = tuple(int(min(255, c * 255)) for c in c_fac[:3])

                v_w = mesh_def.vertices + [rel_x, rel_y, cz]
                uv, zc = project_pts(v_w)
                normals = mesh_def.face_normals

                for f_i, face in enumerate(mesh_def.faces):
                    norm = normals[f_i]
                    v0 = v_w[face[0]]
                    if np.dot(norm, cam_pos - v0) <= 0.0:
                        continue
                    depth = float(zc[face].mean())
                    dot = max(0.0, float(np.dot(norm, L)))
                    shade = 0.45 + 0.55 * dot
                    rgb = tuple(int(min(255, c * shade)) for c in base_rgb)
                    poly = [tuple(uv[idx]) for idx in face]
                    comp_faces.append((depth, poly, rgb))

        comp_faces.sort(key=lambda x: x[0], reverse=True)
        for _, poly, rgb in comp_faces:
            draw.polygon(poly, fill=rgb)

        return img

    def render_frame(
        self,
        placed_indices: set[int],
        interpolating: dict[int, tuple[float, float, float]] | None = None,
        zoom_progress: float = 0.0,
        routed_fraction: float = 0.0,
    ) -> Image.Image:
        """Render a single 3D view frame, using hardware OpenGL or falling back to software 3D."""
        if self.use_pyrender:
            try:
                return self._render_pyrender(placed_indices, interpolating, zoom_progress, routed_fraction)
            except Exception as e:
                log.warning("pyrender frame failed (%s) — switching to software 3D renderer", e)
                self.use_pyrender = False
                return self._render_software(placed_indices, interpolating, zoom_progress, routed_fraction)
        return self._render_software(placed_indices, interpolating, zoom_progress, routed_fraction)


# ---------------------------------------------------------------------------
# Full Animation Orchestrator
# ---------------------------------------------------------------------------

def render_rollout_video(
    board_path: str | Path,
    out_dir: str | Path,
    fps: int = 30,
    duration: float = 15.0,
) -> Path:
    """Generate the full split-screen placement & routing animation."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "splitscreen_rollout.mp4"

    log.info("Loading board data from: %s", board_path)
    design = load_board_data(board_path)
    bounds = design["board"]["bounds"]
    n_comp = len(design["components"])
    log.info("  %d components | bounds: %.1f × %.1f mm", n_comp, bounds["w"], bounds["h"])

    unplaced_slots = compute_unplaced_slots(design["components"], bounds)

    log.info("Initializing 2D and 3D renderers (640×640 each) …")
    renderer_2d = Renderer2D(design, unplaced_slots, size=640)
    renderer_3d = Renderer3D(design, unplaced_slots, size=640)

    # Total frames: 15.0s at 30 fps = 450 frames
    total_frames = int(fps * duration)
    # Timeline division:
    # 0.0s to 0.5s (0 to 15 frames): Initial unplaced state
    # 0.5s to 6.5s (15 to 195 frames = 180 frames): Placement rollout (N components)
    # 6.5s to 7.5s (195 to 225 frames = 30 frames): Dynamic zoom transition to board
    # 7.5s to 13.5s (225 to 405 frames = 180 frames): Routing phase
    # 13.5s to 15.0s (405 to 450 frames = 45 frames): Final hold
    f_init = int(0.5 * fps)
    f_place_end = int(6.5 * fps)
    f_zoom_end = int(7.5 * fps)
    f_route_end = int(13.5 * fps)

    placement_order = design.get("assembly_order", list(range(n_comp)))
    frames_per_comp = (f_place_end - f_init) / max(n_comp, 1)

    log.info("Rendering %d animation frames …", total_frames)

    for f_idx in range(total_frames):
        # 1. Placement state
        if f_idx < f_init:
            placed_indices = set()
            interp_2d = None
            interp_3d = None
            zoom_prog = 0.0
            routed_frac = 0.0
        elif f_idx < f_place_end:
            # During placement rollout
            step_float = (f_idx - f_init) / frames_per_comp
            current_comp_idx = min(int(step_float), n_comp - 1)
            placed_indices = set(placement_order[:current_comp_idx])

            # Interpolate the currently moving component
            frac = step_float - int(step_float)
            target_idx = placement_order[current_comp_idx]
            p_start = unplaced_slots[target_idx]
            pos_dict = design["placement"]["positions"][target_idx]
            p_end = tuple(pos_dict["position"]) if pos_dict else p_start

            # Smooth ease-in-out interpolation
            ease = 0.5 - 0.5 * math.cos(math.pi * frac)
            curr_x = p_start[0] + (p_end[0] - p_start[0]) * ease
            curr_y = p_start[1] + (p_end[1] - p_start[1]) * ease
            # 3D arc flight: lifts up ~7mm into the air before touching down
            arc_z = 0.0 + (1.6 - 0.0) * ease + 7.0 * math.sin(math.pi * ease)

            interp_2d = {target_idx: (curr_x, curr_y)}
            interp_3d = {target_idx: (curr_x, curr_y, arc_z)}
            zoom_prog = 0.0
            routed_frac = 0.0
        else:
            # All components placed
            placed_indices = set(placement_order)
            interp_2d = None
            interp_3d = None

            # Zoom progress (6.5s to 7.5s)
            if f_idx < f_zoom_end:
                zoom_prog = (f_idx - f_place_end) / (f_zoom_end - f_place_end)
            else:
                zoom_prog = 1.0

            # Routing fraction (7.5s to 13.5s)
            if f_idx < f_zoom_end:
                routed_frac = 0.0
            elif f_idx < f_route_end:
                routed_frac = (f_idx - f_zoom_end) / (f_route_end - f_zoom_end)
            else:
                routed_frac = 1.0

        # Render 2D panel
        img_2d = renderer_2d.render_frame(
            placed_indices,
            interpolating=interp_2d,
            zoom_progress=zoom_prog,
            routed_fraction=routed_frac,
        )

        # Render 3D panel
        img_3d = renderer_3d.render_frame(
            placed_indices,
            interpolating=interp_3d,
            zoom_progress=zoom_prog,
            routed_fraction=routed_frac,
        )

        # Compose side-by-side (1280x640)
        combined = Image.new("RGB", (1280, 640), (255, 255, 255))
        combined.paste(img_2d, (0, 0))
        combined.paste(img_3d, (640, 0))

        frame_file = frames_dir / f"frame_{f_idx:04d}.png"
        combined.save(frame_file)

        if (f_idx + 1) % 45 == 0:
            log.info("  %d / %d frames (%.1fs)", f_idx + 1, total_frames, (f_idx + 1) / fps)

    log.info("Stitching video via ffmpeg …")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found in PATH!")
        return out_mp4

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    log.info("Successfully produced rollout animation video: %s (size: %d KB)", out_mp4, out_mp4.stat().st_size // 1024)
    return out_mp4


def main():
    parser = argparse.ArgumentParser(description="Render SCPT-RL Dynamic Placement & Routing Rollout Animation")
    parser.add_argument("--board", default="dataset/base_raw/Driverino-Shield.kicad_pcb", help="Input .kicad_pcb file")
    parser.add_argument("--out-dir", default="renders/driverino_rollout", help="Output directory for frames and video")
    parser.add_argument("--fps", type=int, default=30, help="Framerate (default 30)")
    parser.add_argument("--duration", type=float, default=15.0, help="Total duration in seconds (default 15.0)")
    args = parser.parse_args()

    render_rollout_video(args.board, args.out_dir, args.fps, args.duration)


if __name__ == "__main__":
    main()
