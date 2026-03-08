"""
debug_morphology.py
====================
Visual debugger for all morphological features.

Run from the initial_pipeline/ directory, same as main.py:

    python debug_morphology.py --input_folder_path <path> --max_cells 5

Layout
------
  +-----------------------------+------------------+
  |                             |  Panel title     |
  |   Real image crop with      |                  |
  |   overlays (correct aspect) |  Feature values  |
  |                             |  & legend        |
  |                             |                  |
  +-----------------------------+------------------+
  |  Cell N / panel M   any key=next  q=skip  Esc  |
  +--------------------------------------------------+

Controls
--------
  Any key  – next panel
  q        – skip remaining panels for this cell, go to next cell
  Escape   – quit entirely
"""

import sys
import math
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve, label as scipy_label

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import get_file_name
from object_detection.yolo_pretrained.yolo_inference import yolo_inference
from segmentation.soma_segmentation.gaussian_filter import get_gaussian_filter_soma_masks
from morphology.morphology_features import (
    get_skeletons,
    compute_mask_area,
    compute_mask_perimeter,
    compute_convex_hull,
    compute_convex_hull_area,
    compute_convex_hull_perimeter,
    compute_branch_mask,
    compute_sholl_analysis,
)
from segmentation.custom_segmentation.training.unet import UNet
from segmentation.custom_segmentation.segmentation_inference import unet_inference
from segmentation.custom_segmentation.crf import connect_all_masks

import torch
import argparse
from ultralytics import YOLO


# =========================================================================== #
#  Colours  (BGR)
# =========================================================================== #
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (0,   220, 0)
C_RED     = (0,   0,   220)
C_CYAN    = (220, 220, 0)
C_YELLOW  = (0,   220, 220)
C_MAGENTA = (220, 0,   220)
C_ORANGE  = (0,   140, 255)
C_BLUE    = (200, 80,  0)
C_PINK    = (180, 105, 255)
C_SIDEBAR = (30,  30,  30)

# Sidebar width in pixels (absolute; will hold the text)
SIDEBAR_W  = 420
# Footer bar height
FOOTER_H   = 30


# =========================================================================== #
#  Low-level drawing helpers
# =========================================================================== #

def _crop_real(image_bgr, box, pad=20):
    """Return the real image crop and its top-left origin (ox, oy)."""
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1)


def _tint(canvas, local_mask, color_bgr, alpha=0.45):
    """Alpha-blend colour onto canvas wherever local_mask is True."""
    overlay = canvas.copy()
    overlay[local_mask] = color_bgr
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def _skeleton_pixels(canvas, skeleton, origin, color=C_GREEN):
    ox, oy = origin
    ys, xs = np.where(skeleton)
    for y, x in zip(ys, xs):
        cy, cx = y - oy, x - ox
        if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1]:
            canvas[cy, cx] = color


def _dots(canvas, mask_2d, origin, color, radius=1):
    ox, oy = origin
    ys, xs = np.where(mask_2d)
    for y, x in zip(ys, xs):
        cy, cx = y - oy, x - ox
        if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1]:
            cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _contour(canvas, binary_mask, origin, color, thickness=1):
    ox, oy = origin
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted = [cnt - np.array([[[ox, oy]]]) for cnt in cnts]
    cv2.drawContours(canvas, shifted, -1, color, thickness, cv2.LINE_AA)


def _hull_contour(canvas, hull, origin, color, thickness=1):
    if hull is None:
        return
    ox, oy = origin
    shifted = hull - np.array([[[ox, oy]]])
    cv2.drawContours(canvas, [shifted], -1, color, thickness, cv2.LINE_AA)


def _circle(canvas, cy_global, cx_global, origin, radius, color, thickness=1):
    ox, oy = origin
    cv2.circle(canvas,
               (int(cx_global - ox), int(cy_global - oy)),
               int(radius), color, thickness, cv2.LINE_AA)


def _to_local(global_mask, origin, canvas_shape):
    """Slice a global boolean mask into canvas coordinates."""
    ox, oy = origin
    H, W = canvas_shape[:2]
    H_g, W_g = global_mask.shape
    out = np.zeros((H, W), bool)
    y1g, x1g = oy, ox
    y2g, x2g = oy + H, ox + W
    y1g = max(0, y1g); x1g = max(0, x1g)
    y2g = min(H_g, y2g); x2g = min(W_g, x2g)
    y1c, x1c = y1g - oy, x1g - ox
    y2c, x2c = y2g - oy, x2g - ox
    out[y1c:y2c, x1c:x2c] = global_mask[y1g:y2g, x1g:x2g]
    return out


# =========================================================================== #
#  Sidebar text renderer
# =========================================================================== #

def _make_sidebar(height, title, lines, legend=None):
    """
    Build a dark sidebar image of size (height, SIDEBAR_W).

    title  – bold heading
    lines  – list of plain strings (feature values)
    legend – list of (color_bgr, label) tuples drawn as coloured squares
    """
    sidebar = np.full((height, SIDEBAR_W, 3), C_SIDEBAR, dtype=np.uint8)

    y = 28
    # Title
    cv2.putText(sidebar, title, (14, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)
    y += 10
    cv2.line(sidebar, (14, y), (SIDEBAR_W - 14, y), (80, 80, 80), 1)
    y += 20

    # Feature values
    for line in lines:
        cv2.putText(sidebar, line, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1, cv2.LINE_AA)
        y += 22

    # Legend
    if legend:
        y += 10
        cv2.line(sidebar, (14, y), (SIDEBAR_W - 14, y), (80, 80, 80), 1)
        y += 18
        for color, label in legend:
            cv2.rectangle(sidebar, (14, y - 11), (28, y + 1), color, -1)
            cv2.putText(sidebar, label, (36, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            y += 20

    return sidebar


# =========================================================================== #
#  Main display function
# =========================================================================== #

_WINDOW_NAME = "Morphology Debugger"
_window_open = False


def _show(img_canvas, title, lines, legend, cell_idx, panel_idx, n_panels):
    """
    Compose image + sidebar + footer and display fullscreen.
    Returns the key code pressed.
    """
    global _window_open

    img_h, img_w = img_canvas.shape[:2]
    sidebar = _make_sidebar(img_h, title, lines, legend)

    # join image and sidebar side-by-side
    body = np.hstack([img_canvas, sidebar])

    # footer bar
    footer = np.zeros((FOOTER_H, body.shape[1], 3), dtype=np.uint8)
    footer_text = (f"Cell {cell_idx}  |  Panel {panel_idx}/{n_panels}: {title}"
                   "    [any key = next    q = skip cell    Esc = quit]")
    cv2.putText(footer, footer_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

    display = np.vstack([body, footer])

    if not _window_open:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        _window_open = True

    cv2.imshow(_WINDOW_NAME, display)
    return cv2.waitKey(0) & 0xFF


# =========================================================================== #
#  Panel builders
#  Each returns (img_canvas, title, lines, legend)
#  img_canvas – BGR crop of the real image with overlays
# =========================================================================== #

def panel_orientation(image_bgr, box, mask, skeleton, soma_mask):
    """Overview: real image + cell contour + soma contour + skeleton."""
    canvas, origin = _crop_real(image_bgr, box)
    _contour(canvas, mask,      origin, C_GREEN,  thickness=1)
    _contour(canvas, soma_mask, origin, C_CYAN,   thickness=1)
    _skeleton_pixels(canvas, skeleton, origin, C_YELLOW)
    title = "Orientation"
    lines = ["Real image crop with:", ""]
    legend = [
        (C_GREEN,  "cell boundary"),
        (C_CYAN,   "soma boundary"),
        (C_YELLOW, "skeleton"),
    ]
    return canvas, title, lines, legend


def panel_skeleton_length(image_bgr, box, mask, skeleton, soma_mask):
    canvas, origin = _crop_real(image_bgr, box)
    _contour(canvas, mask, origin, C_WHITE, thickness=1)
    _skeleton_pixels(canvas, skeleton, origin, C_GREEN)
    length = int(skeleton.sum())
    title = "Skeleton length"
    lines = [f"skeleton_length = {length} px"]
    legend = [(C_GREEN, "skeleton pixels")]
    return canvas, title, lines, legend


def panel_junctions(image_bgr, box, mask, skeleton, soma_mask):
    kernel = np.ones((3, 3), np.uint8); kernel[1, 1] = 0
    nbrs = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    junction_mask = skeleton & (nbrs >= 3)

    # Count connected clusters — same logic as compute_junction_count
    structure = np.ones((3, 3), np.uint8)
    labeled, num_junctions = scipy_label(junction_mask, structure=structure)

    canvas, origin = _crop_real(image_bgr, box)
    _skeleton_pixels(canvas, skeleton, origin, C_GREEN)
    # Draw each junction cluster as a distinct dot at its centroid
    for j in range(1, num_junctions + 1):
        ys, xs = np.where(labeled == j)
        cy_g, cx_g = int(ys.mean()), int(xs.mean())
        ox, oy = origin
        cv2.circle(canvas, (cx_g - ox, cy_g - oy), 2, C_RED, -1, cv2.LINE_AA)

    title = "Junctions"
    lines = [
        f"num_junctions = {num_junctions}",
        "",
        "(adjacent junction pixels",
        " merged into one cluster)",
    ]
    legend = [
        (C_GREEN, "skeleton"),
        (C_RED,   "junction centroid (1 dot = 1 junction)"),
    ]
    return canvas, title, lines, legend


def panel_nodes(image_bgr, box, mask, skeleton, soma_mask):
    kernel = np.ones((3, 3), np.uint8); kernel[1, 1] = 0
    nbrs = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    tip_mask   = skeleton & (nbrs == 1)
    soma_bin   = soma_mask > 0
    soma_dil   = convolve(soma_bin.astype(np.uint8), kernel, mode='constant', cval=0) > 0
    end_mask   = tip_mask & (~soma_dil)
    start_mask = tip_mask & soma_dil

    n_end   = int(end_mask.sum())
    n_start = int(start_mask.sum())
    ratio   = n_end / n_start if n_start > 0 else 0.0

    canvas, origin = _crop_real(image_bgr, box)
    _tint(canvas, _to_local(soma_bin, origin, canvas.shape), C_BLUE, alpha=0.25)
    _skeleton_pixels(canvas, skeleton, origin, C_GREEN)
    _dots(canvas, end_mask,   origin, C_CYAN,   radius=1)
    _dots(canvas, start_mask, origin, C_YELLOW, radius=1)

    title = "End / Start nodes"
    lines = [
        f"num_end_nodes      = {n_end}",
        f"num_start_nodes    = {n_start}",
        f"total_nodes        = {n_end + n_start}",
        f"end_to_start_ratio = {ratio:.3f}",
    ]
    legend = [
        (C_GREEN,  "skeleton"),
        (C_BLUE,   "soma (tint)"),
        (C_CYAN,   "end nodes (free tips)"),
        (C_YELLOW, "start nodes (soma attach.)"),
    ]
    return canvas, title, lines, legend


def panel_components(image_bgr, box, mask, skeleton, soma_mask):
    structure = np.ones((3, 3), np.uint8)
    labeled, n = scipy_label(skeleton, structure=structure)

    cmap = [C_GREEN, C_RED, C_CYAN, C_YELLOW, C_MAGENTA,
            C_ORANGE, C_PINK, C_WHITE, C_BLUE]

    canvas, origin = _crop_real(image_bgr, box)
    for comp_id in range(1, n + 1):
        color = cmap[(comp_id - 1) % len(cmap)]
        _skeleton_pixels(canvas, labeled == comp_id, origin, color)

    title = "Skeleton components"
    lines = [f"num_components = {n}"]
    legend = [(c, f"component {i+1}") for i, c in enumerate(cmap[:min(n, len(cmap))])]
    return canvas, title, lines, legend


def panel_soma(image_bgr, box, mask, skeleton, soma_mask):
    soma_bin = soma_mask > 0
    area  = int(soma_bin.sum())
    perim = compute_mask_perimeter(soma_mask)
    circ  = (4 * math.pi * area) / (perim ** 2) if perim > 0 else 0.0

    canvas, origin = _crop_real(image_bgr, box)
    _tint(canvas, _to_local(soma_bin, origin, canvas.shape), C_BLUE, alpha=0.4)
    _contour(canvas, soma_mask, origin, C_CYAN, thickness=1)

    title = "Soma shape"
    lines = [
        f"soma_area        = {area} px",
        f"soma_perimeter   = {perim:.1f} px",
        f"soma_circularity = {circ:.4f}",
        "",
        "(1.0 = perfect circle)",
    ]
    legend = [
        (C_BLUE, "soma fill (tint)"),
        (C_CYAN, "soma contour"),
    ]
    return canvas, title, lines, legend


def panel_cell_shape(image_bgr, box, mask, skeleton, soma_mask):
    area     = compute_mask_area(mask)
    perim    = compute_mask_perimeter(mask)
    ch_area  = compute_convex_hull_area(mask)
    ch_perim = compute_convex_hull_perimeter(mask)
    hull     = compute_convex_hull(mask)

    solidity    = area / ch_area              if ch_area  > 0 else 0.0
    convexity   = ch_perim / perim            if perim    > 0 else 0.0
    circularity = (4 * math.pi * area)  / (perim    ** 2) if perim    > 0 else 0.0
    cx_circ     = (4 * math.pi * ch_area) / (ch_perim ** 2) if ch_perim > 0 else 0.0

    canvas, origin = _crop_real(image_bgr, box)
    _tint(canvas, _to_local(mask > 0, origin, canvas.shape), C_GREEN, alpha=0.15)
    _contour(canvas, mask, origin, C_GREEN, thickness=1)
    _hull_contour(canvas, hull, origin, C_YELLOW, thickness=1)

    title = "Cell shape & convex hull"
    lines = [
        f"cell_area              = {area} px",
        f"cell_perimeter         = {perim:.1f} px",
        f"cell_convex_hull_area  = {ch_area:.0f} px2",
        f"cell_convex_hull_perim = {ch_perim:.1f} px",
        "",
        f"cell_solidity          = {solidity:.4f}",
        f"cell_convexity         = {convexity:.4f}",
        f"cell_circularity       = {circularity:.4f}",
        f"cell_convex_circ       = {cx_circ:.4f}",
    ]
    legend = [
        (C_GREEN,  "cell contour / fill"),
        (C_YELLOW, "convex hull"),
    ]
    return canvas, title, lines, legend


def panel_branch(image_bgr, box, mask, skeleton, soma_mask):
    branch_mask = compute_branch_mask(mask, soma_mask)
    soma_bin    = soma_mask > 0
    b_area  = int(branch_mask.sum())
    b_perim = compute_mask_perimeter(branch_mask)

    canvas, origin = _crop_real(image_bgr, box)
    _tint(canvas, _to_local(soma_bin,    origin, canvas.shape), C_BLUE,  alpha=0.4)
    _tint(canvas, _to_local(branch_mask, origin, canvas.shape), C_GREEN, alpha=0.3)
    _contour(canvas, branch_mask, origin, C_GREEN, thickness=1)
    _contour(canvas, soma_mask,   origin, C_CYAN,  thickness=1)

    title = "Branch area / perimeter"
    lines = [
        f"branch_area      = {b_area} px",
        f"branch_perimeter = {b_perim:.1f} px",
    ]
    legend = [
        (C_BLUE,  "soma (tint)"),
        (C_GREEN, "branch region (tint + contour)"),
        (C_CYAN,  "soma contour"),
    ]
    return canvas, title, lines, legend


def panel_sholl(image_bgr, box, mask, skeleton, soma_mask):
    sholl = compute_sholl_analysis(skeleton, soma_mask)
    s_min = sholl["sholl_min_radius"]
    s_pr  = sholl["sholl_peak_radius"]
    s_max = sholl["sholl_max_radius"]
    s_pk  = sholl["sholl_peak"]
    s_sum = sholl["sholl_sum"]

    soma_bin = soma_mask > 0
    soma_ys, soma_xs = np.where(soma_bin)
    if len(soma_ys) == 0:
        skel_ys, skel_xs = np.where(skeleton)
        cy = float(skel_ys.mean()) if len(skel_ys) > 0 else 0.0
        cx = float(skel_xs.mean()) if len(skel_xs) > 0 else 0.0
    else:
        cy, cx = float(soma_ys.mean()), float(soma_xs.mean())

    skel_ys, skel_xs = np.where(skeleton)
    distances = (np.sqrt((skel_ys - cy) ** 2 + (skel_xs - cx) ** 2)
                 if len(skel_ys) > 0 else np.array([]))

    canvas, origin = _crop_real(image_bgr, box)
    _tint(canvas, _to_local(soma_bin, origin, canvas.shape), C_BLUE, alpha=0.3)
    _skeleton_pixels(canvas, skeleton, origin, C_GREEN)

    if s_max > 0:
        for r in range(1, s_max + 1):
            if r == s_pr:
                col, thick = C_YELLOW, 2
            elif r == s_min:
                col, thick = C_CYAN, 1
            elif r == s_max:
                col, thick = C_ORANGE, 1
            else:
                col, thick = (55, 55, 55), 1
            _circle(canvas, cy, cx, origin, r, col, thick)

    if len(skel_ys) > 0 and s_pr > 0:
        annulus   = (distances >= s_pr - 0.5) & (distances < s_pr + 0.5)
        peak_skel = np.zeros_like(skeleton)
        peak_skel[skel_ys[annulus], skel_xs[annulus]] = True
        _dots(canvas, peak_skel, origin, C_YELLOW, radius=1)

    title = "Sholl analysis"
    lines = [
        f"sholl_min_radius  = {s_min} px",
        f"sholl_peak_radius = {s_pr} px",
        f"sholl_max_radius  = {s_max} px",
        f"sholl_peak        = {s_pk} intersections",
        f"sholl_sum         = {s_sum} total",
    ]
    legend = [
        (C_GREEN,  "skeleton"),
        (C_BLUE,   "soma (tint)"),
        (C_CYAN,   "min radius ring"),
        (C_YELLOW, "peak radius ring + pixels"),
        (C_ORANGE, "max radius ring"),
        ((55,55,55), "intermediate rings"),
    ]
    return canvas, title, lines, legend


# =========================================================================== #
#  Panel registry
# =========================================================================== #
PANELS = [
    panel_orientation,
    panel_skeleton_length,
    panel_junctions,
    panel_nodes,
    panel_components,
    panel_soma,
    panel_cell_shape,
    panel_branch,
    panel_sholl,
]


# =========================================================================== #
#  Main
# =========================================================================== #

def parse_debug_args():
    parser = argparse.ArgumentParser(description="Morphology visual debugger")
    parser.add_argument("--input_folder_path", type=str, default="input/")
    parser.add_argument("--output_name", type=str, default="debug")
    parser.add_argument("--max_cells", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_debug_args()

    print("Loading models...")
    yolo = YOLO(r"object_detection/custom_detection/yolo_good_runs/28_2_rat_pretraining.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet()
    ckpt_path = "segmentation/custom_segmentation/checkpoints/best_run_25_1.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_dir  = Path(args.input_folder_path)
    cell_count = 0
    quit_all   = False
    n_panels   = len(PANELS)

    for scan_folder in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if quit_all:
            break
        scan_folder_name = scan_folder.name
        print(f"\nScan folder: {scan_folder_name}")

        for image_path in sorted(scan_folder.iterdir()):
            if quit_all or cell_count >= args.max_cells:
                break
            if not image_path.is_file():
                continue

            image_path_str = str(image_path)
            print(f"  Image: {image_path.name}")

            image     = cv2.imread(image_path_str)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            yolo_boxes = yolo_inference(
                yolo, image_path_str,
                output_name=args.output_name, output_to_file=False,
                scan_folder=scan_folder_name
            )
            if len(yolo_boxes) == 0:
                print("    No detections, skipping.")
                continue

            segmentation_masks = unet_inference(
                model, yolo_boxes,
                image_path=image_path_str, image_rgb=image_rgb,
                device=device, output_to_file=False,
                output_name=args.output_name, expand_boxes=False,
                scan_folder=scan_folder_name
            )

            connected_masks = connect_all_masks(
                segmentation_masks, image, image_path_str,
                output_to_file=False,
                output_name=args.output_name,
                scan_folder=scan_folder_name
            )

            soma_masks = get_gaussian_filter_soma_masks(
                yolo_boxes, image_path_str, image_rgb,
                output_name=args.output_name, output_to_file=False,
                scan_folder=scan_folder_name
            )

            skeletons = get_skeletons(
                image_rgb, image_path_str,
                connected_masks, soma_masks,
                output_to_file=False,
                output_name=args.output_name,
                scan_folder=scan_folder_name
            )

            for i, (mask, skeleton, soma_mask, box) in enumerate(
                zip(connected_masks, skeletons, soma_masks, yolo_boxes)
            ):
                if quit_all or cell_count >= args.max_cells:
                    break

                cell_count += 1
                print(f"    Cell {cell_count} (detection {i})")

                skip_cell = False
                for panel_idx, panel_fn in enumerate(PANELS, start=1):
                    if skip_cell or quit_all:
                        break

                    try:
                        img_canvas, title, lines, legend = panel_fn(
                            image, box, mask, skeleton, soma_mask
                        )
                    except Exception as exc:
                        img_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
                        title  = "ERROR"
                        lines  = [f"{panel_fn.__name__}:", str(exc)[:70]]
                        legend = []

                    key = _show(img_canvas, title, lines, legend,
                                cell_count, panel_idx, n_panels)

                    if key == ord('q'):
                        skip_cell = True
                    elif key == 27:
                        quit_all = True

    cv2.destroyAllWindows()
    print(f"\nDone. Showed {cell_count} cell(s).")


if __name__ == "__main__":
    main()
