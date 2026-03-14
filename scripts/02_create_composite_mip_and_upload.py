
from __future__ import annotations

import json
import os
# import shutil
# import subprocess
import sys
# import threading
# import time
# import webbrowser
# from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mri import MRISeries, pipeline


REPO_ROOT      = Path(__file__).parent.parent
# Task data lives in label-studio/label-studio/data/<task_xxx>/
DATA_ROOT      = REPO_ROOT / "label-studio" / "label-studio" / "data"
OUTPUT_DIR     = REPO_ROOT / "label-studio" / "label-studio" / "images"
INPUT_MIPS_DIR = OUTPUT_DIR / "input_mips"
PARAMS_FILE    = OUTPUT_DIR / "task_params.json"


# Helpers for task / series discovery (shared convention with script 01)

def find_task_dir(task_name: str | None = None) -> Path:
    """Return the task directory.  If *task_name* is given use it,
    otherwise auto-discover the first (alphabetically) task in DATA_ROOT."""
    if task_name:
        task_dir = DATA_ROOT / task_name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        return task_dir
    task_dirs = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {DATA_ROOT}")
    return task_dirs[0]


def task_mip_filename(task_dir: Path) -> str:
    """Derive the MIP PNG filename from the task folder name.
    Examples: task_001 → 001.png,  task_42 → 042.png"""
    suffix = task_dir.name.split("_")[-1]
    return f"{int(suffix):03d}.png"


def find_series_dirs(task_dir: Path) -> tuple[Path, Path]:
    """Locate the pre- and post-contrast DICOM series directories.

    Convention: the pre-contrast series directory contains 'pre' in its name
    (case-insensitive).  The first non-pre directory is treated as post-contrast.
    """
    subdirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
    if len(subdirs) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 series dirs in {task_dir}, found {len(subdirs)}"
        )
    pre_dirs  = [d for d in subdirs if "pre" in d.name.lower()]
    post_dirs = [d for d in subdirs if "pre" not in d.name.lower()]
    if not pre_dirs:
        raise FileNotFoundError(f"No pre-contrast series dir found in {task_dir}")
    if not post_dirs:
        raise FileNotFoundError(f"No post-contrast series dir found in {task_dir}")
    return pre_dirs[0], post_dirs[0]


# Step 1: build the composite 4-view canvas

def create_composite_image(
    mri_pre: MRISeries,
    mri_post: MRISeries,
    bounding_boxes,
    sub_window,
) -> np.ndarray:
    """
    Reproduce the create_task_image() logic from the original task.py.

    Returns the uint8 composite canvas (big_size+small_size, 2*big_size).
    """
    sub_mri = mri_post - mri_pre
    r_ax_itk, l_ax_itk, r_sag_itk, l_sag_itk = sub_mri.get_cropped_mips(
        bounding_boxes, sub_window
    )

    # ---- Extract 2-D arrays from the projected SimpleITK images ----
    # Axial MIP: projection collapsed along z (dim 2 in sitk → dim 0 in numpy)
    # sitk image shape after axial MIP: (x, y, 1)  →  numpy: (1, y, x)
    r_ax = sitk.GetArrayFromImage(r_ax_itk[:, :, 0])   # numpy (y, x)
    l_ax = sitk.GetArrayFromImage(l_ax_itk[:, :, 0])

    # Rotate outwards so the lesion faces inward (toward midline)
    r_ax = np.rot90(r_ax, k=1)    # 90° CW
    l_ax = np.rot90(l_ax, k=3)    # 90° CCW

    # Sagittal MIP: projection collapsed along x (dim 0 in sitk → dim 2 in numpy)
    # sitk image shape after sagittal MIP: (1, y, z)  →  numpy: (z, y, 1)
    r_sag = sitk.GetArrayFromImage(r_sag_itk[0, :, :])  # numpy (y, z)
    l_sag = sitk.GetArrayFromImage(l_sag_itk[0, :, :])

    r_sag = np.flip(r_sag, axis=0)           # superior = top
    l_sag = np.flip(l_sag, axis=(0, 1))      # flip both axes for left breast

    # ---- Canvas dimensions ----
    (_, r_size), (_, l_size) = bounding_boxes
    big_size   = r_size[1]   # y-dimension (breast height in image space)
    small_size = r_size[0]   # x-dimension (AP depth)

    # ---- Assemble canvas ----
    canvas = np.zeros((big_size + small_size, 2 * big_size), dtype=np.uint8)
    canvas[0:big_size,  0:big_size]  = r_sag
    canvas[0:big_size,  big_size:]   = l_sag
    canvas[big_size:,   0:big_size]  = r_ax
    canvas[big_size:,   big_size:]   = l_ax

    return canvas


# Step 2: start Label Studio  and upload the composite image as a task

#TODO



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script 02: create composite MIP.")
    parser.add_argument(
        "--task", default=None,
        help="Task folder name inside label-studio/label-studio/data/ (e.g. task_001). "
             "Auto-discovers the first task if omitted.",
    )
    args = parser.parse_args()

    task_dir                 = find_task_dir(args.task)
    series_pre, series_post  = find_series_dirs(task_dir)
    mip_filename             = task_mip_filename(task_dir)
    composite_png            = INPUT_MIPS_DIR / mip_filename

    print(f"Task directory  : {task_dir}")
    print(f"Pre series      : {series_pre.name}")
    print(f"Post series     : {series_post.name}")
    print(f"MIP filename    : {mip_filename}")

    # ---- Load params from script 01 ----
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Task params not found at {PARAMS_FILE}.\n"
            "Run 01_compute_task_parameters.py first."
        )
    with open(PARAMS_FILE) as fh:
        params = json.load(fh)

    bounding_boxes = [
        (tuple(params["bounding_boxes"][0][0]), tuple(params["bounding_boxes"][0][1])),
        (tuple(params["bounding_boxes"][1][0]), tuple(params["bounding_boxes"][1][1])),
    ]
    sub_window = tuple(params["sub_window"])

    # ---- Load DICOM series ----
    print("Loading DICOM series …")
    mri_pre  = MRISeries.from_path(series_pre)
    mri_post = MRISeries.from_path(series_post)

    # ---- Build composite canvas ----
    print("Building 4-view composite canvas …")
    canvas = create_composite_image(mri_pre, mri_post, bounding_boxes, sub_window)
    print(f"  Canvas shape : {canvas.shape}  (height × width)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_MIPS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(str(composite_png))
    print(f"  Saved → {composite_png.relative_to(REPO_ROOT)}")

    print(f"\nScript 02 complete.")
    print(f"  Composite image: {composite_png}")

