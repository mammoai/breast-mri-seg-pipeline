"""
01_compute_task_parameters.py
=============================

Purpose: load a single pre- and post-contrast DCE series from
tests/data, run the same pipeline, and persist the extracted parameters as a
JSON file that the next script can consume.

Output
------
label-studio/label-studio/images/task_params.json   bounding boxes, windows, chest-wall position
label-studio/label-studio/images/task_file.txt      LS-style task descriptor (JSON)
label-studio/label-studio/images/input_mips/001.png destination path for the composite MIP (written by script 02)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mri import MRISeries, pipeline


REPO_ROOT = Path(__file__).parent.parent
# Task data lives in label-studio/label-studio/data/<task_xxx>/
DATA_ROOT      = REPO_ROOT / "label-studio" / "label-studio" / "data"
OUTPUT_DIR     = REPO_ROOT / "label-studio" / "label-studio" / "images"
INPUT_MIPS_DIR = OUTPUT_DIR / "input_mips"   # composite MIP lives here

# Fixed task metadata (demographic / acquisition info)
PATIENT_ID   = 1
STUDY_ID     = 1
LATERALITY   = "B"     # B = bilateral composite
STUDY_DATE   = "2026-03-05"
SERVER_URL   = os.environ.get("SERVER_URL", "http://localhost:8083")
# Path under LOCAL_FILES_DOCUMENT_ROOT used by Label Studio local-files endpoint.
LOCAL_FILES_RELATIVE_PATH = os.environ.get(
    "LOCAL_FILES_RELATIVE_PATH", "label-studio/label-studio/images/input_mips"
)


# Helpers for task / series discovery
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


def load_series(pre_path: Path, post_path: Path):
    """Load the pre- and post-contrast DICOM series as MRISeries objects."""
    print(f"Loading pre-contrast series from:  {pre_path}")
    mri_pre = MRISeries.from_path(pre_path)
    print(f"  dims={mri_pre.dims}, spacing={tuple(round(s,3) for s in mri_pre.spacing)}")

    print(f"Loading post-contrast series from: {post_path}")
    mri_post = MRISeries.from_path(post_path)
    print(f"  dims={mri_post.dims}, spacing={tuple(round(s,3) for s in mri_post.spacing)}")

    return mri_pre, mri_post


def compute_and_save_params(
    mri_pre: MRISeries,
    mri_post: MRISeries,
    output_dir: Path,
    series_pre_path: Path,
    series_post_path: Path,
    mip_filename: str,
):
    """
    Run the processing pipeline and persist the task parameters to JSON.

    The pipeline (utils.mri.pipeline) computes:
      - chest_start      : axial index where the chest wall begins
      - window1/window2  : voxel-intensity display windows for pre/post series
      - sub_window       : window for the subtraction series
      - bounding_rectangles: 2-D ROIs around each breast in the axial MIP
      - bounding_boxes   : 3-D boxes (start_voxel, size_voxel) for each breast
    """
    print("\nRunning processing pipeline …")
    results = pipeline(mri_pre, mri_post)

    # Serialise to JSON (tuples → lists so json.dump can handle them)
    # ------------------------------------------------------------------
    def to_serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return [to_serialisable(x) for x in obj]
        if isinstance(obj, list):
            return [to_serialisable(x) for x in obj]
        return obj

    # Compute resampled shape (isotropic z-spacing) from pre-contrast series
    resampled_size, _ = mri_pre.get_z_resampling_size_and_spacing()

    params = {
        "chest_start":         to_serialisable(results["chest_start"]),
        "window1":             to_serialisable(results["window1"]),
        "window2":             to_serialisable(results["window2"]),
        "sub_window":          to_serialisable(results["sub_window"]),
        "bounding_rectangles": to_serialisable(results["bounding_rectangles"]),
        "bounding_boxes":      to_serialisable(results["bounding_boxes"]),
        # resampled_shape: SimpleITK (x, y, z) order – used by script 03
        "resampled_shape":     to_serialisable(resampled_size),
        # original dims: (x, y, z) in SimpleITK order
        "original_dims":       list(mri_pre.dims),
        "series_pre":          str(series_pre_path),
        "series_post":         str(series_post_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "task_params.json"
    with open(out_file, "w") as fh:
        json.dump(params, fh, indent=2)

    print(f"\nTask parameters saved → {out_file}")
    print(f"  chest_start          : {params['chest_start']}")
    print(f"  window1              : {params['window1']}")
    print(f"  sub_window           : {params['sub_window']}")
    print(f"  bounding_rectangles  : {params['bounding_rectangles']}")
    print(f"  bounding_boxes       : {params['bounding_boxes']}")

    # Write task_file.txt  – mirrors the LS task descriptor format
    image_url = f"/data/local-files/?d={LOCAL_FILES_RELATIVE_PATH}/{mip_filename}"
    task_data = {
        "data": {
            "patient_id":       PATIENT_ID,
            "study_id":         STUDY_ID,
            "study_date":       STUDY_DATE,
            "laterality":       LATERALITY,
            "server_url":       SERVER_URL,
            "image_url":        image_url,
            "assessment_text":  "",
        }
    }
    task_file = output_dir / "task_file.txt"
    with open(task_file, "w") as fh:
        json.dump(task_data, fh, indent=4)

    print(f"\nTask file saved       → {task_file}")
    print(f"  image_url : {image_url}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script 01: compute task parameters.")
    parser.add_argument(
        "--task", default=None,
        help="Task folder name inside label-studio/label-studio/data/ (e.g. task_001). "
             "Auto-discovers the first task if omitted.",
    )
    args = parser.parse_args()

    task_dir                 = find_task_dir(args.task)
    series_pre, series_post  = find_series_dirs(task_dir)
    mip_filename             = task_mip_filename(task_dir)

    print(f"Task directory  : {task_dir}")
    print(f"Pre series      : {series_pre.name}")
    print(f"Post series     : {series_post.name}")
    print(f"MIP filename    : {mip_filename}")

    mri_pre, mri_post = load_series(series_pre, series_post)
    compute_and_save_params(mri_pre, mri_post, OUTPUT_DIR, series_pre, series_post, mip_filename)
    print("\nScript 01 complete. Run 02_generate_mips.py next.")
