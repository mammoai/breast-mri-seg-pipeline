from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mri import MRISeries


REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "label-studio" / "label-studio" / "data"
OUTPUT_DIR = REPO_ROOT / "label-studio" / "label-studio" / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
FINAL_ROOT = OUTPUT_DIR / "final"


def discover_task_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Task data root not found: {data_root}")
    task_dirs = sorted(
        d for d in data_root.iterdir() if d.is_dir() and d.name.lower().startswith("task_")
    )
    if not task_dirs:
        task_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {data_root}")
    return task_dirs


def find_task_dir(task_name: str) -> Path:
    task_dir = DATA_ROOT / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    return task_dir


def find_series_dirs(task_dir: Path) -> tuple[Path, Path]:
    subdirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
    if len(subdirs) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 series dirs in {task_dir}, found {len(subdirs)}"
        )
    pre_dirs = [d for d in subdirs if "pre" in d.name.lower()]
    post_dirs = [d for d in subdirs if "pre" not in d.name.lower()]
    if not pre_dirs:
        raise FileNotFoundError(f"No pre-contrast series dir found in {task_dir}")
    if not post_dirs:
        raise FileNotFoundError(f"No post-contrast series dir found in {task_dir}")
    return pre_dirs[0], post_dirs[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect final NIfTI outputs task-wise.")
    parser.add_argument("--task", default=None, help="Task folder name under data/, e.g. task_001")
    parser.add_argument(
        "--study-id",
        default=None,
        help="Optional output folder name under images/final/. Only valid together with --task.",
    )
    args = parser.parse_args()

    if args.study_id and not args.task:
        raise ValueError("--study-id requires --task. Without --task, script 06 exports all tasks using their task folder names.")

    if args.task:
        task_dirs = [find_task_dir(args.task)]
    else:
        task_dirs = discover_task_dirs(DATA_ROOT)

    label_variants = [
        "*_coarse.nii.gz",
        "*_moderately_processed.nii.gz",
        "*_extensively_processed.nii.gz",
    ]

    exported_dirs: list[Path] = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        final_name = args.study_id if args.task and args.study_id else task_name
        final_dir = FINAL_ROOT / final_name
        final_labels_dir = final_dir / "labels"
        final_labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {task_name} -> {final_dir.relative_to(REPO_ROOT)}")

        series_pre, series_post = find_series_dirs(task_dir)

        print("Loading and saving pre-contrast ...")
        mri_pre = MRISeries.from_path(series_pre)
        pre_out = final_dir / "pre_contrast.nii.gz"
        sitk.WriteImage(mri_pre.itk_image, str(pre_out))
        print(f"  -> {pre_out.relative_to(REPO_ROOT)}")

        print("Loading and saving post-contrast ...")
        mri_post = MRISeries.from_path(series_post)
        post_out = final_dir / "post_contrast.nii.gz"
        sitk.WriteImage(mri_post.itk_image, str(post_out))
        print(f"  -> {post_out.relative_to(REPO_ROOT)}")

        print("Saving subtraction ...")
        task_labels_dir = LABELS_DIR / task_name
        sub_src = task_labels_dir / "subtraction_mri_org_dim.nii.gz"
        sub_out = final_dir / "subtraction.nii.gz"
        if sub_src.exists():
            shutil.copy2(sub_src, sub_out)
            print(f"  -> {sub_out.relative_to(REPO_ROOT)}")
        else:
            print("  Subtraction not found, recomputing from DICOM ...")
            sub_mri = mri_post - mri_pre
            sitk.WriteImage(sub_mri.itk_image, str(sub_out))
            print(f"  -> {sub_out.relative_to(REPO_ROOT)}")

        copied = []
        if not task_labels_dir.exists():
            print(f"  Warning: labels directory not found for {task_name}. Run scripts 04 and 05 first.")
        else:
            for pattern in label_variants:
                for src in sorted(task_labels_dir.glob(pattern)):
                    dst = final_labels_dir / src.name
                    shutil.copy2(src, dst)
                    print(f"  -> {dst.relative_to(REPO_ROOT)}")
                    copied.append(src.name)

        if not copied:
            print("  Warning: no label files copied for this task.")
        else:
            print(f"  Copied {len(copied)} label file(s).")

        exported_dirs.append(final_dir)

    print("\nScript 06 complete. Final case folders:")
    for final_dir in exported_dirs:
        print(f"  {final_dir.relative_to(REPO_ROOT)}")
        for path in sorted(final_dir.rglob("*.nii.gz")):
            print(f"    {path.relative_to(REPO_ROOT)}")
