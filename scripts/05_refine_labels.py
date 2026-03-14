from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.filters import threshold_otsu
import skimage as ski


REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "label-studio" / "label-studio" / "images"
LABELS_DIR = OUTPUT_DIR / "labels"


def dilate_volume(volume: np.ndarray) -> np.ndarray:
    image = np.array(volume).astype("uint8")

    for z in range(image.shape[2]):
        sl = image[:, :, z]
        contours, _ = cv2.findContours(sl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            for c in contours:
                if cv2.contourArea(c) < 60:
                    xx = np.array([p[0][1] for p in c])
                    yy = np.array([p[0][0] for p in c])
                    xs, ys = ski.draw.polygon(xx, yy, shape=sl.shape)
                    sl[xs, ys] = 0
        image[:, :, z] = sl

    kernel = np.ones((9, 9, 9))
    closed = ndimage.binary_closing(image, structure=kernel, iterations=1)
    return closed.astype(bool)


def label_refine(coarse_vol: np.ndarray, sub_data_original: np.ndarray) -> np.ndarray:
    radius = [10, 10, 10]
    alpha, beta = 0.3, 1.0
    sub_itk = sitk.GetImageFromArray(sub_data_original.astype(np.float32))
    sub_enh_itk = sitk.AdaptiveHistogramEqualization(sub_itk, radius, alpha, beta)
    sub_enh = sitk.GetArrayFromImage(sub_enh_itk)

    dilated = dilate_volume(coarse_vol.copy())

    image_refine_enh = dilated.copy().astype(np.float32)
    coords = np.argwhere(dilated)
    for coord in coords:
        x, y, z = coord
        image_refine_enh[x, y, z] = sub_enh[x, y, z]

    if image_refine_enh[dilated].size > 1:
        thresh = threshold_otsu(image_refine_enh, nbins=100)
        segmented = image_refine_enh >= thresh
    else:
        segmented = dilated.copy()

    return segmented.astype(bool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine coarse labels task-wise.")
    parser.add_argument("--task", default=None, help="Task folder under labels/, e.g. task_001")
    args = parser.parse_args()

    if args.task:
        task_dirs = [LABELS_DIR / args.task]
    else:
        task_dirs = sorted(d for d in LABELS_DIR.iterdir() if d.is_dir()) if LABELS_DIR.exists() else []

    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {LABELS_DIR}. Run script 04 first.")

    for task_dir in task_dirs:
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        sub_path = task_dir / "subtraction_mri_org_dim.nii.gz"
        if not sub_path.exists():
            raise FileNotFoundError(f"Subtraction NIfTI not found: {sub_path}\nRun script 04 first.")

        print(f"\nProcessing {task_dir.name} ...")
        sub_nib = nib.load(str(sub_path))
        sub_data = sub_nib.get_fdata().astype(np.float32)
        affine = sub_nib.affine
        print(f"Loaded subtraction volume: {sub_data.shape}")

        coarse_files = sorted(task_dir.glob("*_coarse.nii.gz"))
        if not coarse_files:
            print(f"  No coarse labels in {task_dir.name}; skipping.")
            continue

        print(f"Found {len(coarse_files)} coarse label(s):")
        for p in coarse_files:
            print(f"  {p.name}")

        for coarse_path in coarse_files:
            label_slug = coarse_path.name.replace("_coarse.nii.gz", "")
            print(f"\nRefining '{label_slug}' ...")

            coarse_nib = nib.load(str(coarse_path))
            coarse_vol = coarse_nib.get_fdata().astype(np.float32)

            print("  Applying dilate_volume ...")
            mod = dilate_volume(coarse_vol.copy())
            mod_path = task_dir / f"{label_slug}_moderately_processed.nii.gz"
            nib.Nifti1Image(mod.astype(np.float32), affine).to_filename(str(mod_path))
            print(f"  -> {mod_path.relative_to(REPO_ROOT)}")

            print("  Applying label_refine (AHE + Otsu) ...")
            ext = label_refine(coarse_vol, sub_data)
            ext_path = task_dir / f"{label_slug}_extensively_processed.nii.gz"
            nib.Nifti1Image(ext.astype(np.float32), affine).to_filename(str(ext_path))
            print(f"  -> {ext_path.relative_to(REPO_ROOT)}")

    print("\nScript 05 complete. Run 06_save_outputs.py next.")
