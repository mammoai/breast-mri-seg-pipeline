from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.mri import MRISeries, pipeline


REPO_ROOT = Path(__file__).parent.parent
DATA_ROOT = REPO_ROOT / "label-studio" / "label-studio" / "data"
OUTPUT_DIR = REPO_ROOT / "label-studio" / "label-studio" / "images"
TASKIDS_FILE = OUTPUT_DIR / "task_ids.json"
LABELS_DIR = OUTPUT_DIR / "labels"
LS_CONFIG_YAML = REPO_ROOT / "label-studio" / "script" / "config" / "label_studio_config.yaml"


def find_task_dir(task_name: str | None = None) -> Path:
    if task_name:
        task_dir = DATA_ROOT / task_name
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        return task_dir
    task_dirs = sorted(d for d in DATA_ROOT.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {DATA_ROOT}")
    return task_dirs[0]


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


def task_name_from_task_payload(task: dict, fallback_task_id: int) -> str:
    """Derive task folder name from image_url (001.png -> task_001)."""
    image_url = task.get("data", {}).get("image_url", "")
    stem = Path(str(image_url)).stem
    if stem.isdigit():
        return f"task_{int(stem):03d}"
    match = re.search(r"(\d+)", stem)
    if match:
        return f"task_{int(match.group(1)):03d}"
    return f"task_{int(fallback_task_id):03d}"


def load_ls_config() -> tuple[str, str]:
    ls_url = os.environ.get("LS_URL", "")
    ls_token = os.environ.get("LS_TOKEN", "")

    if (not ls_url or not ls_token) and LS_CONFIG_YAML.exists():
        config = yaml.safe_load(LS_CONFIG_YAML.read_text(encoding="utf-8"))
        ls_url = ls_url or f"http://{config['HOST']}:{config['PORT']}"
        ls_token = ls_token or str(config["API_KEY"])

    if not ls_url or not ls_token:
        raise ValueError(
            "Missing Label Studio credentials. Set LS_URL and LS_TOKEN, "
            "or fill label-studio/script/config/label_studio_config.yaml"
        )

    return ls_url, ls_token


def decode_brush_regions(task: dict) -> dict[str, np.ndarray]:
    """Decode brush annotations from a Label Studio task.

    Scans ALL annotations (not just the last one) and, for each label class,
    uses the masks from the most recent annotation that contains that label.
    This allows the user to submit different label classes in separate
    annotation sessions without one overriding another.
    """
    try:
        from label_studio_converter.brush import decode_rle
    except ImportError as exc:
        raise ImportError("Install label-studio-converter: pip install label-studio-converter") from exc

    annotations = task.get("annotations", [])
    if not annotations:
        return {}

    # Sort ascending so later annotations overwrite earlier ones for the same label.
    sorted_annotations = sorted(annotations, key=lambda a: a.get("id", 0))

    # label -> (annotation_id, list of masks)  — keeps the latest per label class.
    latest_masks_by_label: dict[str, tuple[int, list[np.ndarray]]] = {}

    for annotation in sorted_annotations:
        ann_id = annotation.get("id", 0)
        regions = annotation.get("result", [])

        # Collect all brush regions from this annotation, grouped by label.
        regions_in_ann: dict[str, list[np.ndarray]] = defaultdict(list)
        for region in regions:
            label = region.get("value", {}).get("brushlabels", [None])[0]
            rle = region.get("value", {}).get("rle")
            if rle is None or label is None:
                continue
            width = region["original_width"]
            height = region["original_height"]
            mask = np.reshape(decode_rle(rle), [height, width, 4])[:, :, 3].astype(bool)
            regions_in_ann[label].append(mask)

        # For each label present in this annotation, overwrite any older entry.
        for label, masks in regions_in_ann.items():
            existing = latest_masks_by_label.get(label)
            if existing is None or ann_id >= existing[0]:
                latest_masks_by_label[label] = (ann_id, masks)

    return {
        label: np.any(masks, axis=0)
        for label, (_, masks) in latest_masks_by_label.items()
    }


def close_breast_morph_and_fill(im: np.ndarray) -> None:
    image = np.array(im).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(10, 20))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 0, 1)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        image = cv2.drawContours(image, [hull], 0, 255, -1)
    im[:, :] = image.astype(bool)


def project_label_to_3d(
    mask_2d: np.ndarray,
    label: str,
    bboxes: list,
    resampled_shape: list,
    original_z: int,
) -> np.ndarray:
    (rx0, ry0, rz0), (rxs, rys, rzs) = bboxes[0]
    (lx0, ly0, lz0), (lxs, lys, lzs) = bboxes[1]
    rx1, ry1, rz1 = rx0 + rxs, ry0 + rys, rz0 + rzs
    lx1, ly1, lz1 = lx0 + lxs, ly0 + lys, lz0 + lzs

    original_width = rys + lys
    original_height = rxs + rzs

    im = np.array(
        Image.fromarray(mask_2d.astype("uint8") * 255).resize((original_width, original_height))
    ).astype(bool)

    r_sag = im[0:rzs, 0:rys].copy()
    l_sag = im[0:lzs, lys:].copy()
    r_ax = im[rzs:, 0:rys].copy()
    l_ax = im[lzs:, lys:].copy()

    if label == "Breast":
        for view in (r_sag, l_sag, r_ax, l_ax):
            close_breast_morph_and_fill(view)

    l_ax = np.transpose(l_ax)
    r_ax = np.flip(np.transpose(r_ax), axis=[1, 0])
    r_sag = np.flip(r_sag, axis=1)

    breast_box_shape = (l_sag.shape[0], l_sag.shape[1], l_ax.shape[1])
    l_ax_3d = np.broadcast_to(l_ax[np.newaxis, ...], breast_box_shape).copy()
    l_sag_3d = np.broadcast_to(l_sag[..., np.newaxis], breast_box_shape).copy()
    r_ax_3d = np.broadcast_to(r_ax[np.newaxis, ...], breast_box_shape).copy()
    r_sag_3d = np.broadcast_to(r_sag[..., np.newaxis], breast_box_shape).copy()

    l_intersect = np.flip(np.logical_and(l_ax_3d, l_sag_3d), [0, 1])
    r_intersect = np.flip(np.logical_and(r_ax_3d, r_sag_3d), [0, 1])

    label_vol = np.zeros(list(reversed(resampled_shape)), dtype=bool)
    label_vol[rz0:rz1, ry0:ry1, rx0:rx1] = r_intersect
    label_vol[lz0:lz1, ly0:ly1, lx0:lx1] = l_intersect

    label_xyz = label_vol.astype(np.float32).transpose([2, 1, 0])
    scale_z = original_z / label_xyz.shape[2]
    label_org = ndimage.zoom(
        label_xyz,
        (1.0, 1.0, scale_z),
        order=1,
        mode="nearest",
        prefilter=False,
        grid_mode=True,
    )

    # Keep [x, y, z] order so saved labels match subtraction NIfTI dimensions.
    return label_org.astype(np.float32)


def fetch_task(task_id: int, project_id: int, ls_url: str, ls_api_key: str) -> dict:
    try:
        from label_studio_sdk import Client
    except ImportError as exc:
        raise ImportError("Install label-studio-sdk: pip install label-studio-sdk") from exc

    ls = Client(url=ls_url, api_key=ls_api_key)
    project = ls.get_project(project_id)
    return project.get_task(task_id)


def resolve_task_entries(
    cli_project_id: int | None,
    cli_task_id: int | None,
    cli_task_name: str | None,
) -> tuple[int, list[tuple[str, int]]]:
    if cli_project_id is not None and cli_task_id is not None:
        task_name = cli_task_name or f"task_{int(cli_task_id):03d}"
        return int(cli_project_id), [(task_name, int(cli_task_id))]

    if not TASKIDS_FILE.exists():
        raise FileNotFoundError(
            f"Task IDs file not found: {TASKIDS_FILE}\n"
            "Provide --project-id and --task-id, or create task_ids.json."
        )

    ids = json.loads(TASKIDS_FILE.read_text(encoding="utf-8"))
    project_id = int(cli_project_id if cli_project_id is not None else ids["project_id"])

    if cli_task_id is not None:
        task_name = cli_task_name or f"task_{int(cli_task_id):03d}"
        return project_id, [(task_name, int(cli_task_id))]


    entries: list[tuple[str, int]] = []

    if isinstance(ids.get("tasks"), list) and ids["tasks"]:
        if cli_task_name:
            for item in ids["tasks"]:
                if str(item.get("task_name")) == cli_task_name:
                    entries.append((str(item.get("task_name")), int(item["task_id"])))
                    break
            if not entries:
                raise ValueError(
                    f"Task name '{cli_task_name}' not found in {TASKIDS_FILE}. "
                    "Run script 03 again to refresh task_ids.json."
                )
        else:
            entries = [(str(item.get("task_name")), int(item["task_id"])) for item in ids["tasks"]]
        return project_id, entries

    if cli_task_name:
        raise ValueError(
            f"Task name '{cli_task_name}' cannot be resolved from legacy {TASKIDS_FILE}. "
            "Run script 03 again to refresh task_ids.json with a tasks list."
        )

    if "task_id" in ids:
        return project_id, [(f"task_{int(ids['task_id']):03d}", int(ids["task_id"]))]

    if isinstance(ids.get("task_ids"), list) and ids["task_ids"]:
        return project_id, [(f"task_{int(tid):03d}", int(tid)) for tid in ids["task_ids"]]

    raise ValueError(f"No task ids found in {TASKIDS_FILE}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch LS annotation and project to coarse 3D labels")
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task", default=None, help="Task folder name, e.g. task_001")
    args = parser.parse_args()

    project_id, task_entries = resolve_task_entries(args.project_id, args.task_id, args.task)
    ls_url, ls_api_key = load_ls_config()

    all_saved: dict[str, list[str]] = {}
    for entry_task_name, task_id in task_entries:
        task_dir = find_task_dir(entry_task_name)
        series_pre, series_post = find_series_dirs(task_dir)

        print(f"\nProcessing {entry_task_name} (LS task {task_id}) ...")
        print("Loading DICOM series ...")
        mri_pre = MRISeries.from_path(series_pre)
        mri_post = MRISeries.from_path(series_post)
        sub_mri = mri_post - mri_pre
        original_z = mri_pre.dims[2]
        results = pipeline(mri_pre, mri_post)
        bboxes = results["bounding_boxes"]
        resampled_shape, _ = mri_pre.get_z_resampling_size_and_spacing()

        print(f"Fetching task {task_id} from Label Studio project {project_id} ...")
        task = fetch_task(task_id, project_id, ls_url, ls_api_key)
        task_name = entry_task_name or task_name_from_task_payload(task, task_id)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        task_labels_dir = LABELS_DIR / task_name
        task_labels_dir.mkdir(parents=True, exist_ok=True)
        sub_path = task_labels_dir / "subtraction_mri_org_dim.nii.gz"
        print(f"Saving subtraction NIfTI (original dims) -> {sub_path.relative_to(REPO_ROOT)}")
        sitk.WriteImage(sub_mri.itk_image, str(sub_path))

        print("Decoding brush annotations ...")
        label_masks = decode_brush_regions(task)
        if not label_masks:
            print("  No completed annotations found; skipping this task.")
            all_saved[task_name] = []
            continue

        print(f"  Labels found: {list(label_masks.keys())}")

        saved: list[str] = []
        for label, mask in label_masks.items():
            print(f"  Projecting '{label}' ...")
            vol = project_label_to_3d(mask, label, bboxes, resampled_shape, original_z)
            label_slug = label.lower().replace(" ", "_")
            out_path = task_labels_dir / f"{label_slug}_coarse.nii.gz"
            nib.Nifti1Image(vol, np.eye(4)).to_filename(str(out_path))
            print(f"    -> {out_path.relative_to(REPO_ROOT)}")
            saved.append(label_slug)
        all_saved[task_name] = saved

    print(f"\nScript 04 complete. Saved labels by task: {all_saved}")
    print("Run 05_refine_labels.py next.")
