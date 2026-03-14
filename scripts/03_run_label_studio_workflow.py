from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import requests
import yaml
from label_studio_sdk import Client


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_01 = REPO_ROOT / "scripts" / "01_compute_task_parameters.py"
SCRIPT_02 = REPO_ROOT / "scripts" / "02_create_composite_mip_and_upload.py"

LS_CONFIG_YAML = REPO_ROOT / "label-studio" / "script" / "config" / "label_studio_config.yaml"
IMAGES_ROOT = REPO_ROOT / "label-studio" / "label-studio" / "images"
INPUT_MIPS_ROOT = IMAGES_ROOT / "input_mips"
TASK_FILE = IMAGES_ROOT / "task_file.txt"
DATA_ROOT = REPO_ROOT / "label-studio" / "label-studio" / "data"


def run_python_script(path: Path, extra_args: list[str] | None = None) -> None:
    print(f"Running {path.name} ...")
    cmd = [sys.executable, str(path)] + (extra_args or [])
    subprocess.run(cmd, check=True)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tasks(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"Task file not found: {file_path}")

    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Task file is empty: {file_path}")

    # Supports JSON object/list 
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    tasks: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        tasks.append(ast.literal_eval(line))
    return tasks


def create_local_storage(label_studio_url: str, api_key: str, project_id: int, local_path: str) -> None:
    storage_settings = {
        "type": "localfiles",
        "path": local_path,
        "title": "Local Image Storage",
        "description": "Storage for MIP images",
        "project": project_id,
    }

    response = requests.post(
        f"{label_studio_url}/api/storages/localfiles",
        headers={"Authorization": f"Token {api_key}"},
        json=storage_settings,
        timeout=30,
    )

    if response.status_code == 201:
        print("Storage creation response:", response.json())
        return

    # Reuse existing storage without failing on duplicate path/project setups.
    if response.status_code == 400 and "already exists" in response.text.lower():
        print("Storage already exists for this project/path. Continuing.")
        return

    print("Storage creation response:", response.status_code, response.text)
    response.raise_for_status()


def select_project(ls: Client, project_id: int | None, project_title: str, description: str, label_config: str, expert_instruction: str):
    if project_id is not None:
        project = ls.get_project(project_id)
        print("Using existing project:", project.id)
        return project

    existing = [p for p in ls.get_projects() if p.params.get("title") == project_title]
    if existing:
        project = max(existing, key=lambda p: int(p.id))
        print("Reusing existing project:", project.id)
        return project

    project = ls.start_project(
        title=project_title,
        description=description,
        label_config=label_config,
        expert_instruction=expert_instruction,
        show_instructions=True,
        show_skip_button=True,
        enable_empty_annotation=False,
        show_annotation_history=False,
    )
    print("New project created:", project)
    print("Project ID:", project.id)
    return project


def get_project_tasks(label_studio_url: str, api_key: str, project_id: int) -> list[dict]:
    response = requests.get(
        f"{label_studio_url}/api/tasks",
        headers={"Authorization": f"Token {api_key}"},
        params={"project": project_id, "page_size": 100},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "tasks" in payload and isinstance(payload["tasks"], list):
            return payload["tasks"]
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
    return []


def task_signature(task: dict) -> tuple:
    data = task.get("data", {})
    return (
        data.get("image_url"),
        data.get("patient_id"),
        data.get("study_id"),
        data.get("study_date"),
        data.get("laterality"),
    )


def task_name_from_task_data(task: dict, default_index: int) -> str:
    image_url = str(task.get("data", {}).get("image_url", ""))
    stem = Path(image_url).stem
    if stem.isdigit():
        return f"task_{int(stem):03d}"
    match = re.search(r"(\d+)", stem)
    if match:
        return f"task_{int(match.group(1)):03d}"
    return f"task_{default_index:03d}"


def normalize_imported_ids(response) -> list[int]:
    if isinstance(response, list):
        if all(isinstance(x, int) for x in response):
            return response
        out: list[int] = []
        for item in response:
            if isinstance(item, dict) and "id" in item:
                out.append(int(item["id"]))
        return out
    if isinstance(response, dict):
        if "task_ids" in response and isinstance(response["task_ids"], list):
            return [int(x) for x in response["task_ids"]]
        if "id" in response:
            return [int(response["id"])]
    return []


def discover_task_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Task data root not found: {data_root}")
    task_dirs = sorted(
        d for d in data_root.iterdir() if d.is_dir() and d.name.lower().startswith("task_")
    )
    if not task_dirs:
        task_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in: {data_root}")
    return task_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripts 01/02, then create/import Label Studio task(s) with working local file storage."
    )
    parser.add_argument("--skip-generation", action="store_true", help="Skip scripts 01 and 02")
    parser.add_argument("--task", default=None, help="Task folder name (e.g. task_001) passed to scripts 01 and 02")
    parser.add_argument("--project-id", type=int, default=None, help="Import into existing project id")
    parser.add_argument("--project-title", default="MIP Annotation Project", help="Project title when creating a new project")
    parser.add_argument(
        "--description",
        default="Project for MIP annotation ingestion",
        help="Project description when creating a new project",
    )
    args = parser.parse_args()

    tasks: list[dict] = []
    if not args.skip_generation:
        if args.task:
            task_dirs = [DATA_ROOT / args.task]
            if not task_dirs[0].is_dir():
                raise FileNotFoundError(f"Task directory not found: {task_dirs[0]}")
        else:
            task_dirs = discover_task_dirs(DATA_ROOT)

        print(f"Generating MIPs/tasks for {len(task_dirs)} data folder(s) ...")
        for task_dir in task_dirs:
            task_name = task_dir.name
            print(f"  Processing {task_name}")
            task_args = ["--task", task_name]
            run_python_script(SCRIPT_01, task_args)
            run_python_script(SCRIPT_02, task_args)

            generated = parse_tasks(TASK_FILE)
            if not generated:
                raise ValueError(f"No tasks generated for {task_name}")

            # Keep only the task corresponding to this folder.
            matching = [t for t in generated if task_name_from_task_data(t, 0) == task_name]
            tasks.extend(matching if matching else generated[:1])
    else:
        tasks = parse_tasks(TASK_FILE)
        if args.task:
            requested = args.task
            filtered = [t for t in tasks if task_name_from_task_data(t, 0) == requested]
            if not filtered:
                raise ValueError(
                    f"Requested --task {requested} not found in {TASK_FILE}. "
                    "Run without --skip-generation to regenerate task_file(s)."
                )
            tasks = filtered

    # Deduplicate tasks by signature while preserving order.
    seen_sigs: set[tuple] = set()
    deduped_tasks: list[dict] = []
    for task_obj in tasks:
        sig = task_signature(task_obj)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        deduped_tasks.append(task_obj)
    tasks = deduped_tasks

    config = load_config(LS_CONFIG_YAML)
    label_studio_url = f"http://{config['HOST']}:{config['PORT']}"
    api_key = config["API_KEY"]

    print("Connecting to Label Studio:", label_studio_url)
    ls = Client(url=label_studio_url, api_key=api_key)
    print(ls.check_connection())

    label_config = (REPO_ROOT / "label-studio" / "script" / "config" / "interface_config.xml").read_text(encoding="utf-8")
    expert_instruction = (REPO_ROOT / "label-studio" / "script" / "config" / "interface_instruction.html").read_text(encoding="utf-8")
    project = select_project(
        ls=ls,
        project_id=args.project_id,
        project_title=args.project_title,
        description=args.description,
        label_config=label_config,
        expert_instruction=expert_instruction,
    )

    create_local_storage(
        label_studio_url=label_studio_url,
        api_key=api_key,
        project_id=project.id,
        local_path=str(INPUT_MIPS_ROOT),
    )

    if not tasks:
        raise ValueError("No tasks found to import.")

    image_url = tasks[0].get("data", {}).get("image_url", "")

    existing_tasks = get_project_tasks(label_studio_url, api_key, int(project.id))
    existing_sigs = {task_signature(t): int(t.get("id")) for t in existing_tasks}

    new_tasks = [t for t in tasks if task_signature(t) not in existing_sigs]
    imported_task_ids: list[int] = []
    imported_by_sig: dict[tuple, int] = {}

    if new_tasks:
        response = project.import_tasks(new_tasks)
        imported_task_ids = normalize_imported_ids(response)
        for task_obj, tid in zip(new_tasks, imported_task_ids):
            imported_by_sig[task_signature(task_obj)] = int(tid)
        print("Import response:", response)
    else:
        print("No new tasks to import. Existing task already present.")

    task_records: list[dict] = []
    for i, task_obj in enumerate(tasks, start=1):
        sig = task_signature(task_obj)
        task_id = existing_sigs.get(sig) or imported_by_sig.get(sig)
        if task_id is None:
            continue
        task_records.append(
            {
                "task_id": int(task_id),
                "task_name": task_name_from_task_data(task_obj, i),
                "image_url": task_obj.get("data", {}).get("image_url", ""),
            }
        )

    if not task_records:
        raise RuntimeError("Could not resolve task IDs for imported/existing tasks.")

    selected_task_id = int(task_records[0]["task_id"])
    all_task_ids = [int(t["task_id"]) for t in task_records]

    task_ids_file = IMAGES_ROOT / "task_ids.json"
    task_ids_file.write_text(
        json.dumps(
            {
                "project_id": project.id,
                "task_id": selected_task_id,
                "task_ids": all_task_ids,
                "tasks": task_records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Label Studio URL:", label_studio_url)
    print("Input MIPs root:", INPUT_MIPS_ROOT)
    print("Task file:", TASK_FILE)
    print("First task image URL:", image_url)
    print("Resolved task IDs:", all_task_ids)
    print("Task IDs file:", task_ids_file)
    print("Project URL:", f"{label_studio_url}/projects/{project.id}")


if __name__ == "__main__":
    main()
