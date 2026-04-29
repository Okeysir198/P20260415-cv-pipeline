#!/usr/bin/env python3
"""Label Studio bridge -- import/export YOLO annotations and set up projects.

Connects the auto-annotation and QA pipelines with Label Studio for human
review workflows. Supports three subcommands:

  import  -- Push YOLO-annotated images into a Label Studio project as
             pre-annotations (from dataset splits, auto-annotate output,
             or QA fixes).
  export  -- Pull reviewed annotations from Label Studio back to YOLO
             label files.
  setup   -- Create a Label Studio project with the correct labeling
             interface for a given data config.

Usage:
    # Set up a project for the shoes dataset
    python utils/label_studio_bridge.py setup \\
        --data-config features/ppe-shoes_detection/configs/05_data.yaml

    # Import auto-annotated images for review
    python utils/label_studio_bridge.py import \\
        --data-config features/ppe-shoes_detection/configs/05_data.yaml \\
        --from-auto-annotate runs/auto_annotate/shoes/

    # Import QA flagged images for review
    python utils/label_studio_bridge.py import \\
        --data-config features/ppe-shoes_detection/configs/05_data.yaml \\
        --from-qa-fixes runs/qa/shoes/fixes.json

    # Export reviewed annotations back to YOLO
    python utils/label_studio_bridge.py export \\
        --project shoes_review \\
        --output-dir dataset_store/shoes_detection/train/labels

    # Dry-run to preview import without API calls
    python utils/label_studio_bridge.py import \\
        --data-config features/ppe-shoes_detection/configs/05_data.yaml --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # pipeline root

import requests
from label_studio_sdk import Client as LSClient
from loguru import logger
from tqdm import tqdm

from utils.config import load_config, resolve_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_DEFAULT_LS_CONFIG_PATH = _PROJECT_ROOT / "configs" / "_shared" / "04_label_studio.yaml"

# Distinct colors for up to 20 class labels in the Label Studio XML config.
_DEFAULT_LS_URL = "http://localhost:18103"

_LABEL_COLORS: list[str] = [
    "#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa",
    "#00acc1", "#ffb300", "#6d4c41", "#546e7a", "#d81b60",
    "#7cb342", "#039be5", "#f4511e", "#3949ab", "#00897b",
    "#c0ca33", "#5e35b1", "#1565c0", "#ef6c00", "#ad1457",
]

# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------


def yolo_to_ls(
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
    class_names: dict[int, str],
    bbox_index: int = 0,
    img_width: int = 0,
    img_height: int = 0,
) -> dict[str, Any]:
    """Convert a single YOLO bbox to a Label Studio rectanglelabels result.

    Args:
        class_id: Integer class index.
        cx: Center-x, normalized 0-1.
        cy: Center-y, normalized 0-1.
        w: Width, normalized 0-1.
        h: Height, normalized 0-1.
        class_names: Mapping from class_id to class name string.
        bbox_index: Index for generating a unique result id.
        img_width: Original image width in pixels (0 if unknown).
        img_height: Original image height in pixels (0 if unknown).

    Returns:
        Label Studio result dictionary for one rectangle annotation.
    """
    x_pct = (cx - w / 2) * 100.0
    y_pct = (cy - h / 2) * 100.0
    w_pct = w * 100.0
    h_pct = h * 100.0

    # Clamp to valid range
    x_pct = max(0.0, min(x_pct, 100.0))
    y_pct = max(0.0, min(y_pct, 100.0))
    w_pct = max(0.0, min(w_pct, 100.0 - x_pct))
    h_pct = max(0.0, min(h_pct, 100.0 - y_pct))

    label_name = class_names.get(class_id, f"class_{class_id}")

    return {
        "id": f"bbox_{bbox_index}",
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": img_width,
        "original_height": img_height,
        "value": {
            "x": round(x_pct, 4),
            "y": round(y_pct, 4),
            "width": round(w_pct, 4),
            "height": round(h_pct, 4),
            "rectanglelabels": [label_name],
            "rotation": 0,
        },
    }


def ls_to_yolo(
    result: dict[str, Any],
    class_name_to_id: dict[str, int],
) -> tuple[int, float, float, float, float] | None:
    """Convert a single Label Studio rectanglelabels result to YOLO format.

    Args:
        result: Label Studio annotation result dict (must have type=rectanglelabels).
        class_name_to_id: Mapping from class name string to integer class id.

    Returns:
        Tuple of (class_id, cx, cy, w, h) in normalized 0-1 coordinates,
        or None if the result cannot be converted.
    """
    if result.get("type") != "rectanglelabels":
        return None

    value = result.get("value", {})
    labels = value.get("rectanglelabels", [])
    if not labels:
        return None

    label_name = labels[0]
    class_id = class_name_to_id.get(label_name)
    if class_id is None:
        logger.warning("Unknown label '%s', skipping annotation.", label_name)
        return None

    x_pct = value.get("x", 0.0)
    y_pct = value.get("y", 0.0)
    w_pct = value.get("width", 0.0)
    h_pct = value.get("height", 0.0)

    cx = (x_pct + w_pct / 2) / 100.0
    cy = (y_pct + h_pct / 2) / 100.0
    w = w_pct / 100.0
    h = h_pct / 100.0

    return (class_id, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


# ---------------------------------------------------------------------------
# Label file I/O
# ---------------------------------------------------------------------------


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file and return list of (class_id, cx, cy, w, h).

    Args:
        label_path: Path to the YOLO .txt label file.

    Returns:
        List of (class_id, cx, cy, w, h) tuples. Returns empty list if the
        file does not exist or is empty.
    """
    if not label_path.exists():
        return []

    annotations: list[tuple[int, float, float, float, float]] = []
    for line_no, line in enumerate(label_path.read_text().strip().splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            logger.warning("%s:%d -- expected 5 fields, got %d; skipping.", label_path, line_no, len(parts))
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((cid, cx, cy, w, h))
        except ValueError:
            logger.warning("%s:%d -- invalid numeric values; skipping.", label_path, line_no)
    return annotations


def write_yolo_labels(
    label_path: Path,
    annotations: list[tuple[int, float, float, float, float]],
) -> None:
    """Write annotations in YOLO format to a label file.

    Args:
        label_path: Destination path for the .txt label file.
        annotations: List of (class_id, cx, cy, w, h) tuples.
    """
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in annotations]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# Label Studio API client abstraction
# ---------------------------------------------------------------------------


class LabelStudioAPI:
    """Thin wrapper around Label Studio API calls.

    Uses ``label_studio_sdk`` when available, falls back to raw ``requests``.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        email: str = "",
        password: str = "",
    ) -> None:
        """Initialize the API client.

        Args:
            url: Label Studio server URL (e.g. ``http://localhost:8080``).
            api_key: API authentication token.
            email: Login email for session-cookie auth (LS 1.23+ fallback).
            password: Login password for session-cookie auth.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self._client = None
        self._session: requests.Session | None = None

        # Always build a session for endpoints not exposed by the SDK
        # (e.g. local-files storage connector). Use email/password login
        # when provided so Django session-auth endpoints work; otherwise
        # fall back to token-auth headers.
        if email and password:
            self._session = self._login_session(email, password)
        else:
            self._session = requests.Session()
            self._session.headers["Authorization"] = f"Token {self.api_key}"

        # Try SDK on top of the session — SDK handles project/task CRUD
        # cleanly. If SDK init fails (e.g. legacy-token disabled), the raw
        # session above is still used for all calls.
        try:
            client = LSClient(url=self.url, api_key=self.api_key)
            client.get_projects()  # validate connectivity
            self._client = client
            logger.info("Using label-studio-sdk client.")
        except Exception:
            logger.info("SDK auth failed; using raw session-cookie auth for all calls.")

    def _login_session(self, email: str, password: str) -> requests.Session:
        """Create an authenticated requests.Session via login form."""
        session = requests.Session()
        session.get(f"{self.url}/user/login/", timeout=10)
        csrf = session.cookies.get("csrftoken", "")
        session.post(
            f"{self.url}/user/login/",
            data={
                "email": email,
                "password": password,
                "csrfmiddlewaretoken": csrf,
            },
            headers={"Referer": f"{self.url}/user/login/"},
            timeout=10,
        )
        return session

    # -- Project operations --------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """Return a list of all projects (id, title).

        Returns:
            List of dicts with at least 'id' and 'title' keys.
        """
        if self._client is not None:
            projects = self._client.get_projects()
            return [{"id": p.id, "title": p.title} for p in projects]

        resp = self._http().get(
            f"{self.url}/api/projects",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        return [{"id": p["id"], "title": p["title"]} for p in results]

    def find_project(self, name: str) -> dict[str, Any] | None:
        """Find a project by exact title match.

        Args:
            name: Project title to search for.

        Returns:
            Project dict with 'id' and 'title', or None if not found.
        """
        for proj in self.list_projects():
            if proj["title"] == name:
                return proj
        return None

    def create_project(
        self,
        title: str,
        label_config: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create a new project.

        Args:
            title: Project title.
            label_config: XML labeling interface configuration.
            description: Optional project description.

        Returns:
            Dict with 'id' and 'title' of the created project.
        """
        if self._client is not None:
            project = self._client.start_project(
                title=title,
                label_config=label_config,
                description=description,
            )
            return {"id": project.id, "title": project.title}

        payload = {
            "title": title,
            "label_config": label_config,
            "description": description,
        }
        resp = self._http().post(
            f"{self.url}/api/projects",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return {"id": data["id"], "title": data["title"]}

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> int:
        """Import tasks (with optional predictions) into a project.

        Args:
            project_id: Target project ID.
            tasks: List of task dicts following the Label Studio task format.

        Returns:
            Number of tasks successfully imported.
        """
        if not tasks:
            return 0

        if self._client is not None:
            project = self._client.get_project(project_id)
            project.import_tasks(tasks)
            return len(tasks)

        resp = self._http().post(
            f"{self.url}/api/projects/{project_id}/import",
            headers=self._headers(),
            json=tasks,
            timeout=120,
        )
        resp.raise_for_status()
        return len(tasks)

    def get_tasks(
        self,
        project_id: int,
        only_reviewed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch all tasks from a project.

        Args:
            project_id: Project ID.
            only_reviewed: If True, return only tasks that have human annotations
                (not just machine predictions).

        Returns:
            List of task dicts with annotations and predictions.
        """
        if self._client is not None:
            project = self._client.get_project(project_id)
            tasks = project.get_tasks()
        else:
            tasks = []
            page = 1
            while True:
                resp = self._http().get(
                    f"{self.url}/api/projects/{project_id}/tasks",
                    headers=self._headers(),
                    params={"page": page, "page_size": 100},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                page_tasks = data.get("tasks", data) if isinstance(data, dict) else data
                if not page_tasks:
                    break
                tasks.extend(page_tasks)
                # Check if there are more pages — LS 1.23 may omit "next"
                has_next = isinstance(data, dict) and data.get("next") is not None
                if not has_next:
                    break
                page += 1

        if only_reviewed:
            tasks = [t for t in tasks if t.get("annotations")]

        return tasks

    def create_local_storage(
        self,
        project_id: int,
        local_store_path: str,
        title: str = "Dataset",
    ) -> dict[str, Any]:
        """Create a local file storage connector for a project.

        Args:
            project_id: Project ID.
            local_store_path: Absolute path on the LS server to the data.
            title: Display title for the storage.

        Returns:
            Storage connector info dict.
        """
        payload = {
            "project": project_id,
            "title": title,
            "path": local_store_path,
            "use_blob_urls": True,
            "regex_filter": r".*\.(jpg|jpeg|png|bmp|webp)$",
        }

        # Local storage API is not exposed via the SDK — always use raw HTTP.
        resp = self._http().post(
            f"{self.url}/api/storages/localfiles",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # -- helpers -------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers for raw requests calls."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._session and self._session.cookies.get("csrftoken"):
            headers["X-CSRFToken"] = self._session.cookies["csrftoken"]
        return headers

    def _http(self) -> requests.Session:
        """Return the authenticated session for HTTP calls."""
        assert self._session is not None, "No authenticated session available"
        return self._session


# ---------------------------------------------------------------------------
# Config / API key resolution
# ---------------------------------------------------------------------------


def resolve_api_key(
    cli_key: str | None,
    config: dict[str, Any],
) -> str:
    """Resolve the Label Studio API key from CLI arg, env var, or config.

    Priority: CLI arg > LS_API_KEY env var > config file value.

    Args:
        cli_key: Value passed via ``--api-key`` (may be None).
        config: Loaded label_studio config dict.

    Returns:
        API key string.

    Raises:
        ValueError: If no API key is found from any source.
    """
    if cli_key:
        return cli_key

    env_key = os.environ.get("LS_API_KEY", "")
    if env_key:
        return env_key

    cfg_key = config.get("label_studio", {}).get("api_key", "")
    if cfg_key:
        return cfg_key

    raise ValueError(
        "No Label Studio API key found. Provide one via:\n"
        "  --api-key <KEY>    CLI argument\n"
        "  LS_API_KEY=<KEY>   environment variable\n"
        "  api_key: <KEY>     in configs/_shared/04_label_studio.yaml"
    )


def load_ls_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the Label Studio config, falling back to defaults.

    Args:
        config_path: Path to a label_studio YAML config. If None, uses the
            default config at ``configs/_shared/04_label_studio.yaml``.

    Returns:
        Label Studio config dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_LS_CONFIG_PATH
    if path.exists():
        return load_config(path)
    # Sensible defaults when no config file exists
    return {
        "label_studio": {
            "url": _DEFAULT_LS_URL,
            "api_key": "",
            "local_files_root": "/datasets",
        }
    }


# ---------------------------------------------------------------------------
# XML label config generation
# ---------------------------------------------------------------------------


def generate_label_config(class_names: dict[int, str]) -> str:
    """Generate Label Studio XML label config for object detection.

    Args:
        class_names: Mapping from integer class ID to display name.

    Returns:
        XML string for the labeling interface.
    """
    label_lines: list[str] = []
    for idx in sorted(class_names.keys()):
        name = class_names[idx]
        color = _LABEL_COLORS[idx % len(_LABEL_COLORS)]
        label_lines.append(f'    <Label value="{name}" background="{color}"/>')

    labels_block = "\n".join(label_lines)
    # <Choices> lets the reviewer reassign the sample's split (train/val/test/drop).
    # The selected value is read back during export and drives the physical move.
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels_block}\n"
        "  </RectangleLabels>\n"
        '  <Header value="Split assignment"/>\n'
        '  <Choices name="split" toName="image" choice="single" showInLine="true">\n'
        '    <Choice value="train"/>\n'
        '    <Choice value="val"/>\n'
        '    <Choice value="test"/>\n'
        '    <Choice value="drop"/>\n'
        "  </Choices>\n"
        "</View>"
    )


# ---------------------------------------------------------------------------
# Path mapping helpers
# ---------------------------------------------------------------------------


def _image_path_to_ls_url(
    image_path: Path,
    local_files_root: str,
    dataset_base: Path,
) -> str:
    """Map an absolute image path to a Label Studio local-files URL.

    The ``?d=`` parameter is resolved by Label Studio **relative to**
    ``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT``, so we must NOT include
    the mount point prefix here.

    Args:
        image_path: Absolute path to the image.
        local_files_root: The mount point configured in LS (unused — kept
            for API compatibility; LS prepends DOCUMENT_ROOT automatically).
        dataset_base: The root directory of the dataset on the host.

    Returns:
        LS local-files URL string.
    """
    try:
        rel = image_path.resolve().relative_to(dataset_base.resolve())
    except ValueError:
        # image_path is outside dataset_base (e.g. auto-annotate temp dir).
        # Search dataset_base for a file with the same name so the URL
        # includes the correct subdirectory (e.g. val/images/).
        matches = list(dataset_base.resolve().rglob(image_path.name))
        rel = matches[0].relative_to(dataset_base.resolve()) if matches else Path(image_path.name)

    # Build path relative to DOCUMENT_ROOT (LS prepends it automatically)
    full_path = f"{dataset_base.name}/{rel.as_posix()}"
    return f"/data/local-files/?d={full_path}"


def _label_path_for_image(image_path: Path) -> Path:
    """Derive the YOLO label path from an image path.

    Follows the standard YOLO layout convention:
      ``<split>/images/foo.jpg`` -> ``<split>/labels/foo.txt``

    Args:
        image_path: Path to the image file.

    Returns:
        Corresponding label file path.
    """
    label_dir = image_path.parent.parent / "labels"
    return label_dir / (image_path.stem + ".txt")


def _ls_url_to_label_path(
    ls_url: str,
    output_dir: Path,
) -> Path:
    """Derive the YOLO label output path from a Label Studio image URL.

    Args:
        ls_url: The ``data.image`` value from a LS task.
        output_dir: Base output directory for label files.

    Returns:
        Path where the YOLO label should be written.
    """
    # Extract filename from URL like /data/local-files/?d=dataset/split/images/foo.jpg
    if "?d=" in ls_url:
        rel_path = ls_url.split("?d=", 1)[1]
        filename = Path(rel_path).stem
    else:
        filename = Path(ls_url).stem

    return output_dir / f"{filename}.txt"


# ---------------------------------------------------------------------------
# Gather image-label pairs
# ---------------------------------------------------------------------------


def gather_dataset_pairs(
    data_config: dict[str, Any],
    splits: list[str],
    config_dir: Path | None = None,
) -> list[tuple[Path, Path, str]]:
    """Gather (image_path, label_path, split_name) triples from dataset splits.

    The split tag is preserved so we can stamp each LS task with its current
    split and track moves during export.

    Args:
        data_config: Loaded data config dict.
        splits: List of split names to include (e.g. ["train", "val", "test"]).
        config_dir: Directory of the data config file (for resolving relative paths).

    Returns:
        List of (image_path, label_path, split_name) triples.
    """
    base = config_dir if config_dir else _PROJECT_ROOT
    dataset_path = resolve_path(data_config["path"], base)
    triples: list[tuple[Path, Path, str]] = []

    for split in splits:
        split_key = split.strip()
        images_subdir = data_config.get(split_key)
        if not images_subdir:
            logger.warning("Split '%s' not defined in data config; skipping.", split_key)
            continue

        images_dir = dataset_path / images_subdir
        if not images_dir.is_dir():
            logger.warning("Image directory does not exist: %s; skipping.", images_dir)
            continue

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                label_file = _label_path_for_image(img_file)
                triples.append((img_file, label_file, split_key))

    return triples


def gather_auto_annotate_pairs(
    auto_annotate_dir: Path,
) -> list[tuple[Path, Path]]:
    """Gather (image_path, label_path) pairs from auto-annotate output.

    Auto-annotate output directories contain ``labels/`` and optionally
    ``previews/`` subdirectories. Images may be referenced by the label
    filenames or found in a sibling ``images/`` directory.

    Args:
        auto_annotate_dir: Path to the auto-annotate output directory.

    Returns:
        List of (image_path, label_path) tuples.
    """
    labels_dir = auto_annotate_dir / "labels"
    if not labels_dir.is_dir():
        logger.warning("No labels/ directory in auto-annotate output: %s", auto_annotate_dir)
        return []

    # Try to find the images directory
    images_dir = auto_annotate_dir / "images"
    if not images_dir.is_dir():
        # Look one level up
        images_dir = auto_annotate_dir.parent / "images"

    pairs: list[tuple[Path, Path]] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        stem = label_file.stem
        img_path: Path | None = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            logger.debug("No image found for label %s; skipping.", label_file.name)
            continue
        pairs.append((img_path, label_file))

    return pairs


def gather_qa_fixes_pairs(
    fixes_path: Path,
) -> list[tuple[Path, Path]]:
    """Gather (image_path, label_path) pairs from a QA fixes.json file.

    Only returns unique image paths that the QA pipeline flagged for review.

    Args:
        fixes_path: Path to the QA ``fixes.json`` file.

    Returns:
        List of (image_path, label_path) tuples.
    """
    if not fixes_path.exists():
        logger.error("QA fixes file not found: %s", fixes_path)
        return []

    with open(fixes_path) as f:
        fixes_data = json.load(f)

    seen: set[str] = set()
    pairs: list[tuple[Path, Path]] = []
    for fix in fixes_data.get("fixes", []):
        img_path_str = fix.get("image_path", "")
        label_path_str = fix.get("label_path", "")
        if not img_path_str or not label_path_str:
            continue
        if img_path_str in seen:
            continue
        seen.add(img_path_str)
        pairs.append((Path(img_path_str), Path(label_path_str)))

    return pairs


# ---------------------------------------------------------------------------
# Build Label Studio tasks
# ---------------------------------------------------------------------------


def build_task(
    image_path: Path,
    label_path: Path,
    class_names: dict[int, str],
    local_files_root: str,
    dataset_base: Path,
    model_version: str = "auto_annotate_v1",
    split: str | None = None,
) -> dict[str, Any]:
    """Build a Label Studio task dict with pre-annotations from a YOLO label.

    Args:
        image_path: Absolute path to the image.
        label_path: Absolute path to the YOLO label file.
        class_names: Mapping from class_id to class name.
        local_files_root: LS local storage mount point.
        dataset_base: Root directory of the dataset on the host.
        model_version: Version string for the prediction source.

    Returns:
        Label Studio task dictionary ready for import.
    """
    ls_url = _image_path_to_ls_url(image_path, local_files_root, dataset_base)
    annotations = read_yolo_labels(label_path)

    results: list[dict[str, Any]] = []
    for i, (cid, cx, cy, w, h) in enumerate(annotations):
        results.append(
            yolo_to_ls(
                class_id=cid,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                class_names=class_names,
                bbox_index=i,
            )
        )

    task: dict[str, Any] = {
        "data": {"image": ls_url},
    }
    if split:
        task["data"]["split"] = split
        # Pre-select the current split in the LS Choices field.
        results.append({
            "from_name": "split",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [split]},
        })

    if results:
        task["predictions"] = [
            {
                "model_version": model_version,
                "result": results,
            }
        ]

    return task


# ---------------------------------------------------------------------------
# Subcommand: import
# ---------------------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> None:
    """Execute the 'import' subcommand.

    Reads YOLO labels and pushes them as pre-annotated tasks into Label Studio.

    Args:
        args: Parsed CLI arguments for the import subcommand.
    """
    # Load configs
    data_config = load_config(args.data_config)
    ls_config = load_ls_config(args.ls_config)
    ls_cfg = ls_config.get("label_studio", {})
    class_names: dict[int, str] = {int(k): v for k, v in data_config["names"].items()}

    config_dir = Path(args.data_config).resolve().parent
    dataset_name = data_config.get("dataset_name", "dataset")
    project_name = args.project or f"{dataset_name}_review"
    dataset_base = resolve_path(data_config["path"], config_dir)
    local_files_root = ls_cfg.get("local_files_root", "/datasets")

    # Gather image-label pairs
    pairs: list[tuple[Path, Path]] = []
    if args.from_auto_annotate:
        aa_dir = Path(args.from_auto_annotate).resolve()
        pairs = gather_auto_annotate_pairs(aa_dir)
        model_version = "auto_annotate_v1"
        logger.info("Gathered %d pairs from auto-annotate output: %s", len(pairs), aa_dir)
    elif args.from_qa_fixes:
        fixes_path = Path(args.from_qa_fixes).resolve()
        pairs = gather_qa_fixes_pairs(fixes_path)
        model_version = "qa_review_v1"
        logger.info("Gathered %d pairs from QA fixes: %s", len(pairs), fixes_path)
    else:
        splits = args.splits.split()
        triples = gather_dataset_pairs(data_config, splits, config_dir)
        model_version = "dataset_v1"
        logger.info("Gathered %d pairs from splits: %s", len(triples), splits)
        pairs = triples  # list of (img, lbl, split)

    if not pairs:
        logger.warning("No image-label pairs found. Nothing to import.")
        return

    # Build tasks
    tasks: list[dict[str, Any]] = []
    for item in tqdm(pairs, total=len(pairs), desc="Building tasks"):
        if len(item) == 3:
            img_path, lbl_path, split_name = item
        else:
            img_path, lbl_path = item
            split_name = None
        task = build_task(
            image_path=img_path,
            label_path=lbl_path,
            class_names=class_names,
            local_files_root=local_files_root,
            dataset_base=dataset_base,
            model_version=model_version,
            split=split_name,
        )
        tasks.append(task)

    logger.info("Built %d tasks for import.", len(tasks))

    if args.dry_run:
        print(f"\n[DRY RUN] Would import {len(tasks)} tasks into project '{project_name}'.")
        print(f"  Label Studio URL: {ls_cfg.get('url', 'http://localhost:8080')}")
        print(f"  Dataset base:     {dataset_base}")
        print(f"  Model version:    {model_version}")
        if tasks:
            print("\n  Sample task (first):")
            print(f"    image: {tasks[0]['data']['image']}")
            preds = tasks[0].get("predictions", [{}])
            n_bboxes = len(preds[0].get("result", [])) if preds else 0
            print(f"    pre-annotations: {n_bboxes} bboxes")
        return

    # Connect and upload
    api_key = resolve_api_key(args.api_key, ls_config)
    api = LabelStudioAPI(
        url=ls_cfg.get("url", _DEFAULT_LS_URL),
        api_key=api_key,
        email=getattr(args, "email", "") or "",
        password=getattr(args, "password", "") or "",
    )

    # Find or create project
    project = api.find_project(project_name)
    if project is None:
        label_xml = generate_label_config(class_names)
        project = api.create_project(
            title=project_name,
            label_config=label_xml,
            description=f"Review project for {dataset_name} dataset.",
        )
        logger.info("Created project '%s' (id=%d).", project["title"], project["id"])
    else:
        logger.info("Found existing project '%s' (id=%d).", project["title"], project["id"])

    # Import in batches to avoid timeouts
    batch_size = 100
    total_imported = 0
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        imported = api.import_tasks(project["id"], batch)
        total_imported += imported
        logger.info(
            "Imported batch %d-%d (%d tasks).",
            i + 1,
            min(i + batch_size, len(tasks)),
            imported,
        )

    print(f"\nImported {total_imported} tasks into project '{project_name}' (id={project['id']}).")
    print(f"  Open in browser: {ls_cfg.get('url', 'http://localhost:8080')}/projects/{project['id']}")


# ---------------------------------------------------------------------------
# Subcommand: export
# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> None:
    """Execute the 'export' subcommand.

    Fetches reviewed annotations from Label Studio and writes YOLO label files.
    When ``--output-dir`` is omitted but ``--data-config`` is provided, the
    output directory defaults to the dataset's ``train/labels/`` directory
    (resolved from the data config) and backup is auto-enabled.

    Args:
        args: Parsed CLI arguments for the export subcommand.
    """
    ls_config = load_ls_config(args.ls_config)
    ls_cfg = ls_config.get("label_studio", {})
    api_key = resolve_api_key(args.api_key, ls_config)
    api = LabelStudioAPI(
        url=ls_cfg.get("url", _DEFAULT_LS_URL),
        api_key=api_key,
        email=getattr(args, "email", "") or "",
        password=getattr(args, "password", "") or "",
    )

    # Resolve dataset root (for split-subdir routing) + legacy output_dir.
    dataset_root: Path | None = None
    if args.data_config is not None:
        config_dir = Path(args.data_config).resolve().parent
        data_config_for_path = load_config(args.data_config)
        dataset_root = resolve_path(data_config_for_path["path"], config_dir)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
        logger.info("Legacy mode: writing all labels to %s (no split-aware routing).", output_dir)
    elif dataset_root is not None:
        output_dir = dataset_root  # split-aware routing will derive per-split paths
        args.backup = True
    else:
        logger.error("Either --output-dir or --data-config must be provided.")
        sys.exit(1)

    # Find project by name or ID
    project: dict[str, Any] | None = None
    try:
        project_id = int(args.project)
        project = {"id": project_id, "title": f"project_{project_id}"}
    except ValueError:
        project = api.find_project(args.project)

    if project is None:
        logger.error("Project '%s' not found.", args.project)
        sys.exit(1)

    logger.info("Exporting from project '%s' (id=%d)...", project["title"], project["id"])

    # Fetch tasks
    tasks = api.get_tasks(project["id"], only_reviewed=args.only_reviewed)
    if not tasks:
        logger.warning("No tasks found (only_reviewed=%s).", args.only_reviewed)
        return

    logger.info("Fetched %d tasks.", len(tasks))

    # We need to know class_name -> class_id mapping.
    # Build it from the first task's annotations or from data config if provided.
    class_name_to_id: dict[str, int] = {}
    if args.data_config:
        dc = load_config(args.data_config)
        class_name_to_id = {v: int(k) for k, v in dc["names"].items()}
    else:
        # Attempt to infer from label config (best effort: gather unique labels)
        logger.warning(
            "No --data-config provided; will attempt to infer class mapping from annotations. "
            "This may produce incorrect class IDs. Use --data-config for reliable export."
        )
        all_labels: set[str] = set()
        for task in tasks:
            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    for lbl in result.get("value", {}).get("rectanglelabels", []):
                        all_labels.add(lbl)
        class_name_to_id = {name: idx for idx, name in enumerate(sorted(all_labels))}
        logger.info("Inferred class mapping: %s", class_name_to_id)

    # Backup existing labels across all split subdirs when in split-aware mode.
    split_aware = dataset_root is not None and args.output_dir is None
    if args.backup:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if split_aware:
            for split in ("train", "val", "test"):
                src = dataset_root / split / "labels"
                if not src.is_dir():
                    continue
                backup_dir = dataset_root / split / f".backup_{timestamp}"
                backup_dir.mkdir(parents=True, exist_ok=True)
                for txt_file in src.glob("*.txt"):
                    shutil.copy2(txt_file, backup_dir / txt_file.name)
        elif output_dir.exists():
            backup_dir = output_dir / f".backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for txt_file in output_dir.glob("*.txt"):
                shutil.copy2(txt_file, backup_dir / txt_file.name)
        logger.info("Label backup done.")

    # Lazy import so legacy (non-split-aware) export path still works.
    if split_aware:
        from core.p00_data_prep.core.splitter import (
            ensure_split_dirs,
            move_sample,
            refresh_audit_snapshot,
        )
        ensure_split_dirs(dataset_root, include_dropped=True)

    written = 0
    skipped = 0
    split_moves: dict[str, int] = {"train": 0, "val": 0, "test": 0, "drop": 0}

    for task in tqdm(tasks, total=len(tasks), desc="Exporting"):
        annotations_list = task.get("annotations", [])
        if not annotations_list:
            skipped += 1
            continue

        annotation = annotations_list[-1]  # latest
        results = annotation.get("result", [])

        # Extract split choice (if any) + bbox results separately.
        new_split: str | None = None
        yolo_annotations: list[tuple[int, float, float, float, float]] = []
        for result in results:
            if result.get("type") == "choices" and result.get("from_name") == "split":
                choices = result.get("value", {}).get("choices") or []
                if choices:
                    new_split = choices[0]
                continue
            converted = ls_to_yolo(result, class_name_to_id)
            if converted is not None:
                yolo_annotations.append(converted)

        image_url = task.get("data", {}).get("image", "")
        current_split = task.get("data", {}).get("split")
        stem = Path(image_url.split("?d=", 1)[-1] if "?d=" in image_url else image_url).stem

        if split_aware and current_split:
            target_split = new_split or current_split
            # Write corrected labels to the CURRENT split first, then move if needed.
            cur_label_dir = dataset_root / current_split / "labels"
            cur_label_dir.mkdir(parents=True, exist_ok=True)
            cur_label_path = cur_label_dir / f"{stem}.txt"
            write_yolo_labels(cur_label_path, yolo_annotations)

            if target_split != current_split:
                move_sample(
                    dataset_root,
                    stem,
                    from_split=current_split,
                    to_split=target_split,
                )
                split_moves[target_split] = split_moves.get(target_split, 0) + 1
        else:
            # Legacy mode: write everything to the single output_dir.
            output_dir.mkdir(parents=True, exist_ok=True)
            label_path = _ls_url_to_label_path(image_url, output_dir)
            write_yolo_labels(label_path, yolo_annotations)

        written += 1

    if split_aware:
        counts = refresh_audit_snapshot(
            dataset_root,
            seed=data_config_for_path.get("splits", {}).get("seed", 42),
            ratios=(0.8, 0.1, 0.1),
        )
        print(f"\nExported {written} tasks (split-aware).")
        for split, n in counts.items():
            print(f"  {split}: {n} imgs")
        total_moved = sum(split_moves.values())
        if total_moved:
            print(f"  Split moves applied: {split_moves}")
    else:
        print(f"\nExported {written} label files to {output_dir}")

    if skipped:
        print(f"  Skipped {skipped} tasks (no annotations).")


# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    """Execute the 'setup' subcommand.

    Creates a Label Studio project with the correct labeling interface and
    optionally configures a local storage connector.

    Args:
        args: Parsed CLI arguments for the setup subcommand.
    """
    data_config = load_config(args.data_config)
    ls_config = load_ls_config(args.ls_config)
    ls_cfg = ls_config.get("label_studio", {})

    class_names: dict[int, str] = {int(k): v for k, v in data_config["names"].items()}
    dataset_name = data_config.get("dataset_name", "dataset")
    project_name = args.project or f"{dataset_name}_review"

    # Generate label config XML
    label_xml = generate_label_config(class_names)
    print(f"Generated labeling interface:\n{label_xml}\n")

    if args.dry_run:
        print(f"[DRY RUN] Would create project '{project_name}' with the above config.")
        return

    api_key = resolve_api_key(args.api_key, ls_config)
    api = LabelStudioAPI(
        url=ls_cfg.get("url", _DEFAULT_LS_URL),
        api_key=api_key,
        email=getattr(args, "email", "") or "",
        password=getattr(args, "password", "") or "",
    )

    # Check if project already exists
    existing = api.find_project(project_name)
    if existing is not None:
        print(f"Project '{project_name}' already exists (id={existing['id']}).")
        print(f"  Open: {ls_cfg.get('url', 'http://localhost:8080')}/projects/{existing['id']}")
        return

    # Create project
    project = api.create_project(
        title=project_name,
        label_config=label_xml,
        description=f"Annotation review project for {dataset_name}.",
    )
    print(f"Created project '{project['title']}' (id={project['id']}).")

    # Set up local storage connector.
    # Storage path must be a sub-dir of LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
    # (not the top-level mount — LS rejects paths equal to DOCUMENT_ROOT).
    config_dir = Path(args.data_config).resolve().parent
    dataset_path = resolve_path(data_config["path"], config_dir)
    document_root = ls_cfg.get("document_root", ls_cfg.get("local_files_root", "/datasets/training_ready"))

    try:
        storage = api.create_local_storage(
            project_id=project["id"],
            local_store_path=str(Path(document_root) / dataset_path.name),
            title=f"{dataset_name} local storage",
        )
        print(f"Created local storage connector (id={storage.get('id', '?')}).")
    except Exception as exc:
        logger.warning(
            "Could not create local storage connector: %s. "
            "You may need to configure it manually in the Label Studio UI.",
            exc,
        )

    print(f"\n  Open in browser: {ls_cfg.get('url', 'http://localhost:8080')}/projects/{project['id']}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with import/export/setup subcommands.

    Returns:
        Configured argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="label_studio_bridge",
        description=(
            "Bridge between YOLO annotation pipelines and Label Studio. "
            "Supports importing pre-annotations, exporting reviewed labels, "
            "and setting up projects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Login email for session-cookie auth (LS 1.23+ fallback when legacy token auth is disabled).",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Login password for session-cookie auth (LS 1.23+ fallback).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands.")

    # -- import subcommand ---------------------------------------------------
    p_import = subparsers.add_parser(
        "import",
        help="Import YOLO-annotated images into Label Studio as pre-annotations.",
        description=(
            "Reads YOLO label files alongside images and pushes them into a "
            "Label Studio project as pre-annotated tasks for human review."
        ),
    )
    p_import.add_argument(
        "--data-config",
        required=True,
        help="Path to data config YAML (e.g. features/ppe-shoes_detection/configs/05_data.yaml).",
    )
    p_import.add_argument(
        "--from-auto-annotate",
        default=None,
        help="Path to auto-annotate output directory (contains labels/ and previews/).",
    )
    p_import.add_argument(
        "--from-qa-fixes",
        default=None,
        help="Path to QA fixes.json file (import only flagged images).",
    )
    p_import.add_argument(
        "--ls-config",
        default=None,
        help="Path to Label Studio config YAML (default: configs/_shared/04_label_studio.yaml).",
    )
    p_import.add_argument(
        "--api-key",
        default=None,
        help="Label Studio API key (overrides LS_API_KEY env var and config file).",
    )
    p_import.add_argument(
        "--project",
        default=None,
        help="Label Studio project name (default: {dataset_name}_review).",
    )
    p_import.add_argument(
        "--splits",
        default="train val test",
        help="Space-separated list of splits to import (default: 'train val test').",
    )
    p_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be imported without making API calls.",
    )
    p_import.set_defaults(func=cmd_import)

    # -- export subcommand ---------------------------------------------------
    p_export = subparsers.add_parser(
        "export",
        help="Export reviewed annotations from Label Studio to YOLO format.",
        description=(
            "Fetches completed annotations from a Label Studio project and "
            "writes them as YOLO-format label files."
        ),
    )
    p_export.add_argument(
        "--project",
        required=True,
        help="Label Studio project name or numeric ID.",
    )
    p_export.add_argument(
        "--output-dir",
        required=False,
        default=None,
        help=(
            "Directory to write YOLO label files "
            "(defaults to dataset train/labels when --data-config provided)."
        ),
    )
    p_export.add_argument(
        "--data-config",
        default=None,
        help="Path to data config YAML for class name mapping (recommended).",
    )
    p_export.add_argument(
        "--ls-config",
        default=None,
        help="Path to Label Studio config YAML (default: configs/_shared/04_label_studio.yaml).",
    )
    p_export.add_argument(
        "--api-key",
        default=None,
        help="Label Studio API key (overrides LS_API_KEY env var and config file).",
    )
    p_export.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing label files before overwriting.",
    )
    p_export.add_argument(
        "--only-reviewed",
        action="store_true",
        help="Only export tasks with human annotations (not just predictions).",
    )
    p_export.set_defaults(func=cmd_export)

    # -- setup subcommand ----------------------------------------------------
    p_setup = subparsers.add_parser(
        "setup",
        help="Create a Label Studio project configured for a dataset.",
        description=(
            "Creates a new Label Studio project with the correct labeling "
            "interface (RectangleLabels) and local storage connector for "
            "a given dataset."
        ),
    )
    p_setup.add_argument(
        "--data-config",
        required=True,
        help="Path to data config YAML (e.g. features/ppe-shoes_detection/configs/05_data.yaml).",
    )
    p_setup.add_argument(
        "--ls-config",
        default=None,
        help="Path to Label Studio config YAML (default: configs/_shared/04_label_studio.yaml).",
    )
    p_setup.add_argument(
        "--api-key",
        default=None,
        help="Label Studio API key (overrides LS_API_KEY env var and config file).",
    )
    p_setup.add_argument(
        "--project",
        default=None,
        help="Project name (default: {dataset_name}_review).",
    )
    p_setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the project config without creating it.",
    )
    p_setup.set_defaults(func=cmd_setup)

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    import sys
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
