"""
Roboflow SDK MCP Server
Exposes Roboflow platform API (datasets, training, Universe, inference) as Claude Code tools.
"""

import io
import os
import sys
import contextlib
import threading
from pathlib import Path
import requests
from fastmcp import FastMCP


def _load_dotenv() -> None:
    """Populate os.environ from the nearest .env file walking up from this script."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        env_path = parent / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                # Override placeholder values like literal "${VAR}" that Claude Code may pass
                existing = os.environ.get(key, "")
                if not existing or (existing.startswith("${") and existing.endswith("}")):
                    os.environ[key] = val
            return


_load_dotenv()

mcp = FastMCP("Roboflow")

# ---------------------------------------------------------------------------
# Lazy SDK init — authenticate once per session, suppress output
# (Roboflow SDK prints to stdout on init which corrupts MCP stdio transport)
# ---------------------------------------------------------------------------

_rf = None
_ws = None
_init_lock = threading.Lock()


@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout and stderr during SDK calls to protect the MCP stdio transport."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def get_rf():
    """Lazy singleton — initializes the Roboflow client once on first call."""
    global _rf
    if _rf is None:
        with _init_lock:
            if _rf is None:
                api_key = os.environ.get("ROBOFLOW_API_KEY")
                if not api_key:
                    raise ValueError("ROBOFLOW_API_KEY environment variable is required")
                import roboflow
                with _suppress_output():
                    _rf = roboflow.Roboflow(api_key=api_key)
    return _rf


def get_ws(workspace_url: str = ""):
    """Lazy singleton for the default workspace; bypasses cache when workspace_url is given."""
    global _ws
    rf = get_rf()
    if workspace_url:
        with _suppress_output():
            return rf.workspace(workspace_url)
    if _ws is None:
        with _init_lock:
            if _ws is None:
                with _suppress_output():
                    _ws = rf.workspace()
    return _ws


# ---------------------------------------------------------------------------
# Workspace & Project tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_workspaces() -> dict:
    """List the workspace accessible with the current API key.
    Returns workspace name, URL slug, and number of projects."""
    ws = get_ws()
    return {
        "name": ws.name,
        "url": ws.url,
        "project_count": len(ws.project_list),
    }


@mcp.tool()
def list_projects(workspace_url: str = "") -> list:
    """List all projects in a workspace.
    Each entry includes project id, name, type, annotation group, and image count.
    Leave workspace_url empty to use the default workspace."""
    ws = get_ws(workspace_url)
    return [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "type": p.get("type"),
            "annotation": p.get("annotation", ""),
            "images": p.get("images", 0),
            "classes": p.get("classes", {}),
        }
        for p in ws.project_list
    ]


@mcp.tool()
def get_project(project_url: str, workspace_url: str = "") -> dict:
    """Get detailed info about a project: class list, image count, annotation type.
    project_url is the URL slug (e.g. 'my-dataset').
    Leave workspace_url empty for the default workspace."""
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
    return {
        "id": proj.id,
        "name": proj.name,
        "type": proj.type,
        "annotation": proj.annotation,
        "classes": proj.classes,
    }


# ---------------------------------------------------------------------------
# Dataset Version tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_versions(project_url: str, workspace_url: str = "") -> list:
    """List all dataset versions for a project.
    Returns version number, id, and image counts per split."""
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
        versions = proj.versions()
    return [
        {
            "version": v.version,
            "id": v.id,
            "images": v.images,
        }
        for v in versions
    ]


@mcp.tool()
def upload_image(
    project_url: str,
    image_path: str,
    annotation_path: str = "",
    split: str = "train",
    batch_name: str = "",
    workspace_url: str = "",
) -> dict:
    """Upload an image (and optional annotation) to a Roboflow project.
    image_path: absolute local path to the image file.
    annotation_path: optional absolute path to annotation file (PASCAL VOC XML, COCO JSON, etc).
    split: 'train', 'valid', or 'test' (default: 'train').
    batch_name: optional batch label for grouping uploads."""
    if split not in ("train", "valid", "test"):
        raise ValueError(f"split must be 'train', 'valid', or 'test' — got '{split}'")
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
    kwargs: dict = {"split": split, "num_retry_uploads": 2}
    if annotation_path:
        kwargs["annotation_path"] = annotation_path
    if batch_name:
        kwargs["batch_name"] = batch_name
    with _suppress_output():
        result = proj.upload(image_path, **kwargs)
    return {"success": True, "image_path": image_path, "result": str(result)}


@mcp.tool()
def create_version(
    project_url: str,
    preprocessing: dict | None = None,
    augmentation: dict | None = None,
    workspace_url: str = "",
) -> dict:
    """Generate a new dataset version with preprocessing and augmentation settings.
    preprocessing example: {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Stretch to"}}
    augmentation example: {"flip": {"horizontal": True}, "rotation": {"degrees": 15}}
    Leave both empty to generate a version with no modifications."""
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
    settings: dict = {}
    if preprocessing:
        settings["preprocessing"] = preprocessing
    if augmentation:
        settings["augmentation"] = augmentation
    with _suppress_output():
        version = proj.generate_version(settings)
    return {"success": True, "version": str(version)}


@mcp.tool()
def download_dataset(
    project_url: str,
    version_number: int,
    model_format: str = "yolov8",
    location: str = "",
    workspace_url: str = "",
) -> dict:
    """Download a dataset version to a local directory.
    Supported formats: yolov8, yolov5, coco, voc, tensorflow, darknet, createml, multiclass.
    Returns the local path where the dataset was saved.
    Leave location empty to download to the current directory."""
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
        ver = proj.version(version_number)
    kwargs: dict = {"model_format": model_format, "overwrite": True}
    if location:
        kwargs["location"] = location
    with _suppress_output():
        dataset = ver.download(**kwargs)
    return {
        "location": getattr(dataset, "location", location),
        "format": model_format,
        "version": version_number,
    }


# ---------------------------------------------------------------------------
# Roboflow Universe tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_universe(query: str, page: int = 1, results_per_page: int = 12) -> list:
    """Search Roboflow Universe for public datasets and pre-trained models.
    Returns matching projects with name, workspace slug, project slug, type, image count, and class list.
    Use the returned workspace_slug and project_slug with download_universe_dataset."""
    api_key = os.environ["ROBOFLOW_API_KEY"]
    resp = requests.get(
        "https://api.roboflow.com/universe/search",
        params={"q": query, "page": page, "page_size": results_per_page, "api_key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        ws = r.get("workspace") or {}
        universe_url = r.get("url", "")
        # derive project slug from universe URL path: /<workspace>/<project>
        project_slug = universe_url.rstrip("/").rsplit("/", 1)[-1] if universe_url else ""
        out.append({
            "name": r.get("name"),
            "workspace_slug": ws.get("url"),
            "project_slug": project_slug,
            "type": r.get("type"),
            "annotation": r.get("annotation"),
            "classes": r.get("classes"),
            "class_count": r.get("classCount"),
            "images": r.get("images"),
            "downloads": r.get("downloads"),
            "stars": r.get("stars"),
            "license": r.get("license"),
            "latest_version": r.get("latestVersion"),
            "universe_url": universe_url,
        })
    return out


@mcp.tool()
def download_universe_dataset(
    universe_workspace: str,
    universe_project: str,
    version_number: int,
    model_format: str = "yolov8",
    location: str = "",
) -> dict:
    """Download a public dataset from Roboflow Universe directly.
    universe_workspace: workspace slug on Universe (e.g. 'brad-dwyer').
    universe_project: project slug (e.g. 'chess-pieces-new').
    version_number: dataset version to download.
    Supported formats: yolov8, yolov5, coco, voc, tensorflow, darknet."""
    rf = get_rf()
    with _suppress_output():
        proj = rf.workspace(universe_workspace).project(universe_project)
        ver = proj.version(version_number)
    kwargs: dict = {"model_format": model_format, "overwrite": True}
    if location:
        kwargs["location"] = location
    with _suppress_output():
        dataset = ver.download(**kwargs)
    return {
        "location": getattr(dataset, "location", location),
        "format": model_format,
        "version": version_number,
        "source": f"https://universe.roboflow.com/{universe_workspace}/{universe_project}",
    }


# ---------------------------------------------------------------------------
# Inference & Metrics tools
# ---------------------------------------------------------------------------


@mcp.tool()
def run_inference(
    project_url: str,
    version_number: int,
    image_path: str,
    confidence: int = 50,
    overlap: int = 50,
    workspace_url: str = "",
) -> dict:
    """Run inference on an image using a deployed Roboflow model.
    image_path: absolute local file path or a public URL.
    confidence: minimum confidence threshold 0-100 (default 50).
    overlap: maximum bounding box overlap 0-100 (default 50).
    Returns predictions with bounding boxes, classes, and confidence scores."""
    ws = get_ws(workspace_url)
    with _suppress_output():
        proj = ws.project(project_url)
        ver = proj.version(version_number)
        model = ver.model
    if model is None:
        raise ValueError(f"No trained model found for version {version_number}. Train a model first.")
    with _suppress_output():
        predictions = model.predict(image_path, confidence=confidence, overlap=overlap)
    return predictions.json()


@mcp.tool()
def get_model_metrics(
    project_url: str,
    version_number: int,
    workspace_url: str = "",
) -> dict:
    """Get model training metrics (mAP, precision, recall) for a trained version.
    Returns performance metrics if the model has been trained on Roboflow."""
    get_rf()  # ensure auth validated before making the request
    api_key = os.environ["ROBOFLOW_API_KEY"]
    ws_url = workspace_url or get_ws().url
    resp = requests.get(
        f"https://api.roboflow.com/{ws_url}/{project_url}/{version_number}",
        params={"api_key": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    version_data = data.get("version", {})
    model_data = version_data.get("model", {})
    return {
        "version": version_number,
        "mAP": model_data.get("map"),
        "precision": model_data.get("precision"),
        "recall": model_data.get("recall"),
        "training_status": version_data.get("training", {}).get("status"),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
