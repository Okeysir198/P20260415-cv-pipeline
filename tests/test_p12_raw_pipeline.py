"""Test p12: Raw Pipeline — raw images → annotate → QA (SAM3) → LS roundtrip →
data-prep merge+split → explore → train → HPO → eval → error analysis → export
→ inference → video inference.

Simulates a real-world scenario: a dataset of unlabeled images arrives, the full
pipeline runs from auto-annotation through LS human-review (simulated via API)
and a p00 merge/split, all the way to a deployable ONNX model.

Service requirements (graceful skip if unavailable — the skill layer at
.claude/plugins/cv-data-prep/skills/cv-pipeline-e2e-test/ enforces services
up-front so every stage runs in practice):
  - Auto-label service @ :18104  (required for auto_annotate stage)
  - QA service @ :18105          (required for annotation_qa stage)
  - SAM3 @ :18100                (used by QA for geometric verification)
  - Label Studio @ :18103        (required for label_studio_roundtrip stage)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Pick the idle GPU before torch/onnxruntime touch CUDA. Safe to run
# after `import torch` because torch defers CUDA init, but we do it
# at the very top for clarity and to support shared-box environments.
from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

import json  # noqa: E402
import shutil  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import torch  # noqa: E402

from _runner import run_all  # noqa: E402
from utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUTS         = Path(__file__).resolve().parent / "outputs" / "16_raw_pipeline"
RAW_DATASET_DIR = ROOT / "outputs" / "test_raw_pipeline"
SOURCE_DATASET  = ROOT / "dataset_store" / "test_fire_100"

DATA_CONFIG_PATH   = ROOT / "configs" / "_test" / "00_raw_pipeline.yaml"
TRAIN_CONFIG_PATH  = ROOT / "configs" / "_test" / "06_training.yaml"
EXPORT_CONFIG_PATH = ROOT / "configs" / "_shared" / "09_export.yaml"

AUTO_LABEL_URL = "http://localhost:18104"
QA_SERVICE_URL = "http://localhost:18105"
SAM3_URL       = "http://localhost:18100"
LS_URL         = "http://localhost:18103"

OUTPUTS.mkdir(parents=True, exist_ok=True)

# Redirect any pipeline stage that calls utils.config.generate_run_dir()
# into this test's OUTPUTS tree, so stages whose configs live in
# configs/_test/ (outside features/<name>/configs/) don't create a
# ghost `features/<something>/runs/…` folder at the project root.
os.environ.setdefault("CV_RUNS_BASE", str(OUTPUTS / "run_dirs"))

# ---------------------------------------------------------------------------
# Cross-test shared state (replaces pytest fixtures for sequential runner)
# ---------------------------------------------------------------------------

_state: dict = {
    "data_config": None,   # loaded from 00_raw_pipeline.yaml
    "ckpt_path": None,     # Path to best/last checkpoint after training
    "onnx_path": None,     # Path to exported ONNX model
    "predictor": None,     # DetectionPredictor (reused across inference tests)
}

# ---------------------------------------------------------------------------
# Service availability (cached)
# ---------------------------------------------------------------------------

_auto_label_cache: bool | None = None
_qa_service_cache: bool | None = None
_ls_service_cache: bool | None = None


def _check_service(url: str) -> bool:
    try:
        import httpx
        resp = httpx.get(f"{url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def has_auto_label_service() -> bool:
    global _auto_label_cache
    if _auto_label_cache is None:
        _auto_label_cache = _check_service(AUTO_LABEL_URL)
    return _auto_label_cache


def has_qa_service() -> bool:
    global _qa_service_cache
    if _qa_service_cache is None:
        _qa_service_cache = _check_service(QA_SERVICE_URL)
    return _qa_service_cache


def has_label_studio() -> bool:
    global _ls_service_cache
    if _ls_service_cache is None:
        _ls_service_cache = _check_service(LS_URL)
    return _ls_service_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_images(split: str) -> list[Path]:
    images_dir = RAW_DATASET_DIR / split / "images"
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def _get_label_files(split: str) -> list[Path]:
    labels_dir = RAW_DATASET_DIR / split / "labels"
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*.txt"))


def _load_model(ckpt_path: Path):
    from core.p06_models import build_model
    config = load_config(str(TRAIN_CONFIG_PATH))
    config["data"] = {"dataset_config": str(DATA_CONFIG_PATH)}
    model = build_model(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_key = (
        "model_state_dict" if "model_state_dict" in ckpt
        else "model" if "model" in ckpt
        else "state_dict"
    )
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Stage 1 — Setup: copy images from test_fire_100, strip labels
# ---------------------------------------------------------------------------

def test_setup_raw_dataset():
    """Copy images from test_fire_100 into RAW_DATASET_DIR; assert no labels exist."""
    # Clean previous run
    if RAW_DATASET_DIR.exists():
        shutil.rmtree(RAW_DATASET_DIR)

    for split, src_split, n in [("train", "train", 15), ("val", "val", 5)]:
        src_images = SOURCE_DATASET / src_split / "images"
        dst_images = RAW_DATASET_DIR / split / "images"
        dst_images.mkdir(parents=True, exist_ok=True)

        candidates = sorted(p for p in src_images.iterdir()
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for img in candidates[:n]:
            shutil.copy2(img, dst_images / img.name)

    # Verify: images exist, NO label files
    train_imgs = _get_images("train")
    val_imgs   = _get_images("val")
    assert len(train_imgs) >= 5, f"Expected >= 5 train images, got {len(train_imgs)}"
    assert len(val_imgs)   >= 1, f"Expected >= 1 val images, got {len(val_imgs)}"

    all_txts = list(RAW_DATASET_DIR.rglob("*.txt"))
    assert len(all_txts) == 0, f"Raw dataset should have no labels, found: {all_txts}"

    # Cache data config
    _state["data_config"] = load_config(str(DATA_CONFIG_PATH))

    print(f"    Train images : {len(train_imgs)}")
    print(f"    Val images   : {len(val_imgs)}")
    print(f"    Labels       : 0 (raw — no annotations)")


# ---------------------------------------------------------------------------
# Stage 2 — Auto-annotate: P01
# ---------------------------------------------------------------------------

def test_auto_annotate_generates_labels():
    """Run P01 auto-annotate (text mode) on raw images; verify YOLO labels written."""
    if not has_auto_label_service():
        print(f"    SKIP: auto-label service not available at {AUTO_LABEL_URL}")
        return

    train_imgs = _get_images("train")
    if not train_imgs:
        print("    SKIP: no train images (setup was skipped)")
        return

    from core.p01_auto_annotate.graph import build_graph

    config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))

    graph = build_graph()
    initial_state = {
        "data_config": config,
        "annotate_config": {
            "auto_label_service": {"url": AUTO_LABEL_URL, "timeout": 120},
            "processing": {
                "batch_size": 5,
                "confidence_threshold": 0.3,
            },
            "nms": {
                "per_class_iou_threshold": 0.5,
                "cross_class_enabled": False,
            },
            "auto_annotate": {
                "output_dir": str(OUTPUTS / "annotate_report"),
                "splits": ["train"],
            },
            "reporting": {"save_previews": False, "preview_count": 0},
        },
        "dataset_name": "test_raw_pipeline",
        "class_names": config["names"],
        "text_prompts": config.get("text_prompts", {}),
        "config_dir": str(DATA_CONFIG_PATH.parent),
        # Avoid p01's feature-name-from-config-dir derivation — our config
        # lives in configs/_test/, not under features/<name>/configs/.
        "output_dir_override": str(OUTPUTS / "p01_runs"),
        "image_paths": {"train": [str(p) for p in train_imgs]},
        "total_images": len(train_imgs),
        "current_batch_idx": 0,
        "total_batches": 1,
        "batch_size": 5,
        "image_results": [],
        "mode": "text",
        "output_format": "bbox",
        "dry_run": False,
        "filter_mode": "all",
    }

    result = graph.invoke(initial_state)

    assert "summary" in result, "Missing 'summary' in graph output"
    summary = result["summary"]

    assert summary.get("total_images", 0) > 0, "No images processed"
    assert summary.get("dry_run") is False, "Expected dry_run=False"

    label_files = _get_label_files("train")
    print(f"    Processed    : {summary['total_images']} images")
    print(f"    Detections   : {summary.get('total_detections', 0)}")
    print(f"    Per class    : {summary.get('per_class_counts', {})}")
    print(f"    Label files  : {len(label_files)}")

    # Labels directory must exist (even if all images had 0 detections, writer runs)
    labels_dir = RAW_DATASET_DIR / "train" / "labels"
    assert labels_dir.exists(), f"Labels directory not created: {labels_dir}"


# ---------------------------------------------------------------------------
# Stage 3 — Annotation QA: P02 with SAM3
# ---------------------------------------------------------------------------

def test_annotation_qa_passes():
    """Run P02 annotation QA with SAM3 verification; verify quality report produced."""
    if not has_qa_service():
        print(f"    SKIP: QA service not available at {QA_SERVICE_URL}")
        return

    label_files = _get_label_files("train")
    if not label_files:
        print("    SKIP: no label files (auto-annotate was skipped or found 0 detections)")
        return

    from core.p02_annotation_qa.pipeline import qa_pipeline

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    qa_config = load_config(str(ROOT / "configs" / "_shared" / "02_annotation_quality.yaml"))

    # Override for small test dataset
    qa_config["sampling"]["sample_size"] = len(label_files)
    qa_config["sampling"]["min_per_class"] = 1
    qa_config["sampling"]["splits"] = ["train"]
    qa_config["processing"]["batch_size"] = 5
    qa_config["sam3"]["service_url"] = SAM3_URL
    qa_config["qa_service"]["url"] = QA_SERVICE_URL

    initial_state = {
        "data_config": data_config,
        "qa_config": qa_config,
        "dataset_name": "test_raw_pipeline",
        "class_names": data_config["names"],
        "splits": ["train"],
        "config_dir": str(DATA_CONFIG_PATH.parent),
        "use_sam3": True,
        "image_results": [],
        "auto_label_config": {},
    }

    final_state = qa_pipeline.invoke(initial_state)

    assert "summary" in final_state, "Missing 'summary' in QA output"
    summary = final_state["summary"]

    assert "total_checked" in summary, "Summary missing 'total_checked'"
    assert summary["total_checked"] > 0, f"QA checked 0 images"
    assert "grades" in summary, "Summary missing 'grades'"
    assert "avg_quality_score" in summary, "Summary missing 'avg_quality_score'"

    grades = summary["grades"]
    avg_score = summary["avg_quality_score"]

    print(f"    Checked      : {summary['total_checked']} images")
    print(f"    Grades       : good={grades.get('good',0)}, "
          f"review={grades.get('review',0)}, bad={grades.get('bad',0)}")
    print(f"    Avg quality  : {avg_score:.3f}")

    # Save QA report
    report_path = OUTPUTS / "qa_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Report saved : {report_path}")


# ---------------------------------------------------------------------------
# Stage 3b — Label Studio roundtrip: P04 (API-driven human-review simulation)
# ---------------------------------------------------------------------------
#
# Imports the pre-annotations into LS, bulk-"accepts" every prediction by
# POSTing it as a new annotation, then exports the reviewed tasks back to
# YOLO, overwriting train/labels/ in-place. Since we accept predictions
# verbatim, this is an identity round-trip (values preserved within 1e-3)
# — its job is to exercise the p04 bridge + LS API surface, not to change
# the labels. Skips gracefully if :18103 is down; in the skill layer,
# services are required up-front so this stage actually runs.

def test_label_studio_roundtrip():
    """Import YOLO pre-annotations → accept via API → export back to YOLO."""
    if not has_label_studio():
        print(f"    SKIP: Label Studio service not available at {LS_URL}")
        return

    label_files = _get_label_files("train")
    if not label_files:
        print("    SKIP: no labels to import (auto-annotate was skipped)")
        return

    import requests

    from core.p04_label_studio.bridge import (
        LabelStudioAPI,
        _ls_url_to_label_path,
        build_task,
        gather_dataset_pairs,
        generate_label_config,
        ls_to_yolo,
        write_yolo_labels,
    )

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    class_names = {int(k): v for k, v in data_config["names"].items()}
    class_name_to_id = {v: k for k, v in class_names.items()}
    project_name = f"{data_config['dataset_name']}_review"

    email = "admin@admin.com"
    password = "admin123"

    # Authenticate — signup-on-miss so the test works on a freshly-bootstrapped
    # LS instance as well. Mirrors the helper in test_p04_label_studio.py.
    def _session() -> requests.Session:
        s = requests.Session()
        s.get(f"{LS_URL}/user/login/", timeout=10)
        csrf = s.cookies.get("csrftoken", "")
        r = s.post(
            f"{LS_URL}/user/login/",
            data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
            headers={"Referer": f"{LS_URL}/user/login/"},
            timeout=10,
        )
        if "/user/login" in r.url:
            s = requests.Session()
            s.get(f"{LS_URL}/user/signup/", timeout=10)
            csrf = s.cookies.get("csrftoken", "")
            s.post(
                f"{LS_URL}/user/signup/",
                data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
                headers={"Referer": f"{LS_URL}/user/signup/"},
                timeout=10,
            )
            s = requests.Session()
            s.get(f"{LS_URL}/user/login/", timeout=10)
            csrf = s.cookies.get("csrftoken", "")
            s.post(
                f"{LS_URL}/user/login/",
                data={"email": email, "password": password, "csrfmiddlewaretoken": csrf},
                headers={"Referer": f"{LS_URL}/user/login/"},
                timeout=10,
            )
        return s

    session = _session()
    api = LabelStudioAPI(url=LS_URL, api_key="unused", email=email, password=password)

    # Upsert: delete stale project of the same name, then create fresh.
    existing = api.find_project(project_name)
    if existing:
        session.delete(f"{LS_URL}/api/projects/{existing['id']}", timeout=10)
    project = api.create_project(
        title=project_name,
        label_config=generate_label_config(class_names),
    )
    project_id = project["id"]

    # Build tasks from the raw-pipeline train split (val has no labels yet —
    # auto-annotate only ran for train in Stage 2).
    ds_config = {
        "path": str(RAW_DATASET_DIR),
        "train": "train/images",
    }
    # gather_dataset_pairs returns (image, label, split) triples.
    pairs = gather_dataset_pairs(ds_config, ["train"])
    assert len(pairs) > 0, "No image/label pairs gathered for LS import"

    tasks = [
        build_task(
            image_path=img,
            label_path=lbl,
            class_names=class_names,
            local_files_root="/datasets",
            dataset_base=RAW_DATASET_DIR,
            model_version="raw_pipeline_v1",
        )
        for img, lbl, _split in pairs
    ]
    imported = api.import_tasks(project_id, tasks)
    assert imported > 0, f"import_tasks returned {imported}"

    # Bulk "human review" — copy each task's prediction into a new annotation.
    fetched = api.get_tasks(project_id)
    annotated = 0
    for t in fetched:
        preds = t.get("predictions", [])
        if not preds:
            continue
        result = preds[0].get("result", [])
        if not result:
            continue
        resp = session.post(
            f"{LS_URL}/api/tasks/{t['id']}/annotations/",
            json={"result": result},
            timeout=10,
        )
        assert resp.status_code in (200, 201), (
            f"POST annotation failed (task {t['id']}): {resp.status_code}"
        )
        annotated += 1
    assert annotated > 0, "No predictions were converted to annotations — p01 output empty?"

    # Export reviewed tasks back into the raw-pipeline train/labels/ dir.
    reviewed = api.get_tasks(project_id, only_reviewed=True)
    exported = 0
    for t in reviewed:
        anns = t.get("annotations", [])
        if not anns:
            continue
        results = anns[-1].get("result", [])
        yolo_anns = []
        for r in results:
            converted = ls_to_yolo(r, class_name_to_id)
            if converted is not None:
                yolo_anns.append(converted)
        url = t.get("data", {}).get("image", "")
        if "/train/" in url:
            out_dir = RAW_DATASET_DIR / "train" / "labels"
        elif "/val/" in url:
            out_dir = RAW_DATASET_DIR / "val" / "labels"
        else:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        label_path = _ls_url_to_label_path(url, out_dir)
        write_yolo_labels(label_path, yolo_anns)
        exported += 1

    # Sanity: train labels still exist and match in count (identity round-trip).
    post_labels = _get_label_files("train")
    assert len(post_labels) > 0, "Train labels were wiped by the LS roundtrip"

    # Cleanup the test project so repeat runs stay idempotent.
    try:
        session.delete(f"{LS_URL}/api/projects/{project_id}", timeout=10)
    except Exception:
        pass

    print(f"    project      : {project_name} (id={project_id})")
    print(f"    imported     : {imported}")
    print(f"    annotated    : {annotated}")
    print(f"    exported     : {exported}")


# ---------------------------------------------------------------------------
# Stage 3c — Data prep merge+split: P00 (CLI subprocess)
# ---------------------------------------------------------------------------
#
# After LS export, run the canonical p00 CLI to merge the now-reviewed
# raw-pipeline train split into a fresh training_ready/ output with a
# stratified 70/20/10 split. This writes to a separate location and does
# NOT disturb _state["data_config"] — downstream stages keep using the
# original 00_raw_pipeline.yaml so the checkpoint dependency chain (p06 →
# p08/p09/p10) stays intact. What's being exercised here is the p00 code
# path (parsers, class mapper, splitter, file ops) end-to-end via the CLI
# that users actually invoke.

def test_data_prep_merges_and_splits():
    """Run core/p00_data_prep/run.py as a subprocess on the auto-annotated train split."""
    import subprocess
    import tempfile

    import yaml

    label_files = _get_label_files("train")
    if not label_files:
        print("    SKIP: no labels — p01 was skipped")
        return

    merged_output = OUTPUTS / "p00_merged"
    if merged_output.exists():
        shutil.rmtree(merged_output)

    # p00's run.py resolves config-relative paths (not project-root-relative),
    # so when the temp YAML lives in /tmp/, relative paths resolve to /tmp/...
    # Use absolute paths to side-step the convention.
    abs_source = str(RAW_DATASET_DIR / "train")
    abs_output = str(merged_output)
    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    canonical = [data_config["names"][k] for k in sorted(data_config["names"])]

    prep_config = {
        "task": "detection",
        "dataset_name": "test_raw_pipeline_merged",
        "output_dir": abs_output,
        "output_format": "yolo",
        "classes": canonical,
        "sources": [
            {
                "name": "raw_pipeline_train",
                "path": abs_source,
                "format": "yolo",
                "has_splits": False,
                # YOLO labels hold raw class IDs ("0","1",…); without a
                # data.yaml next to the images the parser can't resolve
                # them to names, and ClassMapper (keyed on class names)
                # would drop every sample. Feed the canonical list in
                # via source_classes so index→name happens at parse time.
                "source_classes": canonical,
                "class_map": {c: c for c in canonical},
            }
        ],
        "splits": {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42},
        "options": {
            "copy_images": True,
            "handle_duplicates": "rename",
            "validate_labels": True,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(prep_config, f)
        cfg_path = f.name

    run_py = ROOT / "core" / "p00_data_prep" / "run.py"

    # Dry-run — schema + path check first (fast).
    dry = subprocess.run(
        [sys.executable, str(run_py), "--config", cfg_path, "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert dry.returncode == 0, (
        f"p00 dry-run failed (rc={dry.returncode}):\nSTDOUT:\n{dry.stdout}\nSTDERR:\n{dry.stderr}"
    )

    # Real run.
    real = subprocess.run(
        [sys.executable, str(run_py), "--config", cfg_path],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    assert real.returncode == 0, (
        f"p00 run failed (rc={real.returncode}):\nSTDOUT:\n{real.stdout}\nSTDERR:\n{real.stderr}"
    )

    for split in ("train", "val", "test"):
        assert (merged_output / split / "images").exists(), (
            f"p00 did not create {split}/images"
        )
        assert (merged_output / split / "labels").exists(), (
            f"p00 did not create {split}/labels"
        )

    splits_file = merged_output / "splits.json"
    assert splits_file.exists(), f"p00 did not write splits.json at {splits_file}"

    split_counts = {
        s: len(list((merged_output / s / "images").iterdir()))
        for s in ("train", "val", "test")
    }
    total = sum(split_counts.values())
    assert total > 0, f"p00 output has 0 images across splits: {split_counts}"

    print(f"    output       : {merged_output}")
    print(f"    split counts : {split_counts}")
    print(f"    splits.json  : {splits_file}")


# ---------------------------------------------------------------------------
# Stage 4 — Data Exploration: P05
# ---------------------------------------------------------------------------

def test_data_exploration():
    """Run dataset exploration on annotated data; verify distribution stats."""
    from utils.exploration import (
        explore,
        compute_class_distribution,
        compute_image_stats,
        compute_annotation_stats,
        resolve_split_dir,
    )

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))

    # Run full explore (prints to stdout)
    explore(str(DATA_CONFIG_PATH))

    # Also gather structured stats for assertions
    class_names = {int(k): v for k, v in data_config["names"].items()}
    config_dir = DATA_CONFIG_PATH.parent
    stats = {}
    for split in ["train", "val"]:
        split_dir = resolve_split_dir(data_config, split, config_dir)
        if split_dir is None:
            continue
        if not split_dir.exists() or not any(split_dir.iterdir()):
            continue

        img_stats  = compute_image_stats(split_dir)
        ann_stats  = compute_annotation_stats(split_dir, class_names)
        class_dist = compute_class_distribution(split_dir, class_names)
        stats[split] = {
            "image_stats": img_stats,
            "annotation_stats": ann_stats,
            "class_distribution": class_dist,
        }

    assert len(stats) > 0, "No splits found for exploration"
    for split, s in stats.items():
        assert "count" in s["image_stats"], f"{split}: missing image count"
        assert "avg_objects_per_image" in s["annotation_stats"], (
            f"{split}: missing avg_objects_per_image"
        )

    # Save (convert any tuple keys to strings for JSON compatibility)
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    exploration_path = OUTPUTS / "exploration.json"
    with open(exploration_path, "w") as f:
        json.dump(_json_safe(stats), f, indent=2, default=str)
    print(f"    Exploration saved: {exploration_path}")


# ---------------------------------------------------------------------------
# Stage 5 — Dataset loading: P05
# ---------------------------------------------------------------------------

def test_detection_dataset_loads():
    """Load P05 DetectionDataset from annotated data; verify shape and targets."""
    from core.p05_data.detection_dataset import YOLOXDataset

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    data_config["_config_dir"] = DATA_CONFIG_PATH.parent

    dataset = YOLOXDataset(data_config, split="train", base_dir=DATA_CONFIG_PATH.parent)
    assert len(dataset) > 0, f"Dataset is empty (0 samples)"

    item = dataset[0]
    # YOLOXDataset returns (image_tensor, targets_tensor) or similar
    assert item is not None, "dataset[0] returned None"

    print(f"    Dataset size : {len(dataset)} images")
    print(f"    Item type    : {type(item)}")


# ---------------------------------------------------------------------------
# Stage 6 — Training: P06
# ---------------------------------------------------------------------------

def test_training_runs_and_loss_decreases():
    """Train YOLOX-M for 2 epochs on annotated raw data; verify checkpoint saved."""
    from core.p06_training.trainer import DetectionTrainer

    label_files = _get_label_files("train")
    if not label_files:
        print("    SKIP: no label files — auto-annotate was skipped")
        return

    save_dir = str(OUTPUTS / "runs")

    trainer = DetectionTrainer(
        config_path=str(TRAIN_CONFIG_PATH),
        overrides={
            "data": {"dataset_config": str(DATA_CONFIG_PATH)},
            "training": {"epochs": 2, "patience": 10},
            "logging": {"save_dir": save_dir, "wandb_project": None},
        },
    )

    metrics = trainer.train()
    assert isinstance(metrics, dict), f"train() returned {type(metrics)}"

    # Find checkpoint
    ckpt_files = []
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = Path(save_dir) / name
        if p.exists():
            ckpt_files.append(p)
            break
    # Also search recursively in case save_dir has sub-folders
    if not ckpt_files:
        ckpt_files = (
            list(Path(save_dir).rglob("best.pth")) +
            list(Path(save_dir).rglob("best.pt")) +
            list(Path(save_dir).rglob("last.pth")) +
            list(Path(save_dir).rglob("last.pt"))
        )

    assert len(ckpt_files) > 0, f"No checkpoint found under {save_dir}"
    _state["ckpt_path"] = ckpt_files[0]

    # Optional: verify loss decreases (only if trainer exposes loss_history)
    if "loss_history" in metrics and len(metrics["loss_history"]) >= 2:
        hist = metrics["loss_history"]
        assert hist[-1] < hist[0], (
            f"Loss did not decrease: epoch0={hist[0]:.4f}, epoch{len(hist)-1}={hist[-1]:.4f}"
        )
        print(f"    Loss history : epoch0={hist[0]:.4f} → epoch{len(hist)-1}={hist[-1]:.4f} ✓")
    else:
        print(f"    Metrics keys : {list(metrics.keys())}")

    print(f"    Checkpoint   : {_state['ckpt_path']}")


# ---------------------------------------------------------------------------
# Stage 6b — HPO: P07 (2 trials × 1 epoch, same dataset as Stage 6)
# ---------------------------------------------------------------------------
#
# Tiny HPO smoke — proves the Optuna wiring and search-space machinery works
# against the raw-pipeline dataset. 2 trials × 1 epoch keeps total runtime
# close to one normal training run. Uses HPOOptimizer directly (same pattern
# as test_p07_hpo.py) rather than the CLI so we don't pay the interpreter
# startup cost twice.

def test_hpo_runs():
    """Run 2 Optuna trials at 1 epoch each on the raw-pipeline dataset."""
    if _state["ckpt_path"] is None:
        print("    SKIP: no checkpoint — training was skipped")
        return

    hpo_config_path = ROOT / "configs" / "_shared" / "08_hyperparameter_tuning.yaml"
    if not hpo_config_path.exists():
        print(f"    SKIP: HPO config missing at {hpo_config_path}")
        return

    from core.p07_hpo.optimizer import HPOOptimizer

    optimizer = HPOOptimizer(
        training_config_path=str(TRAIN_CONFIG_PATH),
        hpo_config_path=str(hpo_config_path),
        training_overrides={
            # Point HPO trials at the raw-pipeline dataset.
            "data": {"dataset_config": str(DATA_CONFIG_PATH)},
            "training": {"epochs": 1, "patience": 10},
            "logging": {
                "save_dir": str(OUTPUTS / "hpo_trials"),
                "wandb_project": None,
            },
        },
    )

    study = optimizer.optimize(n_trials=2)
    assert study is not None, "optimize() returned None"
    assert len(study.trials) == 2, f"Expected 2 trials, got {len(study.trials)}"

    best = study.best_trial
    summary = {
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
    }
    summary_path = OUTPUTS / "hpo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"    Trials       : {len(study.trials)}")
    print(f"    Best trial   : #{best.number}, value={best.value:.4f}")
    print(f"    Best params  : {best.params}")
    print(f"    Summary      : {summary_path}")


# ---------------------------------------------------------------------------
# Stage 7 — Evaluation: P08
# ---------------------------------------------------------------------------

def test_evaluation_runs():
    """Evaluate checkpoint on val split; verify mAP dict returned."""
    if _state["ckpt_path"] is None:
        print("    SKIP: no checkpoint (training was skipped)")
        return

    from core.p08_evaluation.evaluator import ModelEvaluator

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    data_config["_config_dir"] = DATA_CONFIG_PATH.parent

    model = _load_model(_state["ckpt_path"])

    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        conf_threshold=0.01,
        iou_threshold=0.5,
        batch_size=4,
        num_workers=0,
    )
    results = evaluator.evaluate(split="val")
    assert isinstance(results, dict), f"evaluate() returned {type(results)}"

    serializable = {}
    for k, v in results.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        elif hasattr(v, "item"):
            serializable[k] = v.item()
        else:
            serializable[k] = v

    metrics_path = OUTPUTS / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    if "mAP" in results:
        print(f"    mAP@0.5      : {results['mAP']:.4f}")
    if "per_class_ap" in results:
        print(f"    Per-class AP : {results['per_class_ap']}")
    print(f"    Metrics saved: {metrics_path}")


# ---------------------------------------------------------------------------
# Stage 8 — Error Analysis: P08
# ---------------------------------------------------------------------------

def test_error_analysis():
    """Run error analysis on val predictions; verify error breakdown report."""
    if _state["ckpt_path"] is None:
        print("    SKIP: no checkpoint (training was skipped)")
        return

    from core.p08_evaluation.error_analysis import ErrorAnalyzer
    from core.p10_inference.predictor import DetectionPredictor

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))
    data_config["_config_dir"] = DATA_CONFIG_PATH.parent

    # Use PyTorch checkpoint for error analysis (no ONNX needed yet)
    predictor = DetectionPredictor(
        model_path=str(_state["ckpt_path"]),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )

    val_images = _get_images("val")
    if not val_images:
        print("    SKIP: no val images")
        return

    class_names = {int(k): v for k, v in data_config["names"].items()}
    analyzer = ErrorAnalyzer(class_names=class_names, iou_threshold=0.5)

    predictions = []
    ground_truths = []
    for img_path in val_images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        preds = predictor.predict(image)
        predictions.append(preds)

        # Load GT from label file
        label_path = RAW_DATASET_DIR / "val" / "labels" / (img_path.stem + ".txt")
        boxes, labels = [], []
        if label_path.exists():
            h, w = image.shape[:2]
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
        ground_truths.append({
            "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
            "labels": np.array(labels, dtype=np.int64),
        })

    report = analyzer.analyze(predictions, ground_truths)
    assert hasattr(report, "summary"), "ErrorReport missing summary"
    assert hasattr(report, "errors"), "ErrorReport missing errors"

    summary = report.summary
    report_path = OUTPUTS / "error_analysis.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"    Total errors : {len(report.errors)}")
    print(f"    Summary      : {summary}")
    print(f"    Report saved : {report_path}")


# ---------------------------------------------------------------------------
# Stage 9 — Export to ONNX: P09
# ---------------------------------------------------------------------------

def test_export_to_onnx():
    """Export checkpoint to ONNX and validate with onnx.checker."""
    if _state["ckpt_path"] is None:
        print("    SKIP: no checkpoint (training was skipped)")
        return

    from core.p09_export.exporter import ModelExporter

    model = _load_model(_state["ckpt_path"])
    model.cpu()

    export_config = load_config(str(EXPORT_CONFIG_PATH))
    export_config["output_dir"] = str(OUTPUTS)

    exporter = ModelExporter(model, export_config, model_name="raw_pipeline_fire")
    onnx_path = exporter.export_onnx(save_path=str(OUTPUTS / "raw_pipeline.onnx"))

    assert Path(onnx_path).exists(), f"ONNX file not found: {onnx_path}"

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    _state["onnx_path"] = onnx_path
    print(f"    ONNX path    : {onnx_path}")
    print(f"    File size    : {Path(onnx_path).stat().st_size / 1e6:.1f} MB")
    print(f"    Checker      : passed")


# ---------------------------------------------------------------------------
# Stage 10 — ONNX Inference: P10
# ---------------------------------------------------------------------------

def test_onnx_inference():
    """Run DetectionPredictor on val images with ONNX model; verify output keys."""
    if _state["onnx_path"] is None:
        print("    SKIP: no ONNX model (export was skipped)")
        return

    from core.p10_inference.predictor import DetectionPredictor

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))

    predictor = DetectionPredictor(
        model_path=str(_state["onnx_path"]),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )
    _state["predictor"] = predictor

    val_images = _get_images("val")
    assert len(val_images) > 0, "No val images for inference"

    for i, img_path in enumerate(val_images[:3]):
        image = cv2.imread(str(img_path))
        assert image is not None, f"Failed to read {img_path}"

        preds = predictor.predict(image)
        assert isinstance(preds, dict), f"predict() returned {type(preds)}"
        assert "boxes" in preds, "Missing 'boxes'"
        assert "scores" in preds, "Missing 'scores'"
        assert "labels" in preds, "Missing 'labels'"

        n_dets = len(preds["boxes"])
        print(f"    Image {i}: {img_path.name} — {n_dets} detections")

    # Save one visualization
    image = cv2.imread(str(val_images[0]))
    vis = predictor.visualize(image, predictor.predict(image))
    vis_path = OUTPUTS / "inference_result.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"    Visualization: {vis_path}")


# ---------------------------------------------------------------------------
# Stage 11 — Video Inference: P10
# ---------------------------------------------------------------------------

def test_video_inference():
    """Process 3 val frames through VideoProcessor; verify frame counter and output shape."""
    if _state["onnx_path"] is None:
        print("    SKIP: no ONNX model (export was skipped)")
        return

    from core.p10_inference.predictor import DetectionPredictor
    from core.p10_inference.video_inference import VideoProcessor

    data_config = _state["data_config"] or load_config(str(DATA_CONFIG_PATH))

    predictor = _state.get("predictor") or DetectionPredictor(
        model_path=str(_state["onnx_path"]),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )

    processor = VideoProcessor(predictor=predictor)
    val_images = _get_images("val")
    assert len(val_images) > 0, "No val images for video inference"

    frames = val_images[:min(3, len(val_images))]
    for frame_idx, img_path in enumerate(frames):
        image = cv2.imread(str(img_path))
        assert image is not None, f"Failed to read {img_path}"

        result = processor.process_frame(image, frame_idx=frame_idx)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"process_frame() should return (annotated, detections, alerts), got {type(result)}"
        )
        annotated, detections, alerts = result
        assert isinstance(annotated, np.ndarray), "annotated is not ndarray"
        assert annotated.shape == image.shape, (
            f"annotated shape {annotated.shape} != input {image.shape}"
        )
        n_dets = len(detections.get("boxes", []))
        print(f"    Frame {frame_idx}: {img_path.name} — {n_dets} dets, {len(alerts)} alerts")

    assert processor._frame_count == len(frames), (
        f"Expected _frame_count={len(frames)}, got {processor._frame_count}"
    )
    print(f"    Frame count  : {processor._frame_count} ✓")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all([
        ("setup_raw_dataset",              test_setup_raw_dataset),
        ("auto_annotate_generates_labels", test_auto_annotate_generates_labels),
        ("annotation_qa_passes",           test_annotation_qa_passes),
        ("label_studio_roundtrip",         test_label_studio_roundtrip),
        ("data_prep_merges_and_splits",    test_data_prep_merges_and_splits),
        ("data_exploration",               test_data_exploration),
        ("detection_dataset_loads",        test_detection_dataset_loads),
        ("training_runs",                  test_training_runs_and_loss_decreases),
        ("hpo_runs",                       test_hpo_runs),
        ("evaluation_runs",                test_evaluation_runs),
        ("error_analysis",                 test_error_analysis),
        ("export_to_onnx",                 test_export_to_onnx),
        ("onnx_inference",                 test_onnx_inference),
        ("video_inference",                test_video_inference),
    ], title="Test p12: Raw Pipeline (Raw Images → Trained Model)")
