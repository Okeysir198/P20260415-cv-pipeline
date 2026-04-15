"""Test 12: Annotation QA — test QA pipeline components (CPU + service-based)."""

import sys
import json
import tempfile
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.config import load_config, resolve_path
from core.p02_annotation_qa.sampler import StratifiedSampler
from core.p02_annotation_qa.scorer import QualityScorer
from core.p02_annotation_qa.reporter import QAReporter


def _check_service(url: str) -> bool:
    """Check if a service is reachable."""
    try:
        resp = httpx.get(f"{url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


_sam3_service_cache: bool | None = None
_qa_service_cache: bool | None = None


def has_sam3_service() -> bool:
    global _sam3_service_cache
    if _sam3_service_cache is None:
        _sam3_service_cache = _check_service("http://localhost:18100")
    return _sam3_service_cache


def has_qa_service() -> bool:
    global _qa_service_cache
    if _qa_service_cache is None:
        _qa_service_cache = _check_service("http://localhost:18105")
    return _qa_service_cache

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "12_annotation_qa"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


def test_stratified_sampler():
    """Create StratifiedSampler and verify it returns paths for train/val splits."""
    qa_config = {
        "sampling": {
            "sample_size": 20,
            "strategy": "stratified",
            "min_per_class": 2,
            "seed": 42,
            "splits": ["train", "val"],
        }
    }
    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent

    sampler = StratifiedSampler(qa_config, data_config, config_dir)
    result = sampler.sample()

    assert isinstance(result, dict), f"sample() returned {type(result)}, expected dict"
    assert "train" in result, "Missing 'train' key in sample result"
    assert "val" in result, "Missing 'val' key in sample result"
    assert isinstance(result["train"], list), "train value is not a list"
    assert isinstance(result["val"], list), "val value is not a list"

    total = len(result["train"]) + len(result["val"])
    assert total > 0, "No paths returned from sampler"

    # Verify paths are Path objects
    for split in ["train", "val"]:
        for p in result[split]:
            assert isinstance(p, Path), f"Expected Path, got {type(p)}"

    print(f"    Sampled: train={len(result['train'])}, val={len(result['val'])}, total={total}")


def test_quality_scorer_perfect():
    """Score a perfect image result and verify high score + good grade."""
    config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }

    scorer = QualityScorer(config)
    image_result = {
        "annotations": [(0, 0.5, 0.5, 0.2, 0.3)],
        "validation_issues": [],
        "sam3_verification": {},
    }

    scored = scorer.score_image(image_result)

    assert "quality_score" in scored, "Missing quality_score in result"
    assert "grade" in scored, "Missing grade in result"
    assert scored["quality_score"] >= 0.9, (
        f"Perfect result should score >= 0.9, got {scored['quality_score']}"
    )
    assert scored["grade"] == "good", f"Expected grade 'good', got '{scored['grade']}'"
    print(f"    Perfect score: {scored['quality_score']:.3f}, grade: {scored['grade']}")


def test_quality_scorer_issues():
    """Score a result with validation issues and verify lower score."""
    config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }

    scorer = QualityScorer(config)
    image_result = {
        "annotations": [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.3, 0.3, 0.1, 0.1)],
        "validation_issues": [
            {"type": "out_of_bounds", "annotation_idx": 0},
            {"type": "tiny_bbox", "annotation_idx": 1},
            {"type": "overlap", "annotation_idx": 0},
        ],
        "sam3_verification": {},
    }

    scored = scorer.score_image(image_result)

    assert "quality_score" in scored, "Missing quality_score in result"
    assert scored["quality_score"] < 0.9, (
        f"Result with issues should score < 0.9, got {scored['quality_score']}"
    )
    print(f"    Issues score: {scored['quality_score']:.3f}, grade: {scored['grade']}")


def test_generate_fixes():
    """Generate fixes for out-of-bounds annotation and verify clip_bbox fix."""
    config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }

    scorer = QualityScorer(config)
    image_result = {
        "annotations": [(0, 1.1, 0.5, 0.2, 0.3)],
        "validation_issues": [
            {"type": "out_of_bounds", "annotation_idx": 0},
        ],
        "sam3_verification": {},
    }

    fixes = scorer.generate_fixes(image_result)

    assert isinstance(fixes, list), f"generate_fixes returned {type(fixes)}, expected list"
    assert len(fixes) >= 1, "Expected at least one fix"

    fix_types = [f.get("type") for f in fixes]
    assert "clip_bbox" in fix_types, f"Expected 'clip_bbox' fix, got types: {fix_types}"
    print(f"    Generated {len(fixes)} fix(es): {fix_types}")


def test_reporter_generate():
    """Generate QA report and verify output files exist."""
    tmpdir = tempfile.mkdtemp(prefix="qa_report_test_")
    reporter_config = {
        "worst_count": 2,
        "save_fixes": True,
        "save_visualizations": False,
    }

    reporter = QAReporter(
        output_dir=tmpdir,
        dataset_name="test",
        config=reporter_config,
    )

    image_results = [
        {
            "image_path": "/tmp/fake_image.jpg",
            "quality_score": 0.95,
            "grade": "good",
            "validation_issues": [],
            "suggested_fixes": [],
        },
        {
            "image_path": "/tmp/fake_image2.jpg",
            "quality_score": 0.55,
            "grade": "review",
            "validation_issues": [{"type": "out_of_bounds", "annotation_idx": 0}],
            "suggested_fixes": [{"type": "clip_bbox", "annotation_idx": 0}],
        },
    ]
    summary = {
        "total_images": 2,
        "good": 1,
        "review": 1,
        "reject": 0,
        "mean_score": 0.75,
    }

    report_dir = reporter.generate_report(image_results, summary)

    report_path = Path(report_dir)
    assert report_path.exists(), f"Report dir not found: {report_path}"

    # Check for report.json and summary.txt
    report_json = report_path / "report.json"
    summary_txt = report_path / "summary.txt"
    assert report_json.exists(), f"Missing report.json in {report_path}"
    assert summary_txt.exists(), f"Missing summary.txt in {report_path}"

    print(f"    Report dir: {report_path}")
    print(f"    report.json: {report_json.stat().st_size} bytes")
    print(f"    summary.txt: {summary_txt.stat().st_size} bytes")


def test_build_qa_pipeline_import():
    """Import and verify the QA functional API pipeline has invoke method."""
    from core.p02_annotation_qa.pipeline import qa_pipeline

    assert qa_pipeline is not None, "qa_pipeline import returned None"
    assert hasattr(qa_pipeline, "invoke"), "qa_pipeline has no 'invoke' method"
    print(f"    qa_pipeline: {type(qa_pipeline).__name__}, has invoke: True")


def test_sampler_min_per_class():
    """Test StratifiedSampler with min_per_class > 1, verify each class is represented."""
    from utils.yolo_io import image_to_label_path, parse_yolo_label

    qa_config = {
        "sampling": {
            "sample_size": 30,
            "strategy": "stratified",
            "min_per_class": 3,
            "seed": 123,
            "splits": ["train"],
        }
    }
    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent

    sampler = StratifiedSampler(qa_config, data_config, config_dir)
    result = sampler.sample()

    assert "train" in result, "Missing 'train' key"
    assert len(result["train"]) > 0, "No images sampled"

    # Check each class is represented by counting classes across sampled images
    class_counts = {}
    for img_path in result["train"]:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)
        for ann in annotations:
            cls_id = ann[0]
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    num_classes = int(data_config["num_classes"])
    print(f"    Sampled {len(result['train'])} images, class counts: {class_counts}")

    # Each class present in the dataset should have at least min_per_class images
    for cls_id in range(num_classes):
        if cls_id in class_counts:
            assert class_counts[cls_id] >= 3, (
                f"Class {cls_id} has only {class_counts[cls_id]} images, expected >= 3"
            )


def test_scorer_edge_cases():
    """Test QualityScorer with edge cases: empty annotations, perfect bbox, tiny bbox."""
    config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }
    scorer = QualityScorer(config)

    # Case 1: Empty annotations — should get low score due to structural penalty
    # Use annotation_idx=-1 (not None) since scorer._get_annotation expects int
    empty_result = {
        "annotations": [],
        "validation_issues": [
            {"type": "empty_label", "severity": "medium", "annotation_idx": -1},
        ],
        "sam3_verification": {},
    }
    scored_empty = scorer.score_image(empty_result)
    assert scored_empty["quality_score"] < 0.85, (
        f"Empty annotations should score < 0.85, got {scored_empty['quality_score']}"
    )
    print(f"    Empty: score={scored_empty['quality_score']:.3f}, grade={scored_empty['grade']}")

    # Case 2: Single annotation with perfect bbox — should get high score
    perfect_result = {
        "annotations": [(0, 0.5, 0.5, 0.3, 0.4)],
        "validation_issues": [],
        "sam3_verification": {},
    }
    scored_perfect = scorer.score_image(perfect_result)
    assert scored_perfect["quality_score"] >= 0.9, (
        f"Perfect bbox should score >= 0.9, got {scored_perfect['quality_score']}"
    )
    assert scored_perfect["grade"] == "good", (
        f"Perfect bbox grade should be 'good', got '{scored_perfect['grade']}'"
    )
    print(f"    Perfect: score={scored_perfect['quality_score']:.3f}, grade={scored_perfect['grade']}")

    # Case 3: Annotations with tiny bbox (w or h < 0.01) — should flag as issue
    tiny_result = {
        "annotations": [(0, 0.5, 0.5, 0.005, 0.005)],
        "validation_issues": [
            {"type": "degenerate_box", "severity": "medium", "annotation_idx": 0,
             "detail": "Box too small: w=0.0050, h=0.0050"},
        ],
        "sam3_verification": {},
    }
    scored_tiny = scorer.score_image(tiny_result)
    assert scored_tiny["quality_score"] < 1.0, (
        f"Tiny bbox should score < 1.0, got {scored_tiny['quality_score']}"
    )
    # Should generate a remove_degenerate fix
    fix_types = [f["type"] for f in scored_tiny.get("suggested_fixes", [])]
    assert "remove_degenerate" in fix_types, (
        f"Expected 'remove_degenerate' fix for tiny bbox, got: {fix_types}"
    )
    print(f"    Tiny: score={scored_tiny['quality_score']:.3f}, fixes={fix_types}")


def test_scorer_all_penalties():
    """Create image_result dicts that trigger each penalty type."""
    config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }
    scorer = QualityScorer(config)

    # Penalty 1: structural — out of bounds
    oob_result = {
        "annotations": [(0, 1.1, 0.5, 0.2, 0.3)],
        "validation_issues": [
            {"type": "out_of_bounds", "severity": "high", "annotation_idx": 0,
             "detail": "cx=1.1000 outside [0, 1]"},
        ],
        "sam3_verification": {},
    }
    scored_oob = scorer.score_image(oob_result)
    assert scored_oob["quality_score"] < 1.0, "Out-of-bounds should reduce score"
    fix_types = [f["type"] for f in scored_oob.get("suggested_fixes", [])]
    assert "clip_bbox" in fix_types, f"Expected clip_bbox fix, got {fix_types}"
    print(f"    OOB: score={scored_oob['quality_score']:.3f}, fixes={fix_types}")

    # Penalty 2: bbox_quality — low SAM3 IoU (bbox_quality penalty)
    low_iou_result = {
        "annotations": [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.3, 0.3, 0.1, 0.1)],
        "validation_issues": [],
        "sam3_verification": {
            "box_ious": [0.3, 0.2],
            "misclassified": [],
            "missing_masks": [],
        },
    }
    scored_low_iou = scorer.score_image(low_iou_result)
    assert scored_low_iou["quality_score"] < 0.85, (
        f"Low IoU should reduce score below good threshold, got {scored_low_iou['quality_score']}"
    )
    print(f"    Low IoU: score={scored_low_iou['quality_score']:.3f}")

    # Penalty 3: classification — misclassified annotations
    misclass_result = {
        "annotations": [(0, 0.5, 0.5, 0.2, 0.3), (1, 0.3, 0.3, 0.1, 0.1)],
        "validation_issues": [],
        "sam3_verification": {
            "box_ious": [0.9, 0.9],
            "misclassified": [0, 1],
            "missing_masks": [],
        },
    }
    scored_misclass = scorer.score_image(misclass_result)
    assert scored_misclass["quality_score"] < 0.85, (
        f"Misclassified should reduce score, got {scored_misclass['quality_score']}"
    )
    fix_types = [f["type"] for f in scored_misclass.get("suggested_fixes", [])]
    assert "remove_annotation" in fix_types, (
        f"Expected remove_annotation fix for misclassified, got {fix_types}"
    )
    print(f"    Misclass: score={scored_misclass['quality_score']:.3f}, fixes={fix_types}")

    # Penalty 4: coverage — missing detections (SAM3 found unannotated objects)
    missing_result = {
        "annotations": [(0, 0.5, 0.5, 0.2, 0.3)],
        "validation_issues": [],
        "sam3_verification": {
            "box_ious": [0.95],
            "misclassified": [],
            "missing_masks": [
                {"bbox": (0.1, 0.1, 0.15, 0.15), "area": 0.0225},
                {"bbox": (0.7, 0.7, 0.1, 0.1), "area": 0.01},
                {"bbox": (0.3, 0.8, 0.12, 0.12), "area": 0.0144},
            ],
        },
    }
    scored_missing = scorer.score_image(missing_result)
    assert scored_missing["quality_score"] < 1.0, (
        f"Missing detections should reduce score, got {scored_missing['quality_score']}"
    )
    print(f"    Missing: score={scored_missing['quality_score']:.3f}")


def test_reporter_with_real_results():
    """Run StratifiedSampler on real data, create scored results, pass to QAReporter."""
    from utils.yolo_io import image_to_label_path, parse_yolo_label

    tmpdir = tempfile.mkdtemp(prefix="qa_real_report_")

    # Sample real data
    qa_config = {
        "sampling": {
            "sample_size": 10,
            "strategy": "stratified",
            "min_per_class": 1,
            "seed": 42,
            "splits": ["train"],
        }
    }
    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent

    sampler = StratifiedSampler(qa_config, data_config, config_dir)
    sampled = sampler.sample()

    assert "train" in sampled and len(sampled["train"]) > 0, "No images sampled"

    # Score each sampled image with structural validation only
    scorer_config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }
    scorer = QualityScorer(scorer_config)

    num_classes = int(data_config["num_classes"])
    image_results = []
    for img_path in sampled["train"]:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)
        # Structural validation: check coords in [0,1], valid class IDs, positive w/h
        issues = []
        for idx, ann in enumerate(annotations):
            cls_id, cx, cy, w, h = ann
            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0.0 or val > 1.0:
                    issues.append({"type": "out_of_bounds", "severity": "high",
                                   "annotation_idx": idx, "detail": f"{name}={val}"})
            if cls_id < 0 or cls_id >= num_classes:
                issues.append({"type": "invalid_class", "severity": "high",
                               "annotation_idx": idx, "detail": f"cls_id={cls_id}"})
            if w <= 0 or h <= 0:
                issues.append({"type": "degenerate_box", "severity": "medium",
                               "annotation_idx": idx, "detail": f"w={w}, h={h}"})

        result = {
            "image_path": str(img_path),
            "split": "train",
            "annotations": annotations,
            "validation_issues": issues,
            "sam3_verification": {},
        }
        scorer.score_image(result)
        image_results.append(result)

    # Build summary
    grade_counts = {"good": 0, "review": 0, "bad": 0}
    for r in image_results:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
    scores = [r["quality_score"] for r in image_results]
    summary = {
        "total_checked": len(image_results),
        "grades": grade_counts,
        "avg_quality_score": round(sum(scores) / len(scores), 4),
        "issue_breakdown": {},
        "per_class_stats": {},
        "worst_images": [],
        "auto_fixable_count": 0,
    }

    # Generate report
    reporter = QAReporter(
        output_dir=tmpdir,
        dataset_name="test_fire_real",
        config={"worst_count": 3, "save_fixes": True, "save_visualizations": False},
    )
    report_dir = reporter.generate_report(image_results, summary)
    report_path = Path(report_dir)

    # Verify report.json
    report_json = report_path / "report.json"
    assert report_json.exists(), f"Missing report.json in {report_path}"
    report_data = json.loads(report_json.read_text())
    assert "metadata" in report_data, "report.json missing 'metadata' key"
    assert "summary" in report_data, "report.json missing 'summary' key"
    assert "image_results" in report_data, "report.json missing 'image_results' key"
    assert report_data["metadata"]["total_images"] == len(image_results), (
        f"Expected {len(image_results)} images in report, got {report_data['metadata']['total_images']}"
    )

    # Verify summary.txt
    summary_txt = report_path / "summary.txt"
    assert summary_txt.exists(), f"Missing summary.txt in {report_path}"
    txt_content = summary_txt.read_text()
    assert len(txt_content) > 0, "summary.txt is empty"

    print(f"    Report dir: {report_path}")
    print(f"    Images scored: {len(image_results)}, report.json keys: {list(report_data.keys())}")
    print(f"    summary.txt: {len(txt_content)} chars")


def test_structural_validation_on_real_labels():
    """For 10 real images, read YOLO labels, do structural validation, and score them."""
    from utils.yolo_io import image_to_label_path, parse_yolo_label
    from utils.exploration import get_image_paths

    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent

    # Resolve the train images directory
    from utils.config import resolve_path
    base_path = resolve_path(data_config["path"], config_dir)
    images_dir = base_path / data_config["train"]
    all_images = get_image_paths(images_dir)
    assert len(all_images) > 0, f"No images found in {images_dir}"

    # Take up to 10 images
    test_images = sorted(all_images)[:10]
    num_classes = int(data_config["num_classes"])

    scorer_config = {
        "scoring": {
            "weights": {
                "structural": 0.3,
                "bbox_quality": 0.3,
                "classification": 0.2,
                "coverage": 0.2,
            },
            "thresholds": {
                "good": 0.85,
                "review": 0.60,
            },
        }
    }
    scorer = QualityScorer(scorer_config)

    good_count = 0
    total_annotations = 0
    total_issues = 0

    for img_path in test_images:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)
        total_annotations += len(annotations)

        # Structural validation
        issues = []
        for idx, ann in enumerate(annotations):
            cls_id, cx, cy, w, h = ann
            # Check coords in [0, 1]
            for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                if val < 0.0 or val > 1.0:
                    issues.append({"type": "out_of_bounds", "severity": "high",
                                   "annotation_idx": idx, "detail": f"{name}={val}"})
            # Check valid class ID
            if cls_id < 0 or cls_id >= num_classes:
                issues.append({"type": "invalid_class", "severity": "high",
                               "annotation_idx": idx, "detail": f"cls_id={cls_id}"})
            # Check positive w/h
            if w <= 0 or h <= 0:
                issues.append({"type": "degenerate_box", "severity": "medium",
                               "annotation_idx": idx, "detail": f"w={w}, h={h}"})
            # Check tiny bbox
            if w < 0.01 or h < 0.01:
                issues.append({"type": "degenerate_box", "severity": "medium",
                               "annotation_idx": idx, "detail": f"tiny: w={w}, h={h}"})

        total_issues += len(issues)

        result = {
            "annotations": annotations,
            "validation_issues": issues,
            "sam3_verification": {},
        }
        scored = scorer.score_image(result)

        if scored["grade"] == "good":
            good_count += 1

    # Real fire dataset labels should be mostly clean
    good_ratio = good_count / len(test_images)
    assert good_ratio >= 0.5, (
        f"Expected at least 50% 'good' scores on clean fire data, "
        f"got {good_count}/{len(test_images)} ({good_ratio:.0%})"
    )

    print(f"    Checked {len(test_images)} images, {total_annotations} annotations, {total_issues} issues")
    print(f"    Good: {good_count}/{len(test_images)} ({good_ratio:.0%})")


# ---------------------------------------------------------------------------
# Helpers for GPU/SAM3 tests
# ---------------------------------------------------------------------------

def _get_test_image_and_annotations():
    """Return (PIL.Image, annotations, img_path) from the test_fire_100 dataset."""
    from PIL import Image
    from utils.yolo_io import image_to_label_path, parse_yolo_label
    from utils.exploration import get_image_paths

    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent
    base_path = resolve_path(data_config["path"], config_dir)
    images_dir = base_path / data_config["train"]
    all_images = sorted(get_image_paths(images_dir))
    assert len(all_images) > 0, f"No images found in {images_dir}"

    # Pick the first image that has at least one annotation
    for img_path in all_images:
        label_path = image_to_label_path(img_path)
        annotations = parse_yolo_label(label_path)
        if len(annotations) > 0:
            image = Image.open(img_path).convert("RGB")
            return image, annotations, img_path

    raise RuntimeError("No annotated images found in test_fire_100 train split")


def _get_test_image_paths(n: int = 5):
    """Return up to n real image paths (as Path objects) from the test dataset."""
    from utils.exploration import get_image_paths

    data_config = load_config(DATA_CONFIG_PATH)
    config_dir = Path(DATA_CONFIG_PATH).parent
    base_path = resolve_path(data_config["path"], config_dir)
    images_dir = base_path / data_config["train"]
    all_images = sorted(get_image_paths(images_dir))
    return all_images[:n]


def _clear_gpu():
    """No-op — services manage their own GPU memory."""
    pass


# ---------------------------------------------------------------------------
# GPU/SAM3 tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_sam3_service(), reason="SAM3 service not running at :18100")
def test_sam3_client_segment_with_box():
    """Load a real fire image, convert YOLO bbox to xyxy, call segment_with_box."""
    from core.p02_annotation_qa.sam3_client import SAM3Client

    image, annotations, img_path = _get_test_image_and_annotations()
    img_w, img_h = image.size

    # Convert first annotation YOLO (cx, cy, w, h) normalised -> absolute xyxy
    cls_id, cx, cy, w, h = annotations[0]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    box = [x1, y1, x2, y2]

    client = SAM3Client(service_url="http://localhost:18100")
    try:
        try:
            result = client.segment_with_box(image, box)
        except httpx.HTTPStatusError as e:
            pytest.skip(f"SAM3 service returned {e.response.status_code} (GPU may be unavailable)")

        assert "mask" in result, "Missing 'mask' key in result"
        assert isinstance(result["mask"], np.ndarray), f"mask type: {type(result['mask'])}"
        assert result["mask"].ndim == 2, f"mask ndim: {result['mask'].ndim}, expected 2"
        assert result["mask"].dtype == bool, f"mask dtype: {result['mask'].dtype}, expected bool"

        assert "bbox" in result, "Missing 'bbox' key in result"
        assert isinstance(result["bbox"], tuple), f"bbox type: {type(result['bbox'])}"
        assert len(result["bbox"]) == 4, f"bbox length: {len(result['bbox'])}, expected 4"
        for v in result["bbox"]:
            assert isinstance(v, float), f"bbox element type: {type(v)}, expected float"

        assert "iou_score" in result, "Missing 'iou_score' key in result"
        assert isinstance(result["iou_score"], float), f"iou_score type: {type(result['iou_score'])}"
        assert result["iou_score"] > 0, f"iou_score: {result['iou_score']}, expected > 0"

        print(f"    mask shape: {result['mask'].shape}, bbox: {result['bbox']}, "
              f"iou_score: {result['iou_score']:.4f}")
    finally:
        client.unload()
        _clear_gpu()


@pytest.mark.skipif(not has_sam3_service(), reason="SAM3 service not running at :18100")
def test_sam3_client_segment_with_text():
    """Load a real fire image, call segment_with_text('fire')."""
    from core.p02_annotation_qa.sam3_client import SAM3Client

    image, _, img_path = _get_test_image_and_annotations()

    client = SAM3Client(service_url="http://localhost:18100")
    try:
        try:
            result = client.segment_with_text(image, "fire")
        except httpx.HTTPStatusError as e:
            pytest.skip(f"SAM3 service returned {e.response.status_code} (GPU may be unavailable)")

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        print(f"    Found {len(result)} text-prompted detections")

        for i, det in enumerate(result):
            assert "mask" in det, f"Detection {i} missing 'mask'"
            assert isinstance(det["mask"], np.ndarray), f"Detection {i} mask type: {type(det['mask'])}"
            assert det["mask"].ndim == 2, f"Detection {i} mask ndim: {det['mask'].ndim}"

            assert "bbox" in det, f"Detection {i} missing 'bbox'"
            assert isinstance(det["bbox"], tuple), f"Detection {i} bbox type: {type(det['bbox'])}"
            assert len(det["bbox"]) == 4, f"Detection {i} bbox length: {len(det['bbox'])}"

            assert "score" in det, f"Detection {i} missing 'score'"
            assert isinstance(det["score"], float), f"Detection {i} score type: {type(det['score'])}"

            print(f"    Detection {i}: mask={det['mask'].shape}, bbox={det['bbox']}, "
                  f"score={det['score']:.4f}")
    finally:
        client.unload()
        _clear_gpu()


@pytest.mark.skipif(not has_sam3_service(), reason="SAM3 service not running at :18100")
def test_sam3_client_auto_mask():
    """Load a real fire image, call auto_mask."""
    from core.p02_annotation_qa.sam3_client import SAM3Client

    image, _, img_path = _get_test_image_and_annotations()

    client = SAM3Client(service_url="http://localhost:18100")
    try:
        try:
            result = client.auto_mask(image)
        except httpx.HTTPStatusError as e:
            pytest.skip(f"SAM3 service returned {e.response.status_code} (GPU may be unavailable)")

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "auto_mask returned empty list"
        print(f"    Found {len(result)} auto-masks")

        for i, det in enumerate(result[:3]):  # Check first 3 for brevity
            assert "mask" in det, f"Mask {i} missing 'mask'"
            assert isinstance(det["mask"], np.ndarray), f"Mask {i} type: {type(det['mask'])}"
            assert det["mask"].ndim == 2, f"Mask {i} ndim: {det['mask'].ndim}"

            assert "bbox" in det, f"Mask {i} missing 'bbox'"
            assert isinstance(det["bbox"], tuple) and len(det["bbox"]) == 4

            assert "area" in det, f"Mask {i} missing 'area'"
            assert isinstance(det["area"], float), f"Mask {i} area type: {type(det['area'])}"

            assert "score" in det, f"Mask {i} missing 'score'"
            assert isinstance(det["score"], float), f"Mask {i} score type: {type(det['score'])}"

            print(f"    Mask {i}: shape={det['mask'].shape}, area={det['area']:.4f}, "
                  f"score={det['score']:.4f}")
    finally:
        client.unload()
        _clear_gpu()


def test_sam3_client_mask_to_bbox():
    """Create a synthetic boolean mask and verify mask_to_bbox computes correct cx, cy, w, h."""
    from core.p02_annotation_qa.sam3_client import SAM3Client

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:61, 30:71] = True  # rows 20-60, cols 30-70 (inclusive)

    cx, cy, w, h = SAM3Client.mask_to_bbox(mask, 100, 100)

    # Expected: x range [30, 71) -> x1=30, x2=71, cx=(30+71)/2/100=0.505, w=(71-30)/100=0.41
    # Expected: y range [20, 61) -> y1=20, y2=61, cy=(20+61)/2/100=0.405, h=(61-20)/100=0.41
    assert abs(cx - 0.505) < 0.02, f"cx={cx}, expected ~0.505"
    assert abs(cy - 0.405) < 0.02, f"cy={cy}, expected ~0.405"
    assert abs(w - 0.41) < 0.02, f"w={w}, expected ~0.41"
    assert abs(h - 0.41) < 0.02, f"h={h}, expected ~0.41"

    print(f"    mask_to_bbox: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")

    # Test empty mask
    empty_mask = np.zeros((50, 50), dtype=bool)
    result = SAM3Client.mask_to_bbox(empty_mask, 50, 50)
    assert result == (0.0, 0.0, 0.0, 0.0), f"Empty mask should return zeros, got {result}"
    print(f"    Empty mask: {result}")


@pytest.mark.skipif(not has_qa_service(), reason="QA service not running at :18105")
def test_qa_pipeline_with_service():
    """Run QA functional API pipeline on 5 real images with SAM3 verification."""
    from core.p02_annotation_qa.pipeline import qa_pipeline

    data_config = load_config(DATA_CONFIG_PATH)
    qa_config = load_config(str(ROOT / "configs" / "_shared" / "02_annotation_quality.yaml"))

    # Override sampling to keep it small
    qa_config["sampling"]["sample_size"] = 5
    qa_config["sampling"]["min_per_class"] = 1
    qa_config["sampling"]["splits"] = ["train"]
    qa_config["processing"]["batch_size"] = 5

    config_dir = str(Path(DATA_CONFIG_PATH).parent)

    initial_state = {
        "data_config": data_config,
        "qa_config": qa_config,
        "dataset_name": "test_fire_100",
        "class_names": data_config["names"],
        "splits": ["train"],
        "config_dir": config_dir,
        "use_sam3": True,
        "image_results": [],
        "auto_label_config": {},
    }

    try:
        final_state = qa_pipeline.invoke(initial_state)

        assert "summary" in final_state, "Final state missing 'summary'"
        summary = final_state["summary"]

        assert "total_checked" in summary, "Summary missing 'total_checked'"
        assert summary["total_checked"] > 0, f"total_checked={summary['total_checked']}"
        assert "grades" in summary, "Summary missing 'grades'"
        assert "avg_quality_score" in summary, "Summary missing 'avg_quality_score'"

        grade_dist = summary["grades"]
        grade_total = sum(grade_dist.values())
        assert grade_total == summary["total_checked"], (
            f"Grade distribution total {grade_total} != total_checked {summary['total_checked']}"
        )

        assert "report_path" in final_state, "Final state missing 'report_path'"
        report_path = Path(final_state["report_path"])
        assert report_path.exists(), f"Report path does not exist: {report_path}"

        print(f"    Pipeline completed: {summary['total_checked']} images checked")
        print(f"    Grades: {grade_dist}")
        print(f"    Avg score: {summary['avg_quality_score']:.3f}")
        print(f"    Report: {report_path}")
    finally:
        _clear_gpu()


@pytest.mark.skipif(not has_qa_service(), reason="QA service not running at :18105")
def test_qa_pipeline_no_sam3():
    """Run QA functional API pipeline on 5 real images without SAM3 (structural only)."""
    from core.p02_annotation_qa.pipeline import qa_pipeline

    data_config = load_config(DATA_CONFIG_PATH)
    qa_config = load_config(str(ROOT / "configs" / "_shared" / "02_annotation_quality.yaml"))

    qa_config["sampling"]["sample_size"] = 5
    qa_config["sampling"]["min_per_class"] = 1
    qa_config["sampling"]["splits"] = ["train"]
    qa_config["processing"]["batch_size"] = 5

    config_dir = str(Path(DATA_CONFIG_PATH).parent)

    initial_state = {
        "data_config": data_config,
        "qa_config": qa_config,
        "dataset_name": "test_fire_100",
        "class_names": data_config["names"],
        "splits": ["train"],
        "config_dir": config_dir,
        "use_sam3": False,
        "image_results": [],
        "auto_label_config": {},
    }

    try:
        final_state = qa_pipeline.invoke(initial_state)
        summary = final_state.get("summary", {})

        assert summary["total_checked"] > 0, f"total_checked={summary['total_checked']}"
        print(f"    Pipeline (no SAM3): {summary['total_checked']} images, "
              f"avg_score={summary.get('avg_quality_score', 0):.3f}")
    finally:
        _clear_gpu()


# ---------------------------------------------------------------------------
# Functional API pipeline + multi-usecase tests
# ---------------------------------------------------------------------------


def test_qa_multi_usecase_configs():
    """Verify all 5 test configs parse correctly for QA use."""
    configs_dir = ROOT / "configs" / "_test"
    config_files = {
        "01_data_fire.yaml": {"has_auto_label": False, "num_classes": 2},
        "01_data_ppe.yaml": {"has_auto_label": True, "num_classes": 3},
        "01_data_shoes.yaml": {"has_auto_label": True, "num_classes": 3},
        "01_data_phone.yaml": {"has_auto_label": False, "num_classes": 1},
        "01_data_fall.yaml": {"has_auto_label": False, "num_classes": 2},
    }

    for filename, expected in config_files.items():
        cfg_path = configs_dir / filename
        if not cfg_path.exists():
            print(f"    SKIP: {filename} not found")
            continue

        cfg = load_config(cfg_path)
        assert "names" in cfg, f"{filename}: missing 'names'"
        assert "dataset_name" in cfg, f"{filename}: missing 'dataset_name'"
        assert len(cfg["names"]) == expected["num_classes"], (
            f"{filename}: expected {expected['num_classes']} classes, got {len(cfg['names'])}"
        )

        has_auto_label = "auto_label" in cfg
        assert has_auto_label == expected["has_auto_label"], (
            f"{filename}: auto_label={'present' if has_auto_label else 'absent'}, "
            f"expected {'present' if expected['has_auto_label'] else 'absent'}"
        )

        if has_auto_label:
            al = cfg["auto_label"]
            assert "detection_classes" in al, f"{filename}: auto_label missing detection_classes"
            assert "class_rules" in al, f"{filename}: auto_label missing class_rules"

        uc = filename.replace("01_data_", "").replace(".yaml", "")
        print(f"    {uc}: {len(cfg['names'])} classes, auto_label={'yes' if has_auto_label else 'no'}")


def test_qa_with_rules_service():
    """Run QA on PPE dataset with auto_label config via functional API pipeline."""
    if not has_qa_service():
        print("    SKIP: QA service not running at :18105")
        return

    ppe_config_path = ROOT / "configs" / "_test" / "01_data_ppe.yaml"
    if not ppe_config_path.exists():
        print("    SKIP: PPE test config not found")
        return

    ppe_config = load_config(ppe_config_path)
    dataset_path = resolve_path(ppe_config["path"], ppe_config_path.parent)
    if not dataset_path.exists():
        print(f"    SKIP: PPE dataset not found at {dataset_path}")
        return

    from core.p02_annotation_qa.pipeline import qa_pipeline

    qa_config_path = ROOT / "configs" / "_shared" / "02_annotation_quality.yaml"
    qa_config = load_config(qa_config_path)

    class_names = {int(k): v for k, v in ppe_config["names"].items()}
    auto_label_config = ppe_config.get("auto_label")

    initial_state = {
        "data_config": ppe_config,
        "qa_config": qa_config,
        "dataset_name": ppe_config["dataset_name"],
        "class_names": class_names,
        "splits": ["train"],
        "config_dir": str(ppe_config_path.parent),
        "total_sampled": 0,
        "current_batch_idx": 0,
        "total_batches": 0,
        "batch_size": 32,
        "image_results": [],
        "summary": {},
        "report_path": "",
        "use_sam3": False,
        "auto_label_config": auto_label_config,
    }

    result = qa_pipeline.invoke(initial_state)

    summary = result.get("summary", {})
    assert summary.get("total_checked", 0) > 0, "QA should check at least some images"
    assert "grades" in summary, "Summary should have grade distribution"

    print(f"    QA pipeline checked {summary['total_checked']} PPE images")
    print(f"    Grades: {summary.get('grades', {})}")
    print(f"    Avg score: {summary.get('avg_quality_score', 0):.3f}")


if __name__ == "__main__":
    cpu_tests = [
        ("stratified_sampler", test_stratified_sampler),
        ("quality_scorer_perfect", test_quality_scorer_perfect),
        ("quality_scorer_issues", test_quality_scorer_issues),
        ("generate_fixes", test_generate_fixes),
        ("reporter_generate", test_reporter_generate),
        ("build_qa_pipeline_import", test_build_qa_pipeline_import),
        ("sampler_min_per_class", test_sampler_min_per_class),
        ("scorer_edge_cases", test_scorer_edge_cases),
        ("scorer_all_penalties", test_scorer_all_penalties),
        ("reporter_with_real_results", test_reporter_with_real_results),
        ("structural_validation_on_real_labels", test_structural_validation_on_real_labels),
        ("qa_multi_usecase_configs", test_qa_multi_usecase_configs),
    ]

    service_tests = []
    if has_sam3_service():
        service_tests.extend([
            ("sam3_client_segment_with_box", test_sam3_client_segment_with_box),
            ("sam3_client_segment_with_text", test_sam3_client_segment_with_text),
            ("sam3_client_auto_mask", test_sam3_client_auto_mask),
        ])
    else:
        print("  SKIP: SAM3 service not running at :18100")

    service_tests.append(("sam3_client_mask_to_bbox", test_sam3_client_mask_to_bbox))

    if has_qa_service():
        service_tests.extend([
            ("qa_pipeline_with_service", test_qa_pipeline_with_service),
            ("qa_pipeline_no_sam3", test_qa_pipeline_no_sam3),
            ("qa_with_rules_service", test_qa_with_rules_service),
        ])
    else:
        print("  SKIP: QA service not running at :18105")

    run_all(
        cpu_tests + service_tests,
        title="Test 12: Annotation QA (CPU + GPU/SAM3)",
    )
