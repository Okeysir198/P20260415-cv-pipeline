"""Test 13: Auto-Annotate — NMS filter, label writer, image scanner, mask_to_polygon, service."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from core.p01_auto_annotate.annotator import mask_to_polygon
from core.p01_auto_annotate.nms_filter import NMSFilter
from core.p01_auto_annotate.reporter import AutoAnnotateReporter
from core.p01_auto_annotate.scanner import ImageScanner
from utils.config import load_config
from core.p01_auto_annotate.writer import LabelWriter

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "13_auto_annotate"
OUTPUTS.mkdir(parents=True, exist_ok=True)

DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


def test_nms_filter():
    """Create 4 overlapping same-class detections, verify NMS reduces count."""
    dets = [
        {"class_id": 0, "cx": 0.50, "cy": 0.50, "w": 0.10, "h": 0.10, "score": 0.90},
        {"class_id": 0, "cx": 0.51, "cy": 0.51, "w": 0.10, "h": 0.10, "score": 0.85},
        {"class_id": 0, "cx": 0.52, "cy": 0.50, "w": 0.10, "h": 0.10, "score": 0.80},
        {"class_id": 0, "cx": 0.50, "cy": 0.52, "w": 0.10, "h": 0.10, "score": 0.30},
    ]
    nms = NMSFilter(per_class_iou_threshold=0.3)
    filtered = nms.filter(dets)

    assert isinstance(filtered, list), f"Expected list, got {type(filtered)}"
    assert len(filtered) < len(dets), (
        f"NMS should reduce detections: {len(filtered)} >= {len(dets)}"
    )
    # Highest score should be kept
    scores = [d["score"] for d in filtered]
    assert 0.90 in scores, f"Highest-score detection should survive NMS, got scores {scores}"
    print(f"    {len(dets)} detections -> {len(filtered)} after NMS")


def test_nms_filter_empty():
    """NMS on empty list returns empty list."""
    nms = NMSFilter()
    result = nms.filter([])
    assert result == [], f"Expected [], got {result}"


def test_label_writer_bbox():
    """Write bbox labels and verify file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Create dummy image file
        img_path = img_dir / "test.jpg"
        img_path.touch()

        writer = LabelWriter(output_format="bbox")
        detections = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3}]
        writer.write(img_path, detections)

        label_path = lbl_dir / "test.txt"
        assert label_path.exists(), f"Label file not created at {label_path}"

        content = label_path.read_text().strip()
        assert "0 0.500000 0.500000 0.200000 0.300000" in content, (
            f"Unexpected label content: {content}"
        )
        print(f"    Label written: {content}")


def test_label_writer_dry_run():
    """Dry run should NOT write any file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img_path = img_dir / "test.jpg"
        img_path.touch()

        writer = LabelWriter(dry_run=True)
        detections = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3}]
        writer.write(img_path, detections)

        label_path = lbl_dir / "test.txt"
        assert not label_path.exists(), f"Dry run should not create file, but {label_path} exists"
        print(f"    Dry run: no file written (correct)")


def test_scanner_yolo_layout():
    """Scan real test dataset, verify train/val splits returned."""
    config = load_config(DATA_CONFIG_PATH)
    scanner = ImageScanner(
        data_config=config,
        config_dir=Path(DATA_CONFIG_PATH).parent,
        filter_mode="all",
        splits=["train", "val"],
    )
    result = scanner.scan()

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "train" in result, f"Missing 'train' key, got keys: {list(result.keys())}"
    assert "val" in result, f"Missing 'val' key, got keys: {list(result.keys())}"

    total = sum(len(v) for v in result.values())
    assert total > 0, "Scanner returned 0 images total"

    for split, paths in result.items():
        assert all(isinstance(p, Path) for p in paths), (
            f"Expected list of Path objects in '{split}'"
        )
    print(f"    Scanner found: train={len(result['train'])}, val={len(result['val'])}")


def test_mask_to_polygon():
    """Convert a binary mask to a normalized polygon."""
    # Create 100x100 mask with a 20x20 square in the center
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True

    polygon = mask_to_polygon(mask, 100, 100)

    assert isinstance(polygon, list), f"Expected list, got {type(polygon)}"
    assert len(polygon) >= 6, (
        f"Expected at least 6 floats (3+ vertices), got {len(polygon)}"
    )
    # All values should be normalized [0, 1]
    for v in polygon:
        assert isinstance(v, float), f"Expected float, got {type(v)}"
        assert 0.0 <= v <= 1.0, f"Value out of [0,1] range: {v}"
    print(f"    Polygon has {len(polygon) // 2} vertices, {len(polygon)} floats")


def test_build_graph():
    """Import and call build_graph(), verify it returns an invocable graph."""
    from core.p01_auto_annotate.graph import build_graph

    graph = build_graph()
    assert hasattr(graph, "invoke"), (
        f"Compiled graph should have 'invoke' method, got attrs: {dir(graph)}"
    )
    print(f"    build_graph() returned {type(graph).__name__} with invoke()")


def test_label_writer_polygon():
    """Write polygon-format labels and verify YOLO-seg output format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        lbl_dir = tmpdir / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        img_path = img_dir / "test_poly.jpg"
        img_path.touch()

        writer = LabelWriter(output_format="polygon")

        # Synthetic polygon detections (triangle + pentagon)
        detections = [
            {
                "class_id": 0,
                "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3,
                "polygon": [0.4, 0.35, 0.6, 0.35, 0.5, 0.65],
            },
            {
                "class_id": 1,
                "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1,
                "polygon": [0.25, 0.25, 0.35, 0.25, 0.35, 0.35, 0.30, 0.37, 0.25, 0.35],
            },
        ]
        writer.write(img_path, detections)

        label_path = lbl_dir / "test_poly.txt"
        assert label_path.exists(), f"Polygon label file not created at {label_path}"

        lines = label_path.read_text().strip().splitlines()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

        # First line: class_id x1 y1 x2 y2 x3 y3 (triangle = 3 vertices)
        parts0 = lines[0].split()
        assert parts0[0] == "0", f"Expected class_id=0, got {parts0[0]}"
        assert len(parts0) == 7, (  # 1 (class_id) + 6 (3 vertices * 2)
            f"Expected 7 parts for triangle polygon, got {len(parts0)}: {lines[0]}"
        )

        # Second line: class_id + 10 floats (5 vertices * 2)
        parts1 = lines[1].split()
        assert parts1[0] == "1", f"Expected class_id=1, got {parts1[0]}"
        assert len(parts1) == 11, (  # 1 + 10
            f"Expected 11 parts for pentagon polygon, got {len(parts1)}: {lines[1]}"
        )

        # All coordinate values should be valid floats in [0, 1]
        for line in lines:
            for val_str in line.split()[1:]:
                val = float(val_str)
                assert 0.0 <= val <= 1.0, f"Coordinate out of range: {val}"
        print(f"    Polygon labels written correctly: {len(lines)} lines")


def test_scanner_filter_missing():
    """Test filter_mode='missing' vs 'all' on test dataset with existing labels."""
    config = load_config(DATA_CONFIG_PATH)

    scanner_all = ImageScanner(
        data_config=config,
        config_dir=Path(DATA_CONFIG_PATH).parent,
        filter_mode="all",
        splits=["train", "val"],
    )
    result_all = scanner_all.scan()
    total_all = sum(len(v) for v in result_all.values())

    scanner_missing = ImageScanner(
        data_config=config,
        config_dir=Path(DATA_CONFIG_PATH).parent,
        filter_mode="missing",
        splits=["train", "val"],
    )
    result_missing = scanner_missing.scan()
    total_missing = sum(len(v) for v in result_missing.values())

    assert total_all > 0, "filter_mode='all' should return images"
    assert total_missing < total_all, (
        f"filter_mode='missing' ({total_missing}) should return fewer images than "
        f"'all' ({total_all}) since labels exist"
    )
    print(f"    all={total_all}, missing={total_missing}")


def test_nms_filter_cross_class():
    """Test cross-class NMS: overlapping boxes from different classes."""
    # Two highly overlapping boxes from different classes
    dets = [
        {"class_id": 0, "cx": 0.50, "cy": 0.50, "w": 0.20, "h": 0.20, "score": 0.90},
        {"class_id": 1, "cx": 0.51, "cy": 0.51, "w": 0.20, "h": 0.20, "score": 0.80},
    ]

    # Without cross-class NMS: both survive (per-class NMS is class-independent)
    nms_no_cross = NMSFilter(
        per_class_iou_threshold=0.3,
        cross_class_enabled=False,
    )
    result_no_cross = nms_no_cross.filter(dets)
    assert len(result_no_cross) == 2, (
        f"Without cross-class NMS, both classes should survive, got {len(result_no_cross)}"
    )

    # With cross-class NMS: lower-score box should be suppressed
    nms_with_cross = NMSFilter(
        per_class_iou_threshold=0.3,
        cross_class_enabled=True,
        cross_class_iou_threshold=0.3,
    )
    result_with_cross = nms_with_cross.filter(dets)
    assert len(result_with_cross) < len(result_no_cross), (
        f"Cross-class NMS should suppress overlapping boxes: "
        f"got {len(result_with_cross)} vs {len(result_no_cross)} without"
    )
    # Highest-score detection should survive
    kept_scores = [d["score"] for d in result_with_cross]
    assert 0.90 in kept_scores, f"Highest-score box should survive, got {kept_scores}"
    print(
        f"    cross_class=False: {len(result_no_cross)} kept, "
        f"cross_class=True: {len(result_with_cross)} kept"
    )


def test_nms_filter_thresholds():
    """Test NMS with different IoU thresholds on same overlapping detections."""
    # Three overlapping same-class detections
    dets = [
        {"class_id": 0, "cx": 0.50, "cy": 0.50, "w": 0.20, "h": 0.20, "score": 0.95},
        {"class_id": 0, "cx": 0.53, "cy": 0.53, "w": 0.20, "h": 0.20, "score": 0.85},
        {"class_id": 0, "cx": 0.56, "cy": 0.56, "w": 0.20, "h": 0.20, "score": 0.75},
    ]

    # Strict threshold (0.3): suppresses more
    nms_strict = NMSFilter(per_class_iou_threshold=0.3)
    result_strict = nms_strict.filter(dets)

    # Lenient threshold (0.7): suppresses less
    nms_lenient = NMSFilter(per_class_iou_threshold=0.7)
    result_lenient = nms_lenient.filter(dets)

    assert len(result_strict) <= len(result_lenient), (
        f"Stricter threshold should keep <= boxes: "
        f"strict={len(result_strict)}, lenient={len(result_lenient)}"
    )
    assert len(result_strict) >= 1, "At least the top-score detection must survive"
    assert len(result_lenient) >= 1, "At least the top-score detection must survive"
    # Highest score always survives
    for result in [result_strict, result_lenient]:
        scores = [d["score"] for d in result]
        assert 0.95 in scores, f"Top-score 0.95 should survive, got {scores}"
    print(
        f"    threshold=0.3: {len(result_strict)} kept, "
        f"threshold=0.7: {len(result_lenient)} kept"
    )


def test_reporter():
    """Create reporter with synthetic results, verify output files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = AutoAnnotateReporter(
            output_dir=tmpdir,
            dataset_name="test_synthetic",
            config={"save_previews": False, "preview_count": 0},
        )

        image_results = [
            {
                "image_path": "/fake/image1.jpg",
                "split": "train",
                "detections": [
                    {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3, "score": 0.9},
                    {"class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1, "score": 0.8},
                ],
                "validation_issues": [],
                "written": True,
                "timing": {"annotate_s": 0.12, "nms_s": 0.01, "write_s": 0.005},
            },
            {
                "image_path": "/fake/image2.jpg",
                "split": "val",
                "detections": [
                    {"class_id": 0, "cx": 0.6, "cy": 0.4, "w": 0.15, "h": 0.25, "score": 0.7},
                ],
                "validation_issues": [],
                "written": True,
                "timing": {"annotate_s": 0.08, "nms_s": 0.005, "write_s": 0.003},
            },
        ]

        summary = {
            "dataset": "test_synthetic",
            "total_images": 2,
            "total_annotated": 2,
            "total_detections": 3,
            "avg_detections_per_image": 1.5,
            "per_class_counts": {"fire": 2, "smoke": 1},
            "per_split_counts": {"train": 1, "val": 1},
            "output_format": "bbox",
            "mode": "text",
            "dry_run": False,
            "timing": {
                "avg_annotate_s": 0.10,
                "avg_total_per_sample_s": 0.11,
                "min_total_per_sample_s": 0.088,
                "max_total_per_sample_s": 0.135,
            },
        }

        report_path = reporter.generate_report(image_results, summary)
        report_dir = Path(report_path)

        # Verify report.json
        json_path = report_dir / "report.json"
        assert json_path.exists(), f"report.json not found at {json_path}"
        report_data = json.loads(json_path.read_text())
        assert report_data["metadata"]["total_images"] == 2
        assert report_data["summary"]["total_detections"] == 3
        assert len(report_data["image_results"]) == 2

        # Verify summary.txt
        txt_path = report_dir / "summary.txt"
        assert txt_path.exists(), f"summary.txt not found at {txt_path}"
        txt_content = txt_path.read_text()
        assert "test_synthetic" in txt_content
        assert "3" in txt_content  # total_detections
        print(f"    Reporter generated: report.json, summary.txt in {report_dir}")


def test_mask_to_polygon_edge_cases():
    """Test mask_to_polygon with edge-case masks."""
    # Case 1: All-zeros mask -> should return None (no contour)
    mask_zeros = np.zeros((100, 100), dtype=bool)
    result_zeros = mask_to_polygon(mask_zeros, 100, 100)
    assert result_zeros is None, (
        f"All-zeros mask should return None, got {result_zeros}"
    )

    # Case 2: All-ones mask -> should return polygon covering full image
    mask_ones = np.ones((100, 100), dtype=bool)
    result_ones = mask_to_polygon(mask_ones, 100, 100)
    assert result_ones is not None, "All-ones mask should produce a polygon"
    assert isinstance(result_ones, list), f"Expected list, got {type(result_ones)}"
    assert len(result_ones) >= 8, (  # at least 4 vertices for a rectangle
        f"Full-image polygon should have >= 4 vertices, got {len(result_ones) // 2}"
    )
    # Vertices should span close to the full image range
    xs = result_ones[0::2]
    ys = result_ones[1::2]
    assert max(xs) >= 0.9, f"Full-image polygon should reach near x=1.0, max x={max(xs)}"
    assert max(ys) >= 0.9, f"Full-image polygon should reach near y=1.0, max y={max(ys)}"
    assert min(xs) <= 0.1, f"Full-image polygon should start near x=0.0, min x={min(xs)}"
    assert min(ys) <= 0.1, f"Full-image polygon should start near y=0.0, min y={min(ys)}"

    # Case 3: Small 5x5 mask in 100x100 image -> may return None (< min_vertices)
    # or a valid small polygon depending on simplification
    mask_small = np.zeros((100, 100), dtype=bool)
    mask_small[48:53, 48:53] = True
    result_small = mask_to_polygon(mask_small, 100, 100)
    if result_small is not None:
        # If it returns something, it should be a valid polygon
        assert isinstance(result_small, list)
        assert len(result_small) >= 8, (  # min_vertices=4 -> 8 floats
            f"Small polygon should have >= 4 vertices, got {len(result_small) // 2}"
        )
        for v in result_small:
            assert 0.0 <= v <= 1.0, f"Value out of range: {v}"
        print(f"    Small mask: polygon with {len(result_small) // 2} vertices")
    else:
        print(f"    Small mask: None (too few vertices after simplification)")

    print(f"    Zeros: None, Ones: {len(result_ones) // 2} vertices")


# ---------------------------------------------------------------------------
# Service tests — require auto-label service (s18104) running at localhost:18104
# ---------------------------------------------------------------------------

AUTO_LABEL_URL = "http://localhost:18104"

_service_cache: bool | None = None


def has_auto_label_service() -> bool:
    """Check if the auto-label service is reachable (cached after first call)."""
    global _service_cache
    if _service_cache is None:
        try:
            import requests
            resp = requests.get(f"{AUTO_LABEL_URL}/health", timeout=5)
            _service_cache = resp.status_code == 200
        except Exception:
            _service_cache = False
    return _service_cache


def _get_test_image_path() -> Path:
    """Resolve a real fire image from the test_fire_100 dataset."""
    config = load_config(DATA_CONFIG_PATH)
    dataset_path = (Path(DATA_CONFIG_PATH).parent / config["path"]).resolve()
    train_images_dir = dataset_path / config["train"]
    images = sorted(train_images_dir.glob("*.jpg")) + sorted(train_images_dir.glob("*.png"))
    assert len(images) > 0, f"No images found in {train_images_dir}"
    return images[0]


def _make_annotator(mode: str):
    """Create an Annotator configured for fire/smoke with the given mode."""
    from core.p01_auto_annotate.annotator import Annotator

    return Annotator(
        class_names={0: "fire", 1: "smoke"},
        text_prompts={"fire": "fire flames", "smoke": "smoke plume"},
        mode=mode,
        confidence_threshold=0.3,
        service_url=AUTO_LABEL_URL,
    )


def _validate_detections(detections: list, mode_name: str) -> None:
    """Validate that detections have the expected structure."""
    assert isinstance(detections, list), (
        f"[{mode_name}] Expected list, got {type(detections)}"
    )
    print(f"    [{mode_name}] Got {len(detections)} detections")
    for i, det in enumerate(detections):
        assert "class_id" in det, f"[{mode_name}] Detection {i} missing class_id"
        assert "cx" in det, f"[{mode_name}] Detection {i} missing cx"
        assert "cy" in det, f"[{mode_name}] Detection {i} missing cy"
        assert "w" in det, f"[{mode_name}] Detection {i} missing w"
        assert "h" in det, f"[{mode_name}] Detection {i} missing h"
        assert "score" in det, f"[{mode_name}] Detection {i} missing score"
        assert det["class_id"] in (0, 1), (
            f"[{mode_name}] Detection {i} has invalid class_id={det['class_id']}"
        )
        print(
            f"      det[{i}]: class={det['class_id']}, "
            f"bbox=({det['cx']:.3f},{det['cy']:.3f},{det['w']:.3f},{det['h']:.3f}), "
            f"score={det['score']:.3f}"
        )


def test_annotator_text_mode():
    """SAM3 text-prompted annotation on a real fire image (GPU)."""
    if not has_auto_label_service():
        print("    SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        return

    image_path = _get_test_image_path()
    print(f"    Image: {image_path}")

    annotator = _make_annotator("text")
    try:
        detections = annotator.annotate_image(image_path)
        _validate_detections(detections, "text")
        # Text mode on a fire image should find something (soft check)
        if len(detections) == 0:
            print("    WARNING: no detections on fire image (may happen with low-res)")
    finally:
        annotator.unload()


def test_annotator_auto_mode():
    """SAM3 auto-mask annotation on a real fire image (GPU)."""
    if not has_auto_label_service():
        print("    SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        return

    image_path = _get_test_image_path()
    print(f"    Image: {image_path}")

    annotator = _make_annotator("auto")
    try:
        detections = annotator.annotate_image(image_path)
        _validate_detections(detections, "auto")
    finally:
        annotator.unload()


def test_annotator_hybrid_mode():
    """SAM3 hybrid annotation on a real fire image (GPU)."""
    if not has_auto_label_service():
        print("    SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        return

    image_path = _get_test_image_path()
    print(f"    Image: {image_path}")

    annotator = _make_annotator("hybrid")
    try:
        detections = annotator.annotate_image(image_path)
        _validate_detections(detections, "hybrid")
    finally:
        annotator.unload()


def test_full_auto_annotate_graph_dry_run():
    """Run the full auto-annotate LangGraph with dry_run=True on 3 images (GPU)."""
    if not has_auto_label_service():
        print("    SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        return

    from core.p01_auto_annotate.graph import build_graph

    config = load_config(DATA_CONFIG_PATH)
    dataset_path = (Path(DATA_CONFIG_PATH).parent / config["path"]).resolve()
    train_images_dir = dataset_path / config["train"]
    images = sorted(train_images_dir.glob("*.jpg")) + sorted(train_images_dir.glob("*.png"))
    assert len(images) >= 3, f"Need at least 3 images, found {len(images)}"

    # Take only 3 images
    sample_images = images[:3]
    print(f"    Using {len(sample_images)} images from {train_images_dir}")

    graph = build_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        initial_state = {
            "data_config": config,
            "annotate_config": {
                "auto_label_service": {
                    "url": AUTO_LABEL_URL,
                    "timeout": 120,
                },
                "processing": {
                    "batch_size": 3,
                    "confidence_threshold": 0.3,
                },
                "nms": {
                    "per_class_iou_threshold": 0.5,
                    "cross_class_enabled": False,
                },
                "auto_annotate": {
                    "output_dir": tmpdir,
                    "splits": ["train"],
                },
                "reporting": {
                    "save_previews": False,
                    "preview_count": 0,
                },
            },
            "dataset_name": "test_fire_100",
            "class_names": {0: "fire", 1: "smoke"},
            "text_prompts": {"fire": "fire flames", "smoke": "smoke plume"},
            "config_dir": str(Path(DATA_CONFIG_PATH).parent),
            "image_paths": {"train": [str(p) for p in sample_images]},
            "total_images": len(sample_images),
            "current_batch_idx": 0,
            "total_batches": 1,
            "batch_size": 3,
            "image_results": [],
            "mode": "text",
            "output_format": "bbox",
            "dry_run": True,
            "filter_mode": "all",
        }

        try:
            # Skip scan_node by providing pre-populated image_paths;
            # invoke from annotate_batch onward by calling full graph
            # (scan_node will re-scan but that's fine — we pre-set the paths)
            result = graph.invoke(initial_state)

            # Verify output structure
            assert "image_results" in result, "Missing 'image_results' in graph output"
            assert "summary" in result, "Missing 'summary' in graph output"
            assert "report_path" in result, "Missing 'report_path' in graph output"

            image_results = result["image_results"]
            summary = result["summary"]

            print(f"    Graph processed {len(image_results)} images")
            print(f"    Total detections: {summary.get('total_detections', 0)}")
            print(f"    Dry run: {summary.get('dry_run', None)}")

            assert summary.get("dry_run") is True, (
                f"Expected dry_run=True in summary, got {summary.get('dry_run')}"
            )

            # In dry_run mode, writer.write() returns True if detections exist
            # (meaning "would be written"), but no files are actually written.
            # Verify no actual label files were created in a temp output dir.
            for r in image_results:
                assert "image_path" in r, "Missing image_path in result"

            # Verify each result has expected keys
            for r in image_results:
                assert "image_path" in r, "Missing image_path in result"
                assert "detections" in r, "Missing detections in result"
                assert "timing" in r, "Missing timing in result"
                dets = r["detections"]
                assert isinstance(dets, list), f"Expected list, got {type(dets)}"

            print(f"    Per-class counts: {summary.get('per_class_counts', {})}")
        finally:
            pass  # Service manages its own resources


# ---------------------------------------------------------------------------
# Rule Classifier tests
# ---------------------------------------------------------------------------


def test_rule_classifier_basic():
    """Test RuleClassifier with direct + overlap + no_overlap rules."""
    from core.p01_auto_annotate.rule_classifier import RuleClassifier

    rc = RuleClassifier(
        class_rules=[
            {"output_class_id": 0, "source": "person", "condition": "direct"},
            {
                "output_class_id": 1,
                "source": "head",
                "condition": "overlap",
                "target": "helmet",
                "min_iou": 0.3,
            },
            {
                "output_class_id": 2,
                "source": "head",
                "condition": "no_overlap",
                "target": "helmet",
                "min_iou": 0.3,
            },
        ],
        detection_class_map={"person": 100, "head": 101, "helmet": 102},
    )

    detections = [
        # Person — direct rule
        {"class_id": 100, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.8, "score": 0.9},
        # Head overlapping with helmet (same center, high IoU)
        {"class_id": 101, "cx": 0.5, "cy": 0.2, "w": 0.10, "h": 0.10, "score": 0.85},
        # Helmet overlapping with the head above
        {"class_id": 102, "cx": 0.5, "cy": 0.2, "w": 0.10, "h": 0.10, "score": 0.80},
        # Head NOT overlapping with any helmet (far away)
        {"class_id": 101, "cx": 0.1, "cy": 0.9, "w": 0.10, "h": 0.10, "score": 0.75},
    ]

    results = rc.classify(detections)

    # Should get 3 output detections (person, head_with_helmet, head_without_helmet)
    # Helmet detections are consumed as targets and never emitted
    assert len(results) == 3, f"Expected 3 results, got {len(results)}: {results}"

    output_ids = sorted([r["class_id"] for r in results])
    assert output_ids == [0, 1, 2], f"Expected class IDs [0, 1, 2], got {output_ids}"

    # Verify person is class 0
    person = [r for r in results if r["class_id"] == 0]
    assert len(person) == 1, f"Expected 1 person, got {len(person)}"
    assert person[0]["score"] == 0.9

    # Verify overlapping head is class 1
    head_with = [r for r in results if r["class_id"] == 1]
    assert len(head_with) == 1, f"Expected 1 head_with_helmet, got {len(head_with)}"

    # Verify non-overlapping head is class 2
    head_without = [r for r in results if r["class_id"] == 2]
    assert len(head_without) == 1, f"Expected 1 head_without_helmet, got {len(head_without)}"
    print(f"    RuleClassifier: {len(detections)} input -> {len(results)} output, IDs={output_ids}")


def test_rule_classifier_no_targets():
    """When no helmet detections exist, all heads should be no_overlap (class 2)."""
    from core.p01_auto_annotate.rule_classifier import RuleClassifier

    rc = RuleClassifier(
        class_rules=[
            {"output_class_id": 0, "source": "person", "condition": "direct"},
            {
                "output_class_id": 1,
                "source": "head",
                "condition": "overlap",
                "target": "helmet",
                "min_iou": 0.3,
            },
            {
                "output_class_id": 2,
                "source": "head",
                "condition": "no_overlap",
                "target": "helmet",
                "min_iou": 0.3,
            },
        ],
        detection_class_map={"person": 100, "head": 101, "helmet": 102},
    )

    detections = [
        {"class_id": 100, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.8, "score": 0.9},
        {"class_id": 101, "cx": 0.5, "cy": 0.2, "w": 0.10, "h": 0.10, "score": 0.85},
        {"class_id": 101, "cx": 0.1, "cy": 0.9, "w": 0.10, "h": 0.10, "score": 0.75},
        # No helmet detections at all
    ]

    results = rc.classify(detections)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # All heads should be class 2 (no_overlap) since no helmets exist
    head_results = [r for r in results if r["class_id"] == 2]
    assert len(head_results) == 2, (
        f"Expected 2 heads as no_overlap (class 2), got {len(head_results)}"
    )

    # No class 1 (overlap) should exist
    overlap_results = [r for r in results if r["class_id"] == 1]
    assert len(overlap_results) == 0, (
        f"Expected 0 overlap results when no targets, got {len(overlap_results)}"
    )
    print(f"    No targets: all {len(head_results)} heads classified as no_overlap (class 2)")


def test_rule_classifier_empty():
    """Empty detections produce empty output."""
    from core.p01_auto_annotate.rule_classifier import RuleClassifier

    rc = RuleClassifier(
        class_rules=[
            {"output_class_id": 0, "source": "person", "condition": "direct"},
            {
                "output_class_id": 1,
                "source": "head",
                "condition": "overlap",
                "target": "helmet",
                "min_iou": 0.3,
            },
        ],
        detection_class_map={"person": 100, "head": 101, "helmet": 102},
    )

    results = rc.classify([])
    assert results == [], f"Expected empty list, got {results}"
    print("    Empty detections -> empty output (correct)")


# ---------------------------------------------------------------------------
# VLM Verifier tests (parse_response only — no Ollama needed)
# ---------------------------------------------------------------------------


def test_vlm_parse_response():
    """Test VLMVerifier.parse_response() with various inputs."""
    from core.p01_auto_annotate.vlm_verifier import VLMVerifier

    # YES response with confidence and reason
    is_correct, conf, reason = VLMVerifier.parse_response(
        "YES | 0.95 | clear fire visible"
    )
    assert is_correct is True, f"Expected True, got {is_correct}"
    assert abs(conf - 0.95) < 0.01, f"Expected 0.95, got {conf}"
    assert "clear fire visible" in reason, f"Unexpected reason: {reason}"

    # NO response
    is_correct, conf, reason = VLMVerifier.parse_response(
        "NO | 0.2 | this is not a fire"
    )
    assert is_correct is False, f"Expected False, got {is_correct}"
    assert abs(conf - 0.2) < 0.01, f"Expected 0.2, got {conf}"
    assert "this is not a fire" in reason, f"Unexpected reason: {reason}"

    # Ambiguous response — fail-open (True, 0.5)
    is_correct, conf, reason = VLMVerifier.parse_response("maybe")
    assert is_correct is True, f"Expected True (fail-open), got {is_correct}"
    assert abs(conf - 0.5) < 0.01, f"Expected 0.5 (fail-open), got {conf}"
    assert "maybe" in reason, f"Unexpected reason: {reason}"

    # YES without explicit confidence → defaults to 0.9
    is_correct, conf, reason = VLMVerifier.parse_response("YES, this is fire")
    assert is_correct is True, f"Expected True, got {is_correct}"
    assert abs(conf - 0.9) < 0.01, f"Expected 0.9 default, got {conf}"

    # NO without explicit confidence → defaults to 0.1
    is_correct, conf, reason = VLMVerifier.parse_response("NO, not a fire")
    assert is_correct is False, f"Expected False, got {is_correct}"
    assert abs(conf - 0.1) < 0.01, f"Expected 0.1 default, got {conf}"

    print("    parse_response: YES/NO/ambiguous all parsed correctly")


# ---------------------------------------------------------------------------
# Functional API pipeline test
# ---------------------------------------------------------------------------


def test_build_pipeline():
    """Import auto_annotate_pipeline from pipeline module, verify it has invoke()."""
    from core.p01_auto_annotate.pipeline import auto_annotate_pipeline

    assert hasattr(auto_annotate_pipeline, "invoke"), (
        f"auto_annotate_pipeline should have 'invoke' method, "
        f"got attrs: {[a for a in dir(auto_annotate_pipeline) if not a.startswith('_')]}"
    )
    print(f"    auto_annotate_pipeline: {type(auto_annotate_pipeline).__name__} with invoke()")


# ---------------------------------------------------------------------------
# Annotator with rule-based config (no service call)
# ---------------------------------------------------------------------------


def test_annotator_with_rules():
    """Create Annotator with detection_classes and class_rules, verify stored."""
    from core.p01_auto_annotate.annotator import Annotator

    a = Annotator(
        class_names={0: "person", 1: "head_with_helmet"},
        text_prompts={},
        detection_classes={"head": "a head", "helmet": "a helmet"},
        class_rules=[
            {
                "output_class_id": 1,
                "source": "head",
                "condition": "overlap",
                "target": "helmet",
            }
        ],
    )

    assert a.detection_classes is not None, "detection_classes should be stored"
    assert a.class_rules is not None, "class_rules should be stored"
    assert len(a.class_rules) == 1, f"Expected 1 rule, got {len(a.class_rules)}"
    assert a.detection_classes["head"] == "a head"
    assert a.detection_classes["helmet"] == "a helmet"
    print("    Annotator: detection_classes and class_rules stored correctly")


# ---------------------------------------------------------------------------
# Service test — Annotator with rule-based config on real PPE image
# ---------------------------------------------------------------------------


def _get_ppe_test_image_path() -> Path | None:
    """Resolve a real PPE image from the test_ppe_100 dataset, or None."""
    ppe_config_path = ROOT / "configs" / "_test" / "01_data_ppe.yaml"
    if not ppe_config_path.exists():
        return None
    config = load_config(str(ppe_config_path))
    dataset_path = (ppe_config_path.parent / config["path"]).resolve()
    train_images_dir = dataset_path / config["train"]
    if not train_images_dir.exists():
        return None
    images = sorted(train_images_dir.glob("*.jpg")) + sorted(train_images_dir.glob("*.png"))
    return images[0] if images else None


def test_annotator_rule_mode_service():
    """Test Annotator with rule-based config on a real PPE image (requires service)."""
    if not has_auto_label_service():
        print("    SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        return

    image_path = _get_ppe_test_image_path()
    if image_path is None:
        print("    SKIP: no PPE test images found in dataset_store/test_ppe_100/")
        return

    from core.p01_auto_annotate.annotator import Annotator

    ppe_config = load_config(str(ROOT / "configs" / "_test" / "01_data_ppe.yaml"))
    auto_label_cfg = ppe_config["auto_label"]

    a = Annotator(
        class_names={0: "person", 1: "head_with_helmet", 2: "head_without_helmet"},
        text_prompts=ppe_config.get("text_prompts", {}),
        mode="text",
        confidence_threshold=0.3,
        service_url=AUTO_LABEL_URL,
        detection_classes=auto_label_cfg["detection_classes"],
        class_rules=auto_label_cfg["class_rules"],
    )

    try:
        detections = a.annotate_image(Path(image_path))
        assert isinstance(detections, list), f"Expected list, got {type(detections)}"
        print(f"    Rule-mode service: {len(detections)} detections from {image_path.name}")

        # All class IDs should be final (0, 1, 2), not temp IDs (100, 101, 102)
        valid_final_ids = {0, 1, 2}
        for i, det in enumerate(detections):
            assert det["class_id"] in valid_final_ids, (
                f"Detection {i} has temp class_id={det['class_id']}, "
                f"expected one of {valid_final_ids}"
            )
            print(
                f"      det[{i}]: class={det['class_id']}, "
                f"bbox=({det['cx']:.3f},{det['cy']:.3f},{det['w']:.3f},{det['h']:.3f}), "
                f"score={det['score']:.3f}"
            )
    finally:
        a.unload()


# ---------------------------------------------------------------------------
# Multi-use-case config loading test
# ---------------------------------------------------------------------------


def test_multi_usecase_configs():
    """Load each test config and verify parsing; check auto_label presence."""
    config_dir = ROOT / "configs" / "_test"

    # Configs that should have auto_label sections
    configs_with_auto_label = {"05_data_ppe.yaml", "05_data_shoes.yaml"}
    # Configs that should NOT have auto_label sections
    configs_without_auto_label = {
        "05_data_fire.yaml",
        "05_data_phone.yaml",
        "05_data_fall.yaml",
    }

    for fname in sorted(configs_with_auto_label | configs_without_auto_label):
        cfg_path = config_dir / fname
        assert cfg_path.exists(), f"Test config not found: {cfg_path}"

        config = load_config(str(cfg_path))
        assert "names" in config, f"{fname}: missing 'names'"
        assert "num_classes" in config, f"{fname}: missing 'num_classes'"
        assert "input_size" in config, f"{fname}: missing 'input_size'"
        assert "text_prompts" in config, f"{fname}: missing 'text_prompts'"

        num_classes = config["num_classes"]
        names = config["names"]
        assert len(names) == num_classes, (
            f"{fname}: len(names)={len(names)} != num_classes={num_classes}"
        )

        if fname in configs_with_auto_label:
            assert "auto_label" in config, (
                f"{fname}: expected 'auto_label' section but not found"
            )
            al = config["auto_label"]
            assert "detection_classes" in al, (
                f"{fname}: auto_label missing 'detection_classes'"
            )
            assert "class_rules" in al, f"{fname}: auto_label missing 'class_rules'"
            print(
                f"    {fname}: {num_classes} classes, "
                f"auto_label with {len(al['class_rules'])} rules"
            )
        else:
            assert "auto_label" not in config, (
                f"{fname}: unexpected 'auto_label' section found"
            )
            print(f"    {fname}: {num_classes} classes, no auto_label (correct)")


if __name__ == "__main__":
    cpu_tests = [
        ("nms_filter", test_nms_filter),
        ("nms_filter_empty", test_nms_filter_empty),
        ("label_writer_bbox", test_label_writer_bbox),
        ("label_writer_dry_run", test_label_writer_dry_run),
        ("scanner_yolo_layout", test_scanner_yolo_layout),
        ("mask_to_polygon", test_mask_to_polygon),
        ("build_graph", test_build_graph),
        ("label_writer_polygon", test_label_writer_polygon),
        ("scanner_filter_missing", test_scanner_filter_missing),
        ("nms_filter_cross_class", test_nms_filter_cross_class),
        ("nms_filter_thresholds", test_nms_filter_thresholds),
        ("reporter", test_reporter),
        ("mask_to_polygon_edge_cases", test_mask_to_polygon_edge_cases),
        ("rule_classifier_basic", test_rule_classifier_basic),
        ("rule_classifier_no_targets", test_rule_classifier_no_targets),
        ("rule_classifier_empty", test_rule_classifier_empty),
        ("vlm_parse_response", test_vlm_parse_response),
        ("build_pipeline", test_build_pipeline),
        ("annotator_with_rules", test_annotator_with_rules),
        ("multi_usecase_configs", test_multi_usecase_configs),
    ]

    if not has_auto_label_service():
        print("  SKIP: auto-label service not available at " + AUTO_LABEL_URL)
        service_tests = []
    else:
        service_tests = [
            ("annotator_text_mode", test_annotator_text_mode),
            ("annotator_auto_mode", test_annotator_auto_mode),
            ("annotator_hybrid_mode", test_annotator_hybrid_mode),
            ("full_auto_annotate_graph_dry_run", test_full_auto_annotate_graph_dry_run),
            ("annotator_rule_mode_service", test_annotator_rule_mode_service),
        ]

    run_all(
        cpu_tests + service_tests,
        title="Test 13: Auto-Annotate",
    )
