"""Test 14: Inference — run detection predictor on real images with trained model."""

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p10_inference.predictor import DetectionPredictor
from utils.config import load_config, resolve_path
from core.p10_inference.video_inference import VideoProcessor

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "14_inference"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
EXPORT_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "12_export"
DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


def _get_test_images(n=5):
    """Get real test images from val set."""
    config = load_config(DATA_CONFIG_PATH)
    base_dir = Path(DATA_CONFIG_PATH).parent
    dataset_path = resolve_path(config["path"], base_dir)
    val_images_dir = dataset_path / config["val"]

    image_paths = sorted(val_images_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return image_paths[:n]


def _get_model_path():
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            return p
    return None


def test_predict_pytorch():
    """Run predictor with PyTorch model on real images."""
    model_path = _get_model_path()
    assert model_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )

    test_images = _get_test_images(5)
    assert len(test_images) > 0, "No test images found"

    for i, img_path in enumerate(test_images):
        image = cv2.imread(str(img_path))
        assert image is not None, f"Failed to read {img_path}"

        preds = predictor.predict(image)
        assert isinstance(preds, dict), f"predict() returned {type(preds)}"
        assert "boxes" in preds, f"Missing 'boxes' key"
        assert "scores" in preds, f"Missing 'scores' key"
        assert "labels" in preds, f"Missing 'labels' key"

        n_dets = len(preds["boxes"])
        print(f"    Image {i}: {img_path.name} — {n_dets} detections")

        # Visualize and save
        vis = predictor.visualize(image, preds)
        out_path = OUTPUTS / f"pytorch_{i:02d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), vis)

    print(f"    Saved {len(test_images)} visualization PNGs to {OUTPUTS}")


def test_predict_onnx():
    """Run predictor with ONNX model on real images."""
    onnx_path = EXPORT_OUTPUTS / "model.onnx"
    if not onnx_path.exists():
        print("  SKIP: model.onnx not found — run test_core17_export.py first")
        return

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(onnx_path),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )

    test_images = _get_test_images(3)
    for i, img_path in enumerate(test_images):
        image = cv2.imread(str(img_path))
        preds = predictor.predict(image)
        assert isinstance(preds, dict)

        n_dets = len(preds["boxes"])
        print(f"    ONNX Image {i}: {img_path.name} — {n_dets} detections")

        vis = predictor.visualize(image, preds)
        out_path = OUTPUTS / f"onnx_{i:02d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), vis)


def test_predict_batch():
    """Test batch prediction on multiple images."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    test_images = _get_test_images(3)
    images = [cv2.imread(str(p)) for p in test_images]
    assert all(img is not None for img in images)

    results = predictor.predict_batch(images)
    assert isinstance(results, list), f"predict_batch returned {type(results)}"
    assert len(results) == len(images), f"Expected {len(images)} results, got {len(results)}"
    print(f"    Batch prediction: {len(results)} images processed")


def test_video_processor_single_frame():
    """Create VideoProcessor, process a single real frame, verify output tuple."""
    model_path = _get_model_path()
    assert model_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )

    processor = VideoProcessor(predictor=predictor)

    test_images = _get_test_images(1)
    assert len(test_images) > 0, "No test images found"

    frame = cv2.imread(str(test_images[0]))
    assert frame is not None, f"Failed to read {test_images[0]}"

    annotated, detections, alerts = processor.process_frame(frame, frame_idx=0)
    assert isinstance(annotated, np.ndarray), f"Expected ndarray, got {type(annotated)}"
    assert isinstance(detections, dict), f"Expected dict, got {type(detections)}"
    assert isinstance(alerts, list), f"Expected list, got {type(alerts)}"
    assert "boxes" in detections, "Missing 'boxes' key in detections"
    print(f"    Frame processed: {len(detections['boxes'])} detections, {len(alerts)} alerts")


def test_predict_different_sizes():
    """Predict on real images resized to different sizes to verify no crash."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    # Use a real image resized to different dimensions
    test_images = _get_test_images(1)
    assert len(test_images) > 0, "No test images found"
    original = cv2.imread(str(test_images[0]))
    assert original is not None, f"Failed to read {test_images[0]}"

    # Small: 320x240
    small = cv2.resize(original, (320, 240))
    preds_small = predictor.predict(small)
    assert isinstance(preds_small, dict), f"predict() returned {type(preds_small)}"
    assert "boxes" in preds_small, "Missing 'boxes' key for small image"
    print(f"    Small (320x240): {len(preds_small['boxes'])} detections")

    # Large: 1280x720
    large = cv2.resize(original, (1280, 720))
    preds_large = predictor.predict(large)
    assert isinstance(preds_large, dict), f"predict() returned {type(preds_large)}"
    assert "boxes" in preds_large, "Missing 'boxes' key for large image"
    print(f"    Large (1280x720): {len(preds_large['boxes'])} detections")


def test_visualize_output():
    """Verify visualize returns image with same H,W as input."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    test_images = _get_test_images(1)
    assert len(test_images) > 0, "No test images found"

    image = cv2.imread(str(test_images[0]))
    assert image is not None, f"Failed to read {test_images[0]}"

    preds = predictor.predict(image)
    vis = predictor.visualize(image, preds)

    assert isinstance(vis, np.ndarray), f"Expected ndarray, got {type(vis)}"
    assert vis.shape[:2] == image.shape[:2], (
        f"H,W mismatch: vis {vis.shape[:2]} vs input {image.shape[:2]}"
    )
    print(f"    Visualize output shape: {vis.shape} (matches input)")


def test_alert_immediate_trigger():
    """Test that fire detection triggers an immediate alert."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    alert_config = {
        "confidence_thresholds": {"fire": 0.5},
        "frame_windows": {},  # no window = immediate
        "window_ratio": 0.8,
        "cooldown_frames": 10,
    }
    processor = VideoProcessor(predictor=predictor, alert_config=alert_config)
    processor._reset_state()

    # Synthetic detection: fire with high confidence
    detections = {
        "boxes": np.array([[10, 10, 100, 100]]),
        "scores": np.array([0.9]),
        "labels": np.array([0]),
        "class_names": ["fire"],
    }

    alerts = processor._check_alerts(detections, frame_idx=0)
    assert len(alerts) > 0, "Fire should trigger immediate alert"
    assert alerts[0]["type"] == "fire"
    print(f"    Immediate alert triggered: {alerts[0]['message']}")


def test_alert_cooldown():
    """Test that cooldown suppresses repeated alerts."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    alert_config = {
        "confidence_thresholds": {"fire": 0.5},
        "frame_windows": {},
        "window_ratio": 0.8,
        "cooldown_frames": 10,
    }
    processor = VideoProcessor(predictor=predictor, alert_config=alert_config)
    processor._reset_state()

    detections = {
        "boxes": np.array([[10, 10, 100, 100]]),
        "scores": np.array([0.9]),
        "labels": np.array([0]),
        "class_names": ["fire"],
    }

    # First alert at frame 0
    alerts1 = processor._check_alerts(detections, frame_idx=0)
    assert len(alerts1) > 0, "First detection should trigger alert"

    # Same detection at frame 5 (within cooldown of 10)
    alerts2 = processor._check_alerts(detections, frame_idx=5)
    assert len(alerts2) == 0, "Alert within cooldown should be suppressed"

    # Same detection at frame 15 (after cooldown)
    alerts3 = processor._check_alerts(detections, frame_idx=15)
    assert len(alerts3) > 0, "Alert after cooldown should fire again"
    print(f"    Cooldown: alert at 0, suppressed at 5, fired again at 15")


def test_alert_window_based():
    """Test window-based alert requires sustained violations."""
    model_path = _get_model_path()
    assert model_path is not None, "No checkpoint found"

    data_config = load_config(DATA_CONFIG_PATH)
    predictor = DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
    )

    alert_config = {
        "confidence_thresholds": {"helmet_violation": 0.5},
        "frame_windows": {"helmet_violation": 5},  # need 5 frames
        "window_ratio": 0.8,  # 80% of 5 = need 4 violations
        "cooldown_frames": 100,
    }
    processor = VideoProcessor(predictor=predictor, alert_config=alert_config)
    processor._reset_state()

    violation = {
        "boxes": np.array([[10, 10, 100, 100]]),
        "scores": np.array([0.9]),
        "labels": np.array([0]),
        "class_names": ["helmet_violation"],
    }
    # Send 3 violations - not enough
    alerts = []
    for i in range(3):
        alerts = processor._check_alerts(violation, frame_idx=i)
    assert len(alerts) == 0, "3 frames should not trigger window-based alert"

    # Send 2 more violations (total 5 consecutive)
    alerts_at_4 = processor._check_alerts(violation, frame_idx=3)
    alerts_at_5 = processor._check_alerts(violation, frame_idx=4)

    # At least one of the last two should trigger
    triggered = len(alerts_at_4) > 0 or len(alerts_at_5) > 0
    assert triggered, "5 consecutive violations should trigger window-based alert"
    print(f"    Window-based: triggered after 5 consecutive violations")


# ---------------------------------------------------------------------------
# ModelAdapter tests
# ---------------------------------------------------------------------------

from core.p10_inference.model_adapter import (  # noqa: E402
    ModelAdapter,
    PredictorAdapter,
    HFAdapter,
    TorchScriptAdapter,
    register_adapter,
    resolve_adapter,
)


def test_adapter_registry_can_handle():
    """PredictorAdapter matches .pth/.pt/.onnx; HFAdapter matches HF ids."""
    data_config = load_config(DATA_CONFIG_PATH)

    assert PredictorAdapter.can_handle("model.pth", data_config, None)
    assert PredictorAdapter.can_handle("model.pt", data_config, None)
    assert PredictorAdapter.can_handle("model.onnx", data_config, None)
    assert not PredictorAdapter.can_handle("PekingU/rtdetr_v2_r18vd", data_config, None)

    assert HFAdapter.can_handle("PekingU/rtdetr_v2_r18vd", data_config, None)
    assert not HFAdapter.can_handle("model.pth", data_config, None)
    assert not HFAdapter.can_handle("model.onnx", data_config, None)

    # TorchScriptAdapter is intentionally never matched
    assert not TorchScriptAdapter.can_handle("model.pt", data_config, None)

    print("    can_handle() dispatch: all assertions passed")


def test_adapter_resolve_pth():
    """resolve_adapter returns PredictorAdapter for a .pth path."""
    model_path = _get_model_path()
    if model_path is None:
        print("  SKIP: no checkpoint found — run test_p06_training.py first")
        return

    data_config = load_config(DATA_CONFIG_PATH)
    adapter = resolve_adapter(str(model_path), data_config, None, conf_threshold=0.1)
    assert isinstance(adapter, PredictorAdapter), f"Expected PredictorAdapter, got {type(adapter)}"
    print(f"    resolve_adapter({model_path.suffix}) → {type(adapter).__name__}")


def test_adapter_predict_batch_pth():
    """PredictorAdapter.predict_batch returns correct dict structure."""
    model_path = _get_model_path()
    if model_path is None:
        print("  SKIP: no checkpoint found — run test_p06_training.py first")
        return

    data_config = load_config(DATA_CONFIG_PATH)
    adapter = resolve_adapter(str(model_path), data_config, None, conf_threshold=0.1)

    images = [cv2.imread(str(p)) for p in _get_test_images(3)]
    assert all(img is not None for img in images)

    results = adapter.predict_batch(images)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == len(images)

    for i, r in enumerate(results):
        assert "boxes" in r and "scores" in r and "labels" in r, f"Missing keys in result {i}"
        assert r["boxes"].dtype == np.float32, f"boxes dtype: {r['boxes'].dtype}"
        assert r["scores"].dtype == np.float32, f"scores dtype: {r['scores'].dtype}"
        assert r["labels"].dtype == np.int64, f"labels dtype: {r['labels'].dtype}"
        n = len(r["boxes"])
        assert r["scores"].shape == (n,), f"scores shape mismatch"
        assert r["labels"].shape == (n,), f"labels shape mismatch"
        print(f"    Image {i}: {n} detections")


def test_adapter_predict_batch_onnx():
    """PredictorAdapter wraps .onnx correctly (single-image batch — exported ONNX has static batch=1)."""
    onnx_path = EXPORT_OUTPUTS / "model.onnx"
    if not onnx_path.exists():
        print("  SKIP: model.onnx not found — run test_p09_export.py first")
        return

    data_config = load_config(DATA_CONFIG_PATH)
    adapter = resolve_adapter(str(onnx_path), data_config, None, conf_threshold=0.1)
    assert isinstance(adapter, PredictorAdapter)

    images = [cv2.imread(str(p)) for p in _get_test_images(2)]
    total = 0
    for img in images:
        results = adapter.predict_batch([img])
        assert len(results) == 1
        r = results[0]
        assert {"boxes", "scores", "labels"} <= r.keys()
        total += len(r["boxes"])
    print(f"    ONNX adapter: {total} total detections across {len(images)} images")


def test_adapter_custom_registration():
    """Custom adapter registered with @register_adapter is picked up by resolve_adapter."""
    import copy  # noqa: PLC0415
    from core.p10_inference import model_adapter as _mod  # noqa: PLC0415

    original_registry = copy.copy(_mod._ADAPTER_REGISTRY)

    @register_adapter
    class _DummyAdapter(ModelAdapter):
        _called = False

        def __init__(self, model_path, data_config, training_config,
                     conf_threshold, iou_threshold, device) -> None:
            _DummyAdapter._called = True

        @classmethod
        def can_handle(cls, model_path, data_config, training_config) -> bool:
            return model_path == "__dummy__"

        def predict_batch(self, images):
            return [{"boxes": np.zeros((0, 4), np.float32),
                     "scores": np.zeros(0, np.float32),
                     "labels": np.zeros(0, np.int64)} for _ in images]

    try:
        data_config = load_config(DATA_CONFIG_PATH)
        adapter = resolve_adapter("__dummy__", data_config, None)
        assert isinstance(adapter, _DummyAdapter), f"Got {type(adapter)}"
        assert _DummyAdapter._called
        print("    Custom @register_adapter picked up by resolve_adapter correctly")
    finally:
        _mod._ADAPTER_REGISTRY[:] = original_registry


def test_adapter_unknown_format_raises():
    """resolve_adapter raises ValueError for unrecognised formats."""
    import importlib  # noqa: PLC0415
    from core.p10_inference import model_adapter as _mod  # noqa: PLC0415
    import copy  # noqa: PLC0415

    data_config = load_config(DATA_CONFIG_PATH)
    original = copy.copy(_mod._ADAPTER_REGISTRY)
    # Temporarily clear registry to guarantee no match
    _mod._ADAPTER_REGISTRY.clear()
    try:
        try:
            resolve_adapter("some_unknown.xyz", data_config, None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No adapter can handle" in str(e)
            print(f"    ValueError raised correctly: {e}")
    finally:
        _mod._ADAPTER_REGISTRY[:] = original


if __name__ == "__main__":
    run_all([
        ("predict_pytorch", test_predict_pytorch),
        ("predict_onnx", test_predict_onnx),
        ("predict_batch", test_predict_batch),
        ("video_processor_single_frame", test_video_processor_single_frame),
        ("predict_different_sizes", test_predict_different_sizes),
        ("visualize_output", test_visualize_output),
        ("alert_immediate_trigger", test_alert_immediate_trigger),
        ("alert_cooldown", test_alert_cooldown),
        ("alert_window_based", test_alert_window_based),
        ("adapter_registry_can_handle", test_adapter_registry_can_handle),
        ("adapter_resolve_pth", test_adapter_resolve_pth),
        ("adapter_predict_batch_pth", test_adapter_predict_batch_pth),
        ("adapter_predict_batch_onnx", test_adapter_predict_batch_onnx),
        ("adapter_custom_registration", test_adapter_custom_registration),
        ("adapter_unknown_format_raises", test_adapter_unknown_format_raises),
    ], title="Test 08: Inference")
