"""Test 15: Video Inference — VideoProcessor init, frame processing, alert config."""

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p10_inference.predictor import DetectionPredictor
from core.p10_inference.video_inference import VideoProcessor
from utils.config import load_config, resolve_path

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "15_video_inference"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


def _get_model_path() -> Path | None:
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            return p
    return None


def _build_predictor() -> DetectionPredictor:
    model_path = _get_model_path()
    assert model_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )
    data_config = load_config(DATA_CONFIG_PATH)
    return DetectionPredictor(
        model_path=str(model_path),
        data_config=data_config,
        conf_threshold=0.1,
        iou_threshold=0.45,
    )


def _get_test_images(n: int = 5) -> list[Path]:
    """Get real test images from val set."""
    config = load_config(DATA_CONFIG_PATH)
    base_dir = Path(DATA_CONFIG_PATH).parent
    dataset_path = resolve_path(config["path"], base_dir)
    val_images_dir = dataset_path / config["val"]

    image_paths = sorted(val_images_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return image_paths[:n]


_has_checkpoint = _get_model_path() is not None
_skip_reason = "No checkpoint — run test_core13_training.py first"


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_video_processor_init():
    """Initialize VideoProcessor with a real model checkpoint."""
    predictor = _build_predictor()
    processor = VideoProcessor(predictor=predictor)

    assert processor.predictor is predictor
    assert isinstance(processor.alert_config, dict)
    assert "confidence_thresholds" in processor.alert_config
    assert "frame_windows" in processor.alert_config
    assert "cooldown_frames" in processor.alert_config
    print(f"    VideoProcessor initialized with alert classes: "
          f"{list(processor.alert_config['confidence_thresholds'].keys())}")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_process_single_frame():
    """Process a single real image frame through the video pipeline."""
    from fixtures import real_image

    predictor = _build_predictor()
    processor = VideoProcessor(predictor=predictor)

    frame = real_image(idx=0, split="val")
    assert frame is not None and frame.size > 0, "Failed to load real image"

    annotated, detections, alerts = processor.process_frame(frame, frame_idx=0)

    # Validate annotated frame
    assert isinstance(annotated, np.ndarray), f"Expected ndarray, got {type(annotated)}"
    assert annotated.shape[:2] == frame.shape[:2], (
        f"Annotated shape {annotated.shape[:2]} != input {frame.shape[:2]}"
    )
    assert annotated.dtype == np.uint8, f"Expected uint8, got {annotated.dtype}"

    # Validate detections dict
    assert isinstance(detections, dict), f"Expected dict, got {type(detections)}"
    assert "boxes" in detections, "Missing 'boxes' key"
    assert "scores" in detections, "Missing 'scores' key"
    assert "labels" in detections, "Missing 'labels' key"

    # Validate alerts list
    assert isinstance(alerts, list), f"Expected list, got {type(alerts)}"

    n_dets = len(detections["boxes"])
    print(f"    Frame processed: {n_dets} detections, {len(alerts)} alerts")
    print(f"    Annotated frame shape: {annotated.shape}")

    # Save annotated frame
    out_path = OUTPUTS / "single_frame_annotated.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"    Saved: {out_path}")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_process_multiple_frames():
    """Process multiple real frames sequentially through the video pipeline."""
    predictor = _build_predictor()
    processor = VideoProcessor(predictor=predictor)

    test_images = _get_test_images(3)
    assert len(test_images) > 0, "No test images found"

    total_dets = 0
    total_alerts = 0
    for i, img_path in enumerate(test_images):
        frame = cv2.imread(str(img_path))
        assert frame is not None, f"Failed to read {img_path}"

        annotated, detections, alerts = processor.process_frame(frame, frame_idx=i)
        n_dets = len(detections["boxes"])
        total_dets += n_dets
        total_alerts += len(alerts)

        out_path = OUTPUTS / f"multi_frame_{i:02d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)

    print(f"    Processed {len(test_images)} frames: {total_dets} total detections, "
          f"{total_alerts} total alerts")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_alert_config_loading():
    """Test that alert configs are loaded from dict, not hardcoded."""
    predictor = _build_predictor()

    # Custom alert config with non-default values
    custom_alert_config = {
        "confidence_thresholds": {"fire": 0.3, "custom_class": 0.8},
        "frame_windows": {"custom_class": 10},
        "window_ratio": 0.6,
        "cooldown_frames": 50,
    }

    processor = VideoProcessor(predictor=predictor, alert_config=custom_alert_config)

    # Verify custom config is applied, not defaults
    assert processor._conf_thresholds["fire"] == 0.3, (
        f"Expected fire threshold 0.3, got {processor._conf_thresholds['fire']}"
    )
    assert "custom_class" in processor._conf_thresholds, (
        "Custom class not found in confidence thresholds"
    )
    assert processor._conf_thresholds["custom_class"] == 0.8
    assert processor._frame_windows.get("custom_class") == 10
    assert processor._window_ratio == 0.6
    assert processor._cooldown_frames == 50
    print("    Custom alert config applied correctly")

    # Verify default config is used when None
    processor_default = VideoProcessor(predictor=predictor, alert_config=None)
    assert processor_default._cooldown_frames == 90, (
        f"Expected default cooldown 90, got {processor_default._cooldown_frames}"
    )
    print("    Default alert config applied when alert_config=None")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_video_processor_with_tracking():
    """Initialize VideoProcessor with tracking enabled."""
    predictor = _build_predictor()

    # Tracking may fail if bytetrack is not installed — that's OK, test graceful handling
    processor = VideoProcessor(predictor=predictor, enable_tracking=True)

    # If tracking init succeeded, verify tracker exists
    if processor.enable_tracking:
        assert processor._tracker is not None, "Tracker should exist when tracking is enabled"
        print(f"    Tracking enabled with tracker: {type(processor._tracker).__name__}")
    else:
        print("    Tracking disabled (tracker init failed gracefully)")

    # Process a frame either way
    from fixtures import real_image
    frame = real_image(idx=0, split="val")
    annotated, detections, alerts = processor.process_frame(frame, frame_idx=0)
    assert isinstance(annotated, np.ndarray)
    print(f"    Frame processed with tracking={'on' if processor.enable_tracking else 'off'}: "
          f"{len(detections['boxes'])} detections")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_reset_state():
    """Verify _reset_state clears counters between video runs."""
    predictor = _build_predictor()
    processor = VideoProcessor(predictor=predictor)

    # Process a frame to populate state
    from fixtures import real_image
    frame = real_image(idx=0, split="val")
    processor.process_frame(frame, frame_idx=0)

    # Reset and verify clean state
    processor._reset_state()
    assert processor._frame_count == 0, (
        f"frame_count={processor._frame_count} after reset"
    )
    print(f"    State reset verified: frame_count={processor._frame_count}")


if __name__ == "__main__":
    run_all([
        ("video_processor_init", test_video_processor_init),
        ("process_single_frame", test_process_single_frame),
        ("process_multiple_frames", test_process_multiple_frames),
        ("alert_config_loading", test_alert_config_loading),
        ("video_processor_with_tracking", test_video_processor_with_tracking),
        ("reset_state", test_reset_state),
    ], title="Test 11: Video Inference")
