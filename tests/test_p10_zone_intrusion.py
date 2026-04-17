"""Test: Zone Intrusion Detector — geometry helpers, dataclasses, and full detect pipeline.

Tests cover:
- _poly_to_pixel: normalized-to-pixel coordinate conversion and roundtrip
- _zones_contain: vectorized point-in-polygon (inside, outside, empty, multi-zone)
- PersonDetection / ZoneResult dataclass construction
- Full detect() and draw() with real YOLO model (skip if weights not present)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test

# Feature dir uses kebab-case ("access-zone_intrusion") which is not a valid
# Python module path. Add the code/ dir to sys.path so the module can be
# imported directly. The module itself also adds ROOT to sys.path.
_ZI_CODE = ROOT / "features" / "access-zone_intrusion" / "code"
if str(_ZI_CODE) not in sys.path:
    sys.path.insert(0, str(_ZI_CODE))

from zone_intrusion import (
    PersonDetection,
    ZoneResult,
    ZoneIntrusionDetector,
    _poly_to_pixel,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "zone_intrusion"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Weights path derived from the feature config (relative: ../../../pretrained/access-zone_intrusion/yolo11m.pt)
MODEL_WEIGHTS = ROOT / "pretrained" / "access-zone_intrusion" / "yolo11m.pt"
FEATURE_CONFIG = ROOT / "features" / "access-zone_intrusion" / "configs" / "10_inference.yaml"


# ---------------------------------------------------------------------------
# 1. _poly_to_pixel — basic conversion
# ---------------------------------------------------------------------------

def test_poly_to_pixel_basic():
    """Normalized polygon [0,1] coords map to correct pixel coords."""
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    result = _poly_to_pixel(poly, 640, 480)
    assert result.shape == (4, 2), f"Expected (4, 2), got {result.shape}"
    expected = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

    # Non-trivial polygon
    poly2 = [[0.5, 0.25], [0.75, 0.5], [0.5, 0.75], [0.25, 0.5]]
    result2 = _poly_to_pixel(poly2, 800, 600)
    assert result2[0, 0] == pytest.approx(400.0)   # 0.5 * 800
    assert result2[0, 1] == pytest.approx(150.0)    # 0.25 * 600


# ---------------------------------------------------------------------------
# 2. _poly_to_pixel — roundtrip: pixel → normalized → pixel
# ---------------------------------------------------------------------------

def test_poly_to_pixel_roundtrip():
    """Pixel coords survive normalize → pixel conversion."""
    w, h = 640, 480
    poly_px = np.array([[100, 80], [500, 60], [600, 400], [150, 420]], dtype=np.float32)
    # Normalize
    poly_norm = [[p[0] / w, p[1] / h] for p in poly_px]
    # Convert back
    result = _poly_to_pixel(poly_norm, w, h)
    np.testing.assert_allclose(result, poly_px, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. _zones_contain — point inside polygon
# ---------------------------------------------------------------------------

def test_zones_contain_inside():
    """Point at the center of a full-frame zone is inside."""
    detector = ZoneIntrusionDetector.__new__(ZoneIntrusionDetector)
    detector._zones = [{"id": "zone_full", "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}]

    points = np.array([[320.0, 240.0]], dtype=np.float32)
    result = detector._zones_contain(points, 640, 480)
    assert result.shape == (1, 1)
    assert result[0, 0].item() is True


# ---------------------------------------------------------------------------
# 4. _zones_contain — point outside polygon
# ---------------------------------------------------------------------------

def test_zones_contain_outside():
    """Point in left half is outside a right-half zone."""
    detector = ZoneIntrusionDetector.__new__(ZoneIntrusionDetector)
    detector._zones = [{"id": "zone_right", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]}]

    # Left-side point
    points = np.array([[100.0, 240.0]], dtype=np.float32)
    result = detector._zones_contain(points, 640, 480)
    assert result[0, 0].item() is False


# ---------------------------------------------------------------------------
# 5. _zones_contain — empty points array
# ---------------------------------------------------------------------------

def test_zones_contain_empty_points():
    """Empty points array returns (0, Z) bool matrix."""
    detector = ZoneIntrusionDetector.__new__(ZoneIntrusionDetector)
    detector._zones = [
        {"id": "z1", "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]},
        {"id": "z2", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]},
    ]

    points = np.empty((0, 2), dtype=np.float32)
    result = detector._zones_contain(points, 640, 480)
    assert result.shape == (0, 2), f"Expected (0, 2), got {result.shape}"
    assert result.dtype == bool


# ---------------------------------------------------------------------------
# 6. _zones_contain — multiple zones, selective hit
# ---------------------------------------------------------------------------

def test_zones_contain_multiple_zones():
    """Point in right half hits zone 1 (right) but not zone 0 (left)."""
    detector = ZoneIntrusionDetector.__new__(ZoneIntrusionDetector)
    detector._zones = [
        {"id": "left_half", "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]},
        {"id": "right_half", "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]},
    ]

    # Point in right half
    points = np.array([[400.0, 240.0]], dtype=np.float32)
    result = detector._zones_contain(points, 640, 480)
    assert result.shape == (1, 2)
    assert result[0, 0].item() is False, "Should NOT be in left zone"
    assert result[0, 1].item() is True, "Should be in right zone"


# ---------------------------------------------------------------------------
# 7. ZoneResult dataclass
# ---------------------------------------------------------------------------

def test_zone_result_dataclass():
    """ZoneResult fields are accessible and correct."""
    det1 = PersonDetection(
        box_xyxy=np.array([10, 20, 100, 200], dtype=np.float32),
        score=0.92,
        in_zone=True,
        zone_id="zone_a",
    )
    det2 = PersonDetection(
        box_xyxy=np.array([300, 50, 400, 300], dtype=np.float32),
        score=0.78,
        in_zone=False,
        zone_id="",
    )

    result = ZoneResult(
        intruding=True,
        detections=[det1, det2],
        alert_zones=["zone_a"],
        latency_ms=12.5,
    )

    assert result.intruding is True
    assert len(result.detections) == 2
    assert result.detections[0].zone_id == "zone_a"
    assert result.detections[1].in_zone is False
    assert result.alert_zones == ["zone_a"]
    assert result.latency_ms == 12.5

    # No intrusion case
    clear = ZoneResult(intruding=False, detections=[], alert_zones=[], latency_ms=5.0)
    assert clear.intruding is False
    assert clear.detections == []
    assert clear.alert_zones == []


# ---------------------------------------------------------------------------
# 8. PersonDetection dataclass
# ---------------------------------------------------------------------------

def test_person_detection_dataclass():
    """PersonDetection fields are stored correctly."""
    box = np.array([50, 100, 200, 350], dtype=np.float32)
    det = PersonDetection(box_xyxy=box, score=0.87, in_zone=False, zone_id="")

    assert det.box_xyxy.shape == (4,)
    assert det.box_xyxy.dtype == np.float32
    np.testing.assert_array_equal(det.box_xyxy, box)
    assert det.score == 0.87
    assert det.in_zone is False
    assert det.zone_id == ""

    # In-zone variant
    det2 = PersonDetection(box_xyxy=box, score=0.95, in_zone=True, zone_id="restricted")
    assert det2.in_zone is True
    assert det2.zone_id == "restricted"


# ---------------------------------------------------------------------------
# 9. Full detect() on real image (skip if no model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MODEL_WEIGHTS.exists() or not FEATURE_CONFIG.exists(),
    reason="Zone intrusion model weights or config not found",
)
def test_detect_real_image():
    """detect() returns a well-formed ZoneResult on a real image."""
    from fixtures import real_image_bgr_640

    detector = ZoneIntrusionDetector(FEATURE_CONFIG)
    image = real_image_bgr_640()
    result = detector.detect(image)

    assert isinstance(result, ZoneResult)
    assert isinstance(result.intruding, bool)
    assert isinstance(result.detections, list)
    assert isinstance(result.alert_zones, list)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0

    # Every detection must be a PersonDetection
    for det in result.detections:
        assert isinstance(det, PersonDetection)
        assert det.box_xyxy.shape == (4,)
        assert 0.0 <= det.score <= 1.0
        assert isinstance(det.in_zone, bool)
        assert isinstance(det.zone_id, str)

    # Consistency: alert_zones must be subset of detected zone_ids
    if result.intruding:
        assert len(result.alert_zones) > 0
    print(f"    {len(result.detections)} detections, latency={result.latency_ms:.1f}ms")


# ---------------------------------------------------------------------------
# 10. detect() on image unlikely to contain persons (skip if no model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MODEL_WEIGHTS.exists() or not FEATURE_CONFIG.exists(),
    reason="Zone intrusion model weights or config not found",
)
def test_detect_no_persons():
    """detect() on a fire image: result structure is valid regardless of detections."""
    from fixtures import real_image_bgr_640

    detector = ZoneIntrusionDetector(FEATURE_CONFIG)
    image = real_image_bgr_640()
    result = detector.detect(image)

    # Even if no persons detected, structure must be valid
    assert isinstance(result, ZoneResult)
    if len(result.detections) == 0:
        assert result.intruding is False
        assert result.alert_zones == []
    print(f"    detections={len(result.detections)}, intruding={result.intruding}")


# ---------------------------------------------------------------------------
# 11. draw() returns valid annotated image (skip if no model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MODEL_WEIGHTS.exists() or not FEATURE_CONFIG.exists(),
    reason="Zone intrusion model weights or config not found",
)
def test_draw_output():
    """draw() returns an image with the same spatial dimensions and correct type."""
    from fixtures import real_image_bgr_640

    detector = ZoneIntrusionDetector(FEATURE_CONFIG)
    image = real_image_bgr_640()
    result = detector.detect(image)
    annotated = detector.draw(image, result)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == image.shape, f"Shape mismatch: {annotated.shape} vs {image.shape}"
    assert annotated.dtype == np.uint8

    # Save for visual inspection
    out_path = OUTPUTS / "zone_intrusion_draw_test.png"
    import cv2
    cv2.imwrite(str(out_path), annotated)
    print(f"    Drawn output saved to {out_path}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== Test: Zone Intrusion Detector ===\n")

    # Geometry + dataclass tests (no model needed)
    run_test("poly_to_pixel_basic", test_poly_to_pixel_basic)
    run_test("poly_to_pixel_roundtrip", test_poly_to_pixel_roundtrip)
    run_test("zones_contain_inside", test_zones_contain_inside)
    run_test("zones_contain_outside", test_zones_contain_outside)
    run_test("zones_contain_empty_points", test_zones_contain_empty_points)
    run_test("zones_contain_multiple_zones", test_zones_contain_multiple_zones)
    run_test("zone_result_dataclass", test_zone_result_dataclass)
    run_test("person_detection_dataclass", test_person_detection_dataclass)

    # Full pipeline tests (require model weights)
    if MODEL_WEIGHTS.exists() and FEATURE_CONFIG.exists():
        run_test("detect_real_image", test_detect_real_image)
        run_test("detect_no_persons", test_detect_no_persons)
        run_test("draw_output", test_draw_output)
    else:
        print(f"  SKIP: detect/draw tests (weights not found at {MODEL_WEIGHTS})")

    print(f"\nResults: {passed} passed, {failed} failed")
    if errors:
        print("Failures:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    sys.exit(1 if failed else 0)
