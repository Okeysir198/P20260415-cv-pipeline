"""Synthetic-keypoint tests for StairSafetyDetector.

Bypasses pose model + person detector — drives the rule directly with
hand-crafted COCO-17 hip positions. The rule assumes the stair axis is
roughly horizontal in the camera frame; walking *along* the stairs therefore
shows as a near-horizontal hip-midpoint trajectory. "Straight up the stairs"
in this rule's frame of reference = horizontal in pixel space.

Cases:
  1. Straight-along-axis (horizontal) trajectory ≠ triggered (0° deviation).
  2. Diagonal (45°) trajectory triggers once min_frames is reached.
  3. Fewer than min_frames → not triggered (still buffering).
  4. reset() clears state — buffer empty + first sample re-enters buffering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

from stair_safety_detector import StairSafetyDetector  # noqa: E402


def _kpts_with_hips(x: float, y: float) -> np.ndarray:
    """COCO-17 zero array with both hips planted at (x, y)."""
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[11] = [x, y]      # left hip
    kpts[12] = [x + 10, y]  # right hip (mid-hip ≈ x+5)
    return kpts


def _full_scores() -> np.ndarray:
    return np.full(17, 0.9, dtype=np.float32)


def test_straight_up_trajectory_does_not_trigger() -> None:
    """Walking straight up the stairs (= along axis = horizontal in camera frame)
    has 0° deviation from horizontal → below 20° threshold."""
    detector = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=5)
    triggered_any = False
    for i in range(7):
        kpts = _kpts_with_hips(100 + i * 50, 200)  # constant y → horizontal
        result = detector.check(kpts, _full_scores())
        if result.triggered:
            triggered_any = True
    assert not triggered_any, "horizontal trajectory must not fire stair_diagonal"


def test_diagonal_trajectory_triggers() -> None:
    """45° diagonal exceeds the 20° threshold once buffer has min_frames."""
    detector = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=5)
    fired_at = -1
    for i in range(7):
        kpts = _kpts_with_hips(100 + i * 50, 200 + i * 50)  # 45° down-right
        result = detector.check(kpts, _full_scores())
        if result.triggered and fired_at < 0:
            fired_at = i
    # First valid evaluation is at i = min_frames - 1 (5th sample = index 4).
    assert fired_at == 4, f"expected first trigger at frame 4, got {fired_at}"


def test_fewer_than_min_frames_not_triggered() -> None:
    """Below min_frames, the rule is still buffering — never fires."""
    detector = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=5)
    for i in range(4):  # 4 < min_frames=5
        kpts = _kpts_with_hips(100 + i * 50, 200 + i * 50)  # 45° diagonal
        result = detector.check(kpts, _full_scores())
        assert not result.triggered
        assert "buffering" in result.debug_info.get("reason", "")


def test_reset_clears_state() -> None:
    """After reset() the buffer is empty — same diagonal must rebuild before firing."""
    detector = StairSafetyDetector(max_diagonal_angle_deg=20.0, min_frames=5)
    # Push 5 diagonal frames → fires on 5th.
    for i in range(5):
        detector.check(_kpts_with_hips(100 + i * 50, 200 + i * 50), _full_scores())
    pre_reset_buf = len(detector._hip_positions)
    assert pre_reset_buf == 5

    detector.reset()
    assert len(detector._hip_positions) == 0

    # First post-reset sample must be in buffering mode (not triggered).
    result = detector.check(_kpts_with_hips(100, 200), _full_scores())
    assert not result.triggered
    assert "buffering" in result.debug_info.get("reason", "")


if __name__ == "__main__":
    test_straight_up_trajectory_does_not_trigger()
    test_diagonal_trajectory_triggers()
    test_fewer_than_min_frames_not_triggered()
    test_reset_clears_state()
    print("All 4 tests passed.")
