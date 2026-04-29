"""Tests for the pose-backend factory."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_FEAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FEAT / "code"))

from pose_backend import PoseBackend, build_pose_backend  # noqa: E402


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError):
        build_pose_backend({"backend": "totally_made_up"})


def test_missing_backend_key_raises():
    with pytest.raises(ValueError):
        build_pose_backend({})


def test_dwpose_onnx_missing_weights_raises():
    with pytest.raises(ValueError):
        build_pose_backend({"backend": "dwpose_onnx"})


def test_dwpose_onnx_missing_weights_file_raises():
    with pytest.raises(FileNotFoundError):
        build_pose_backend({
            "backend": "dwpose_onnx",
            "weights": "/no/such/path/model.onnx",
        })


def test_generic_backend_missing_config_raises():
    for backend in ("rtmpose", "mediapipe", "hf_keypoint"):
        with pytest.raises(ValueError):
            build_pose_backend({"backend": backend})


class _MockBackend:
    """Concrete implementation of the PoseBackend protocol for shape checking."""

    def __call__(self, image_bgr: np.ndarray):
        return [(
            np.zeros((17, 2), dtype=np.float32),
            np.zeros(17, dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )]


def test_protocol_shape_is_callable():
    mock: PoseBackend = _MockBackend()
    out = mock(np.zeros((10, 10, 3), dtype=np.uint8))
    assert isinstance(out, list)
    assert len(out) == 1
    kpts, scores, box = out[0]
    assert kpts.shape == (17, 2)
    assert scores.shape == (17,)
    assert box.shape == (4,)
