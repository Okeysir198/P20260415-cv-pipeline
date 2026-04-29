"""Pose-backend abstraction for the point-and-call orchestrator.

Backends are dispatched by ``cfg["backend"]``:
  * ``dwpose_onnx`` -> in-module DWPose ONNX adapter (CUDA EP, optional CPU
    fallback only on session-creation error).
  * ``rtmpose`` / ``mediapipe`` / ``hf_keypoint`` -> generic adapter that
    delegates to ``core.p10_inference.pose_predictor.PosePredictor`` loaded
    from a sibling YAML referenced via ``cfg["config"]``.

All backends return ``list[PoseSample]`` where each sample is
``(kpts_17xy: (17, 2) float32, scores_17: (17,) float32, person_box_xyxy: (4,)
float32)`` in original-image coords.

Project rule: ``code/`` may import from ``core/`` and ``utils/`` but NOT
from another feature's ``code/`` directory. The DWPose / person-detector
helpers below are deliberately copied (not imported) from the
``safety-poketenashi`` reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

# (kpts_17xy, scores_17, person_box_xyxy)
PoseSample = tuple[np.ndarray, np.ndarray, np.ndarray]

_REPO = Path(__file__).resolve().parents[3]


# COCO body slice when reading from a 133-keypoint wholebody output.
_WB_BODY = slice(0, 17)


class PoseBackend(Protocol):
    def __call__(self, image_bgr: np.ndarray) -> list[PoseSample]:
        ...


# ---------------------------------------------------------------------------
# Person detector (lazy YOLO11n; whole-frame fallback)
# ---------------------------------------------------------------------------

class _PersonDetector:
    """Lazy YOLO11n person detector, falls back to whole-frame box.

    Mirrors ``_detect_persons`` in ``safety-poketenashi/code/orchestrator.py``
    (copied per project rule -- ``code/`` may not cross-import from another
    feature's ``code/``).
    """

    _DEFAULT_PT = _REPO / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"

    def __init__(self, weights_path: Path | None = None, conf: float = 0.35) -> None:
        self._weights = Path(weights_path) if weights_path else self._DEFAULT_PT
        self._conf = float(conf)
        self._model: Any = None
        self._tried_load = False

    def _ensure_loaded(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        if not self._weights.exists():
            self._model = None
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights))
        except Exception:
            self._model = None

    def __call__(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        whole = [np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32)]
        if self._model is None:
            return whole
        try:
            det = self._model.predict(
                image_bgr, classes=[0], conf=self._conf, verbose=False
            )[0]
        except Exception:
            return whole
        if det.boxes is None or len(det.boxes) == 0:
            return whole
        return [b.astype(np.float32) for b in det.boxes.xyxy.cpu().numpy()]


# ---------------------------------------------------------------------------
# DWPose ONNX adapter
# ---------------------------------------------------------------------------

class _DWPoseAdapter:
    """DWPose top-down ONNX adapter (RTMPose SimCC head).

    Copied from ``safety-poketenashi/code/orchestrator.py::_DWPose`` to
    keep this feature self-contained. CUDA-first; falls back to CPU EP
    only if session creation under CUDA raises.
    """

    INPUT_HW = (384, 288)  # (H, W)

    def __init__(self, onnx_path: Path, person_detector: _PersonDetector) -> None:
        import onnxruntime as ort

        try:
            self._sess = ort.InferenceSession(
                str(onnx_path), providers=["CUDAExecutionProvider"]
            )
        except Exception:
            self._sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        self._in_name = self._sess.get_inputs()[0].name
        self._person_detector = person_detector

    def _affine(self, box_xyxy: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = box_xyxy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = x1 - x0, y1 - y0
        oh, ow = self.INPUT_HW
        aspect = ow / oh
        if bw / (bh + 1e-9) > aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        bw *= 1.25
        bh *= 1.25
        src = np.array(
            [[cx, cy], [cx + bw / 2, cy], [cx, cy + bh / 2]], dtype=np.float32
        )
        dst = np.array(
            [[ow / 2, oh / 2], [ow, oh / 2], [ow / 2, oh]], dtype=np.float32
        )
        return cv2.getAffineTransform(src, dst)

    def _infer_one(
        self, img_bgr: np.ndarray, box_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        oh, ow = self.INPUT_HW
        M = self._affine(box_xyxy)
        crop = cv2.warpAffine(img_bgr, M, (ow, oh), flags=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        x = (crop.astype(np.float32) - mean) / std
        x = x.transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self._sess.run(None, {self._in_name: x})
        sx = simcc_x[0].argmax(axis=-1).astype(np.float32) / 2.0
        sy = simcc_y[0].argmax(axis=-1).astype(np.float32) / 2.0
        scores = np.minimum(simcc_x[0].max(axis=-1), simcc_y[0].max(axis=-1))
        Minv = cv2.invertAffineTransform(M)
        ones = np.ones((sx.shape[0], 1), dtype=np.float32)
        pts_in = np.concatenate([sx[:, None], sy[:, None], ones], axis=1)
        pts_orig = pts_in @ Minv.T
        return pts_orig.astype(np.float32), scores.astype(np.float32)

    def __call__(self, image_bgr: np.ndarray) -> list[PoseSample]:
        out: list[PoseSample] = []
        for box in self._person_detector(image_bgr):
            kpts, scores = self._infer_one(image_bgr, box)
            out.append((kpts[_WB_BODY], scores[_WB_BODY], box.astype(np.float32)))
        return out


# ---------------------------------------------------------------------------
# Generic adapter (rtmpose / mediapipe / hf_keypoint via core.p10_inference)
# ---------------------------------------------------------------------------

class _GenericPoseAdapter:
    """Delegate to ``core.p10_inference.pose_predictor.PosePredictor``.

    The wrapped predictor takes whole-frame BGR and internally runs its own
    person detector + pose model -- so the adapter does NOT need to feed
    person boxes externally. This matches ``PosePredictor.predict()`` in
    ``core/p10_inference/pose_predictor.py``.

    If predictor construction fails (e.g. missing arch / weights), the
    adapter raises at build time so ``build_pose_backend`` surfaces the
    error to the caller.
    """

    def __init__(self, predictor_cfg_path: Path) -> None:
        # Lazy import: unrelated backends shouldn't pull torch / hf at module load.
        import sys

        sys.path.insert(0, str(_REPO))
        from core.p06_models.pose_registry import build_pose_model
        from core.p10_inference.pose_predictor import PosePredictor
        from core.p10_inference.predictor import DetectionPredictor
        from utils.config import load_config

        cfg = load_config(predictor_cfg_path)
        det_cfg = cfg.get("person_detector", {}) or {}
        det_weights = det_cfg.get("weights") or str(
            _REPO / "pretrained" / "access-zone_intrusion" / "yolo11n.pt"
        )
        self._predictor = PosePredictor(
            detector=DetectionPredictor(model_path=det_weights),
            pose_model=build_pose_model(cfg),
        )

    def __call__(self, image_bgr: np.ndarray) -> list[PoseSample]:
        results = self._predictor.predict_coco(image_bgr)
        boxes = np.asarray(results.get("boxes", []), dtype=np.float32)
        kpts = np.asarray(results.get("keypoints", []), dtype=np.float32)
        if boxes.size == 0 or kpts.size == 0:
            return []
        out: list[PoseSample] = []
        n = min(boxes.shape[0], kpts.shape[0])
        for i in range(n):
            kp_i = kpts[i]
            xy = kp_i[:, :2].astype(np.float32)
            scores = (
                kp_i[:, 2].astype(np.float32)
                if kp_i.shape[-1] >= 3
                else np.ones(kp_i.shape[0], dtype=np.float32)
            )
            out.append((xy[:17], scores[:17], boxes[i].astype(np.float32)))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path, base_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def build_pose_backend(cfg: dict) -> PoseBackend:
    """Build a pose backend from the ``pose:`` block of an inference YAML.

    Recognised ``cfg["backend"]`` values:
      * ``dwpose_onnx`` -- requires ``cfg["weights"]`` (path to .onnx).
      * ``rtmpose``, ``mediapipe``, ``hf_keypoint`` -- requires
        ``cfg["config"]`` (path to a sibling YAML consumed by
        ``core.p10_inference.pose_predictor.PosePredictor``).

    Relative paths are resolved against ``cfg["_config_dir"]`` if set,
    else the current working directory. ``cfg["_config_dir"]`` is injected
    by the orchestrator after loading ``10_inference.yaml``.
    """
    if not isinstance(cfg, dict):
        raise ValueError("pose backend config must be a dict")

    backend = cfg.get("backend")
    if not backend:
        raise ValueError("pose backend config missing 'backend' key")

    base_dir = Path(cfg.get("_config_dir", Path.cwd()))

    if backend == "dwpose_onnx":
        weights = cfg.get("weights")
        if not weights:
            raise ValueError("dwpose_onnx backend requires 'weights' (path to .onnx)")
        onnx_path = _resolve_path(weights, base_dir)
        if not onnx_path.exists():
            raise FileNotFoundError(f"DWPose ONNX not found: {onnx_path}")
        det_weights_cfg = cfg.get("person_detector", {}) or {}
        det_weights = det_weights_cfg.get("weights")
        det_weights_path = (
            _resolve_path(det_weights, base_dir) if det_weights else None
        )
        det = _PersonDetector(weights_path=det_weights_path,
                              conf=float(det_weights_cfg.get("conf", 0.35)))
        return _DWPoseAdapter(onnx_path=onnx_path, person_detector=det)

    if backend in {"rtmpose", "mediapipe", "hf_keypoint"}:
        cfg_yaml = cfg.get("config")
        if not cfg_yaml:
            raise ValueError(
                f"{backend} backend requires 'config' (path to pose YAML)"
            )
        return _GenericPoseAdapter(
            predictor_cfg_path=_resolve_path(cfg_yaml, base_dir),
        )

    raise ValueError(f"unknown pose backend: {backend}")
