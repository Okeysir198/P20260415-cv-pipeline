"""SCRFD face detection using ONNX Runtime inference.

Anchor-free face detector from InsightFace with 5-point landmark output.
Supports SCRFD-500M (lightweight) and SCRFD-2.5G (higher accuracy) variants
via the face detector registry.

Typical usage::

    from core.p06_models.face_registry import build_face_detector

    config = {
        "face_detector": {
            "arch": "scrfd-500m",
            "model_path": "pretrained/scrfd_500m.onnx",
        }
    }
    detector = build_face_detector(config)
    result = detector.detect_faces(image, bbox)
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort

from core.p06_models.face_base import FaceDetector
from core.p06_models.face_registry import _FACE_DETECTOR_VARIANT_MAP, register_face_detector

logger = logging.getLogger(__name__)

# Default ONNX paths per variant
_DEFAULT_ONNX_PATHS: dict[str, str] = {
    "scrfd-500m": "pretrained/scrfd_500m.onnx",
    "scrfd-2.5g": "pretrained/scrfd_2.5g.onnx",
}

# SCRFD normalization constants
_SCRFD_MEAN = 127.5
_SCRFD_SCALE = 1.0 / 128.0

# Stride levels for anchor generation
_STRIDES = (8, 16, 32)


class SCRFDModel(FaceDetector):
    """SCRFD face detector using ONNX Runtime.

    Detects faces within a cropped region of the full image (typically a
    person/head bounding box from the violation detector). Returns bounding
    boxes, confidence scores, and 5-point facial landmarks mapped back to
    full image coordinates.

    Args:
        model_path: Path to the ``.onnx`` model file.
        input_h: Network input height (default 640).
        input_w: Network input width (default 640).
        conf_threshold: Minimum confidence to keep a detection (default 0.5).
        nms_threshold: IoU threshold for NMS (default 0.4).
    """

    def __init__(
        self,
        model_path: str,
        input_h: int = 640,
        input_w: int = 640,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        self._input_h = input_h
        self._input_w = input_w
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._model_path = model_path

        # Configure ONNX Runtime session
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "GPU required: onnxruntime-gpu not available. "
                "Install onnxruntime-gpu to use SCRFD."
            )
        self._session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        # Cache input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Output layout varies by SCRFD export:
        #   6  outputs = score + bbox per stride (2 anchors/position)
        #   9  outputs = score + bbox + kps per stride (2 anchors/position)
        #   12 outputs = cls + obj + bbox + kps per stride (1 anchor/position)
        n_outputs = len(self._output_names)
        assert n_outputs in (6, 9, 12), (
            f"Unsupported SCRFD output count: {n_outputs} "
            f"(expected 6, 9, or 12). Outputs: {self._output_names}"
        )
        self._outputs_per_stride = n_outputs // len(_STRIDES)
        self._has_landmarks = self._outputs_per_stride >= 3
        self._has_obj = self._outputs_per_stride == 4
        self._num_anchors = 1 if self._outputs_per_stride == 4 else 2

        # Cache for anchor grids keyed by (stride, height, width)
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

        logger.info(
            "Loaded SCRFD ONNX model: %s (input=%dx%d, outputs=%d, landmarks=%s)",
            model_path,
            input_w,
            input_h,
            len(self._output_names),
            self._has_landmarks,
        )

    def _get_anchor_centers(
        self, stride: int, feat_h: int, feat_w: int
    ) -> np.ndarray:
        """Get or create cached anchor center grid for a stride level.

        Args:
            stride: Feature map stride (8, 16, or 32).
            feat_h: Feature map height.
            feat_w: Feature map width.

        Returns:
            ``(feat_h * feat_w, 2)`` float32 array of ``(x, y)`` anchor
            centers in input image coordinates.
        """
        key = (stride, feat_h, feat_w)
        if key in self._center_cache:
            return self._center_cache[key]

        # Grid of (x, y) anchor positions — SCRFD uses 2 anchors per position
        xs = np.arange(feat_w, dtype=np.float32) * stride
        ys = np.arange(feat_h, dtype=np.float32) * stride
        grid_x, grid_y = np.meshgrid(xs, ys)
        centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        if self._num_anchors > 1:
            centers = np.repeat(centers, self._num_anchors, axis=0)

        self._center_cache[key] = centers
        return centers

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Resize and normalize an image for SCRFD inference.

        Args:
            image: BGR image ``(H, W, 3)`` uint8.

        Returns:
            Tuple of:
                - ``input_tensor``: ``(1, 3, input_h, input_w)`` float32.
                - ``scale_h``: vertical scale factor (input_h / original_h).
                - ``scale_w``: horizontal scale factor (input_w / original_w).
        """
        orig_h, orig_w = image.shape[:2]
        scale_h = self._input_h / orig_h
        scale_w = self._input_w / orig_w

        # Resize to network input dimensions
        resized = cv2.resize(
            image, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )

        # Normalize: (pixel - 127.5) / 128.0, keep BGR channel order
        blob = (resized.astype(np.float32) - _SCRFD_MEAN) * _SCRFD_SCALE

        # HWC -> CHW, add batch dimension
        input_tensor = blob.transpose(2, 0, 1)[np.newaxis, ...]

        return input_tensor, scale_h, scale_w

    def _decode_outputs(
        self,
        outputs: list[np.ndarray],
        scale_h: float,
        scale_w: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw ONNX outputs into boxes, scores, and landmarks.

        Iterates over the 3 stride levels, generates anchor grids, and
        converts distance offsets into absolute coordinates. Applies
        confidence thresholding per level.

        Args:
            outputs: Raw ONNX output arrays (6 or 9 elements).
            scale_h: Vertical scale from preprocessing.
            scale_w: Horizontal scale from preprocessing.

        Returns:
            Tuple of:
                - ``boxes``: ``(K, 4)`` float32 ``[x1, y1, x2, y2]``.
                - ``scores``: ``(K,)`` float32 confidence scores.
                - ``landmarks``: ``(K, 5, 2)`` float32 landmark coordinates.
                  Zeros if the model does not output landmarks.
        """
        all_boxes = []
        all_scores = []
        all_landmarks = []

        # Outputs are grouped by type, not interleaved per stride:
        #   6 outputs: scores[0..2], bbox[3..5]
        #   9 outputs: scores[0..2], bbox[3..5], kps[6..8]
        #   12 outputs: cls[0..2], obj[3..5], bbox[6..8], kps[9..11]
        num_strides = len(_STRIDES)

        if self._has_obj:
            cls_off, obj_off, bbox_off, kps_off = 0, num_strides, 2 * num_strides, 3 * num_strides
        else:
            cls_off, obj_off, bbox_off, kps_off = 0, None, num_strides, 2 * num_strides

        for idx, stride in enumerate(_STRIDES):
            # ONNX outputs may be (N, C) or (1, N, C) depending on export.
            # Reshape everything to 2D so downstream indexing is uniform.
            cls_blob = outputs[cls_off + idx].reshape(-1, 1)         # (N, 1)
            bbox_blob = outputs[bbox_off + idx].reshape(-1, 4)       # (N, 4)

            if self._has_landmarks:
                kps_blob = outputs[kps_off + idx].reshape(-1, 10)    # (N, 10)

            cls_raw = cls_blob.reshape(-1)
            if self._has_obj:
                obj_raw = outputs[obj_off + idx].reshape(-1)
                scores_flat = (1.0 / (1.0 + np.exp(-cls_raw))) * (1.0 / (1.0 + np.exp(-obj_raw)))
            else:
                scores_flat = 1.0 / (1.0 + np.exp(-cls_raw))

            bbox_raw = bbox_blob

            # Filter by confidence threshold
            mask = scores_flat >= self._conf_threshold
            if not np.any(mask):
                continue

            scores_filtered = scores_flat[mask]
            bbox_filtered = bbox_raw[mask]

            # Compute feature map dimensions and get anchors
            feat_h = self._input_h // stride
            feat_w = self._input_w // stride
            anchors = self._get_anchor_centers(stride, feat_h, feat_w)
            anchors_filtered = anchors[mask]

            # Decode bounding boxes: distance offsets from anchor centers
            # bbox format: (left, top, right, bottom) distances
            x1 = (anchors_filtered[:, 0] - bbox_filtered[:, 0] * stride) / scale_w
            y1 = (anchors_filtered[:, 1] - bbox_filtered[:, 1] * stride) / scale_h
            x2 = (anchors_filtered[:, 0] + bbox_filtered[:, 2] * stride) / scale_w
            y2 = (anchors_filtered[:, 1] + bbox_filtered[:, 3] * stride) / scale_h

            boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            all_boxes.append(boxes)
            all_scores.append(scores_filtered)

            # Decode landmarks if available
            if self._has_landmarks:
                kps_filtered = kps_blob[mask]  # (M, 10)
                lms = np.zeros((len(kps_filtered), 5, 2), dtype=np.float32)
                for k in range(5):
                    lms[:, k, 0] = (
                        anchors_filtered[:, 0] + kps_filtered[:, 2 * k] * stride
                    ) / scale_w
                    lms[:, k, 1] = (
                        anchors_filtered[:, 1] + kps_filtered[:, 2 * k + 1] * stride
                    ) / scale_h
                all_landmarks.append(lms)

        if not all_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
            )

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        if self._has_landmarks and all_landmarks:
            landmarks = np.concatenate(all_landmarks, axis=0)
        else:
            landmarks = np.zeros((len(boxes), 5, 2), dtype=np.float32)

        return boxes, scores, landmarks

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Apply Non-Maximum Suppression.

        Args:
            boxes: ``(N, 4)`` float32 ``[x1, y1, x2, y2]``.
            scores: ``(N,)`` float32 confidence scores.
            threshold: IoU threshold for suppression.

        Returns:
            ``(K,)`` int array of indices to keep.
        """
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.intp)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= threshold)[0]
            order = order[remaining + 1]

        return np.array(keep, dtype=np.intp)

    def detect_faces(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Detect faces within a bounding box region.

        Crops the bbox region from the full image, runs SCRFD inference on
        the crop, applies NMS, then maps all coordinates back to full image
        space.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: Region bbox ``[x1, y1, x2, y2]`` to search within.

        Returns:
            Dict with:
                - ``face_boxes``: ``(M, 4)`` float32 ``[x1, y1, x2, y2]``
                  in full image coordinates.
                - ``face_scores``: ``(M,)`` float32 detection confidence.
                - ``landmarks``: ``(M, 5, 2)`` float32 five-point landmarks
                  (left_eye, right_eye, nose, left_mouth, right_mouth)
                  in full image coordinates. Zeros if model has no landmarks.
        """
        empty_result: dict[str, np.ndarray] = {
            "face_boxes": np.empty((0, 4), dtype=np.float32),
            "face_scores": np.empty((0,), dtype=np.float32),
            "landmarks": np.empty((0, 5, 2), dtype=np.float32),
        }

        # Crop the bbox region from the full image
        img_h, img_w = image.shape[:2]
        x1 = int(np.clip(bbox[0], 0, img_w))
        y1 = int(np.clip(bbox[1], 0, img_h))
        x2 = int(np.clip(bbox[2], 0, img_w))
        y2 = int(np.clip(bbox[3], 0, img_h))

        if x2 <= x1 or y2 <= y1:
            return empty_result

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return empty_result

        # Preprocess crop
        input_tensor, scale_h, scale_w = self._preprocess(crop)

        # Run ONNX inference
        outputs = self._session.run(
            self._output_names, {self._input_name: input_tensor}
        )

        # Decode outputs (coordinates are in crop space)
        boxes, scores, landmarks = self._decode_outputs(outputs, scale_h, scale_w)

        if len(boxes) == 0:
            return empty_result

        # Apply NMS
        keep = self._nms(boxes, scores, self._nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        # Map coordinates from crop space back to full image space
        boxes[:, 0] += x1
        boxes[:, 1] += y1
        boxes[:, 2] += x1
        boxes[:, 3] += y1

        landmarks[:, :, 0] += x1
        landmarks[:, :, 1] += y1

        return {
            "face_boxes": boxes,
            "face_scores": scores,
            "landmarks": landmarks,
        }

    @property
    def input_size(self) -> tuple[int, int]:
        """Expected network input size ``(H, W)``."""
        return (self._input_h, self._input_w)


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


@register_face_detector("scrfd")
def build_scrfd(config: dict) -> SCRFDModel:
    """Build an SCRFD face detector from config.

    Reads ``config["face_detector"]`` for:
        - ``arch``: variant name (``"scrfd-500m"`` or ``"scrfd-2.5g"``)
        - ``model_path``: path to ONNX file (optional, defaults per variant)
        - ``input_h``: network input height (default 640)
        - ``input_w``: network input width (default 640)
        - ``conf_threshold``: detection confidence threshold (default 0.5)
        - ``nms_threshold``: NMS IoU threshold (default 0.4)

    Args:
        config: Full config dict with a ``"face_detector"`` section.

    Returns:
        Configured :class:`SCRFDModel` instance.
    """
    det_cfg = config.get("face_detector", {})
    arch = det_cfg.get("arch", "scrfd-500m").lower()

    model_path = det_cfg.get("model_path", _DEFAULT_ONNX_PATHS.get(arch))
    if model_path is None:
        model_path = _DEFAULT_ONNX_PATHS["scrfd-500m"]

    input_h = det_cfg.get("input_h", 640)
    input_w = det_cfg.get("input_w", 640)
    conf_threshold = det_cfg.get("conf_threshold", 0.5)
    nms_threshold = det_cfg.get("nms_threshold", 0.4)

    return SCRFDModel(
        model_path=model_path,
        input_h=input_h,
        input_w=input_w,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )


# Register variant aliases so both "scrfd-500m" and "scrfd-2.5g" resolve
# to the canonical "scrfd" builder.
_FACE_DETECTOR_VARIANT_MAP["scrfd-500m"] = "scrfd"
_FACE_DETECTOR_VARIANT_MAP["scrfd-2.5g"] = "scrfd"
