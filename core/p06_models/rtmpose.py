"""RTMPose-S/T pose estimation using ONNX Runtime inference.

Top-down single/multi-person pose estimation with SimCC coordinate
decoding. Supports RTMPose-S (5.47M params) and RTMPose-T (3.34M params)
variants via the pose model registry.

Typical usage::

    from core.p06_models.pose_registry import build_pose_model

    config = {
        "pose_model": {
            "arch": "rtmpose-s",
            "model_path": "pretrained/rtmpose_s_256x192.onnx",
        }
    }
    model = build_pose_model(config)
    result = model.predict_keypoints(image, bbox)
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch

from core.p06_models.pose_base import COCO_KEYPOINT_NAMES, COCO_SKELETON, PoseModel
from loguru import logger
from core.p06_models.pose_registry import _POSE_VARIANT_MAP, register_pose_model

# Default ONNX paths per variant
_DEFAULT_ONNX_PATHS: dict[str, str] = {
    "rtmpose-s": "pretrained/rtmpose_s_256x192.onnx",
    "rtmpose-t": "pretrained/rtmpose_t_256x192.onnx",
}

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Bbox padding factor (standard for top-down pose estimation)
_BBOX_PADDING = 1.25


def _get_warp_matrix(
    center: np.ndarray, scale: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    """Get affine transform matrix for top-down pose estimation.

    Args:
        center: ``(2,)`` bbox center in image coordinates.
        scale: ``(2,)`` bbox size (width, height) with padding applied.
        output_size: ``(w, h)`` target crop size.

    Returns:
        ``(2, 3)`` affine transform matrix.
    """
    src = np.array(
        [
            center,
            center + [0, -0.5 * scale[1]],
            center + [0.5 * scale[0], 0],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [output_size[0] * 0.5, output_size[1] * 0.5],
            [output_size[0] * 0.5, 0],
            [output_size[0], output_size[1] * 0.5],
        ],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)


def _decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decode SimCC coordinate representations.

    The model outputs logits over discretized x and y bins at 2x the input
    resolution. We take the argmax for coordinates and use the sigmoid of
    the max logit as a confidence proxy (avoids full softmax).

    Args:
        simcc_x: ``(N, K, Wx)`` logits for x-coordinates.
        simcc_y: ``(N, K, Wy)`` logits for y-coordinates.

    Returns:
        Tuple of:
            - ``coords``: ``(N, K, 2)`` float32 keypoint coordinates in crop
              space (at original input resolution, i.e. divided by 2).
            - ``scores``: ``(N, K)`` float32 per-keypoint confidence scores.
    """
    # Move to GPU once, do argmax/max/sigmoid in fused kernels, then bring back.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tx = torch.from_numpy(simcc_x).to(device)
    ty = torch.from_numpy(simcc_y).to(device)

    max_x_out = torch.max(tx, dim=-1)
    max_y_out = torch.max(ty, dim=-1)
    x_locs_t = max_x_out.indices
    y_locs_t = max_y_out.indices
    max_x = max_x_out.values
    max_y = max_y_out.values

    scores_t = torch.minimum(torch.sigmoid(max_x), torch.sigmoid(max_y))

    # SimCC coords are at 2x resolution
    coords_t = torch.stack(
        [x_locs_t.to(torch.float32) / 2.0, y_locs_t.to(torch.float32) / 2.0], dim=-1
    )  # (N, K, 2)

    coords = coords_t.cpu().numpy().astype(np.float32)
    scores = scores_t.cpu().numpy().astype(np.float32)
    return coords, scores


class RTMPoseModel(PoseModel):
    """RTMPose pose estimator using ONNX Runtime.

    Performs top-down pose estimation: given a full image and a person
    bounding box, crops and resizes the region, runs ONNX inference, and
    decodes SimCC outputs back to image coordinates.

    Args:
        model_path: Path to the ``.onnx`` model file.
        input_h: Input crop height (default 256).
        input_w: Input crop width (default 192).
    """

    def __init__(
        self, model_path: str, input_h: int = 256, input_w: int = 192
    ) -> None:
        self._input_h = input_h
        self._input_w = input_w
        self._model_path = model_path

        # Configure ONNX Runtime session
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "GPU required: onnxruntime-gpu not available. "
                "Install onnxruntime-gpu to use RTMPose."
            )
        self._session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        # Cache input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info(
            "Loaded RTMPose ONNX model: %s (input=%dx%d, outputs=%s)",
            model_path,
            input_w,
            input_h,
            self._output_names,
        )

    def _preprocess(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop, resize, and normalize a single person region.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: ``[x1, y1, x2, y2]`` person bounding box.

        Returns:
            Tuple of:
                - ``input_tensor``: ``(1, 3, input_h, input_w)`` float32.
                - ``inv_warp_mat``: ``(2, 3)`` inverse affine transform matrix
                  for mapping crop coords back to image coords.
        """
        x1, y1, x2, y2 = bbox[:4]
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        scale = np.array(
            [(x2 - x1) * _BBOX_PADDING, (y2 - y1) * _BBOX_PADDING], dtype=np.float32
        )

        output_size = (self._input_w, self._input_h)
        warp_mat = _get_warp_matrix(center, scale, output_size)

        # Warp the image region
        crop = cv2.warpAffine(
            image,
            warp_mat,
            output_size,
            flags=cv2.INTER_LINEAR,
        )

        # BGR -> RGB, normalize with ImageNet stats
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_rgb = (crop_rgb - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC -> CHW, add batch dimension
        input_tensor = crop_rgb.transpose(2, 0, 1)[np.newaxis, ...]

        # Compute inverse affine for mapping back to image coords
        inv_warp_mat = cv2.invertAffineTransform(warp_mat)

        return input_tensor, inv_warp_mat

    def predict_keypoints(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Estimate keypoints for a single person.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: Person bounding box ``[x1, y1, x2, y2]`` in image coords.

        Returns:
            Dict with:
                - ``keypoints``: ``(17, 3)`` float32 array of (x, y, score)
                  in image coordinates.
                - ``score``: float — mean confidence of all keypoints.
        """
        input_tensor, inv_warp_mat = self._preprocess(image, bbox)

        # Run ONNX inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        simcc_x, simcc_y = outputs[0], outputs[1]  # (1, K, Wx), (1, K, Wy)

        # Decode SimCC to crop-space coordinates
        coords, scores = _decode_simcc(simcc_x, simcc_y)  # (1, K, 2), (1, K)
        coords = coords[0]  # (K, 2)
        scores = scores[0]  # (K,)

        # Map crop coords back to image coords via inverse affine
        ones = np.ones((coords.shape[0], 1), dtype=np.float32)
        coords_hom = np.concatenate([coords, ones], axis=1)  # (K, 3)
        coords_img = (inv_warp_mat @ coords_hom.T).T  # (K, 2)

        # Assemble (K, 3) output: x, y, score
        keypoints = np.concatenate(
            [coords_img, scores[:, np.newaxis]], axis=1
        ).astype(np.float32)

        return {
            "keypoints": keypoints,
            "score": float(np.mean(scores)),
        }

    def predict_keypoints_batch(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """Estimate keypoints for multiple persons in a single ONNX call.

        Stacks all person crops into one batch tensor for efficient GPU/CPU
        inference, then decodes and maps results back individually.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bboxes: ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.

        Returns:
            List of *N* prediction dicts (same format as
            :meth:`predict_keypoints`).
        """
        n = len(bboxes)
        if n == 0:
            return []

        # Preprocess all crops
        batch_tensors = []
        inv_warp_mats = []
        for i in range(n):
            tensor, inv_mat = self._preprocess(image, bboxes[i])
            batch_tensors.append(tensor[0])  # remove batch dim, will re-stack
            inv_warp_mats.append(inv_mat)

        # Stack into single batch: (N, 3, H, W)
        batch_input = np.stack(batch_tensors, axis=0).astype(np.float32)

        # Run ONNX inference on full batch
        outputs = self._session.run(
            self._output_names, {self._input_name: batch_input}
        )
        simcc_x, simcc_y = outputs[0], outputs[1]  # (N, K, Wx), (N, K, Wy)

        # Decode all at once
        all_coords, all_scores = _decode_simcc(simcc_x, simcc_y)  # (N, K, 2), (N, K)

        # Map each person's coords back to image space
        results = []
        for i in range(n):
            coords = all_coords[i]  # (K, 2)
            scores = all_scores[i]  # (K,)

            ones = np.ones((coords.shape[0], 1), dtype=np.float32)
            coords_hom = np.concatenate([coords, ones], axis=1)  # (K, 3)
            coords_img = (inv_warp_mats[i] @ coords_hom.T).T  # (K, 2)

            keypoints = np.concatenate(
                [coords_img, scores[:, np.newaxis]], axis=1
            ).astype(np.float32)

            results.append(
                {
                    "keypoints": keypoints,
                    "score": float(np.mean(scores)),
                }
            )

        return results

    @property
    def keypoint_names(self) -> list[str]:
        """Ordered list of COCO 17 keypoint names."""
        return COCO_KEYPOINT_NAMES

    @property
    def num_keypoints(self) -> int:
        """Number of keypoints (17 for COCO schema)."""
        return 17

    @property
    def skeleton(self) -> list[tuple[int, int]]:
        """COCO skeleton bone connectivity."""
        return COCO_SKELETON

    @property
    def input_size(self) -> tuple[int, int]:
        """Expected crop input size ``(H, W)``."""
        return (self._input_h, self._input_w)


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


@register_pose_model("rtmpose")
def build_rtmpose(config: dict) -> RTMPoseModel:
    """Build an RTMPose model from config.

    Reads ``config["pose_model"]`` for:
        - ``arch``: variant name (``"rtmpose-s"`` or ``"rtmpose-t"``)
        - ``model_path``: path to ONNX file (optional, defaults per variant)
        - ``input_h``: crop height (default 256)
        - ``input_w``: crop width (default 192)

    Args:
        config: Full config dict with a ``"pose_model"`` section.

    Returns:
        Configured :class:`RTMPoseModel` instance.
    """
    pose_cfg = config.get("pose_model", {})
    arch = pose_cfg.get("arch", "rtmpose-s").lower()

    model_path = pose_cfg.get("model_path", _DEFAULT_ONNX_PATHS.get(arch))
    if model_path is None:
        model_path = _DEFAULT_ONNX_PATHS["rtmpose-s"]

    input_h = pose_cfg.get("input_h", 256)
    input_w = pose_cfg.get("input_w", 192)

    return RTMPoseModel(model_path=model_path, input_h=input_h, input_w=input_w)


# Register variant aliases so both "rtmpose-s" and "rtmpose-t" resolve
# to the canonical "rtmpose" builder.
_POSE_VARIANT_MAP["rtmpose-s"] = "rtmpose"
_POSE_VARIANT_MAP["rtmpose-t"] = "rtmpose"
