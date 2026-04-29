"""MobileFaceNet (ArcFace) face embedding using ONNX Runtime inference.

Extracts 512-dimensional L2-normalized face embeddings for identity matching.
Supports affine-aligned face crops (when 5-point landmarks are available from
SCRFD) or simple crop+resize fallback.

Typical usage::

    from core.p06_models.face_registry import build_face_embedder

    config = {
        "face_embedder": {
            "arch": "mobilefacenet",
            "model_path": "pretrained/mobilefacenet_arcface.onnx",
        }
    }
    model = build_face_embedder(config)
    embedding = model.extract_embedding(image, face_box, landmarks)
"""

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from core.p06_models.face_base import ARCFACE_REF_LANDMARKS, FaceEmbedder
from core.p06_models.face_registry import _FACE_EMBEDDER_VARIANT_MAP, register_face_embedder

# Default ONNX path
_DEFAULT_ONNX_PATH = "pretrained/mobilefacenet_arcface.onnx"


class MobileFaceNetModel(FaceEmbedder):
    """MobileFaceNet face embedder using ONNX Runtime.

    Loads a MobileFaceNet model trained with ArcFace loss and runs inference
    via ONNX Runtime. Given a full image and a face bounding box (plus
    optional 5-point landmarks), produces a 512-d L2-normalized embedding.

    When landmarks are provided, the face is aligned to the standard ArcFace
    112x112 template using a similarity transform, which significantly
    improves embedding quality. Without landmarks, a simple crop+resize
    fallback is used.

    Args:
        model_path: Path to the ``.onnx`` model file.
        input_h: Input face crop height (default 112).
        input_w: Input face crop width (default 112).
    """

    def __init__(
        self, model_path: str, input_h: int = 112, input_w: int = 112
    ) -> None:
        self._input_h = input_h
        self._input_w = input_w
        self._model_path = model_path

        # Configure ONNX Runtime session
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "GPU required: onnxruntime-gpu not available. "
                "Install onnxruntime-gpu to use MobileFaceNet."
            )
        self._session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

        # Cache input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        logger.info(
            "Loaded MobileFaceNet ONNX model: %s (input=%dx%d, outputs=%s)",
            model_path,
            input_w,
            input_h,
            self._output_names,
        )

    def _align_face(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """Align a face using similarity transform from detected landmarks.

        Computes a similarity transform (rotation, uniform scale, translation)
        mapping the detected 5-point landmarks to the ArcFace reference
        template, then warps the image to produce a 112x112 aligned crop.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            landmarks: ``(5, 2)`` float32 five-point landmarks
                (left_eye, right_eye, nose, left_mouth, right_mouth).

        Returns:
            ``(112, 112, 3)`` BGR uint8 aligned face crop.
        """
        # estimateAffinePartial2D computes a 4-DOF similarity transform
        # (rotation, uniform scale, translation)
        warp_mat, _ = cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            ARCFACE_REF_LANDMARKS,
            method=cv2.LMEDS,
        )

        aligned = cv2.warpAffine(
            image,
            warp_mat,
            (self._input_w, self._input_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned

    def _crop_face(
        self, image: np.ndarray, face_box: np.ndarray
    ) -> np.ndarray:
        """Crop and resize a face region (fallback when landmarks unavailable).

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            face_box: ``[x1, y1, x2, y2]`` face bounding box.

        Returns:
            ``(112, 112, 3)`` BGR uint8 face crop.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = face_box[:4]

        # Clip to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        crop = image[y1:y2, x1:x2]

        # Guard against degenerate boxes
        if crop.size == 0:
            return np.zeros(
                (self._input_h, self._input_w, 3), dtype=np.uint8
            )

        resized = cv2.resize(
            crop,
            (self._input_w, self._input_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Normalize a face crop for ONNX inference.

        Converts BGR to RGB, normalizes pixel values to [-1, 1] range
        using ``(pixel / 255 - 0.5) / 0.5``, and rearranges to NCHW.

        Args:
            face_crop: ``(112, 112, 3)`` BGR uint8 face crop.

        Returns:
            ``(1, 3, 112, 112)`` float32 input tensor.
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Normalize to [-1, 1]: (pixel / 255 - 0.5) / 0.5
        rgb = (rgb / 255.0 - 0.5) / 0.5

        # HWC -> CHW, add batch dimension
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]
        return tensor

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embedding vectors along the last axis.

        Args:
            embeddings: ``(...)`` float32 array with embedding vectors
                along the last dimension.

        Returns:
            L2-normalized array of the same shape.
        """
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        return embeddings / norms

    def extract_embedding(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Extract a face embedding from a detected face region.

        Uses affine alignment when 5-point landmarks are available (from
        SCRFD), otherwise falls back to simple crop+resize.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            face_box: ``[x1, y1, x2, y2]`` of the detected face.
            landmarks: Optional ``(5, 2)`` five-point landmarks for alignment.

        Returns:
            ``(512,)`` float32 L2-normalized embedding vector.
        """
        # 1. Align (if landmarks) or crop (fallback)
        if landmarks is not None and landmarks.shape == (5, 2):
            face_crop = self._align_face(image, landmarks)
        else:
            face_crop = self._crop_face(image, face_box)

        # 2. Preprocess
        input_tensor = self._preprocess(face_crop)

        # 3. ONNX inference
        outputs = self._session.run(
            self._output_names, {self._input_name: input_tensor}
        )
        raw_embedding = outputs[0]  # (1, 512)

        # 4. L2 normalize and flatten
        embedding = self._l2_normalize(raw_embedding)[0]  # (512,)

        return embedding.astype(np.float32)

    def extract_embedding_batch(
        self,
        image: np.ndarray,
        face_boxes: np.ndarray,
        landmarks_list: list[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        """Extract embeddings for multiple faces in a single ONNX call.

        Stacks all face crops into one batch tensor for efficient GPU/CPU
        inference, then L2-normalizes all embeddings.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            face_boxes: ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.
            landmarks_list: Optional list of *N* landmark arrays ``(5, 2)``
                (or ``None`` for faces without landmarks).

        Returns:
            ``(N, 512)`` float32 L2-normalized embedding matrix.
        """
        n = len(face_boxes)
        if n == 0:
            return np.empty((0, 512), dtype=np.float32)

        # Preprocess all face crops
        crop_tensors = []
        for i in range(n):
            lm = None if landmarks_list is None else landmarks_list[i]

            if lm is not None and lm.shape == (5, 2):
                face_crop = self._align_face(image, lm)
            else:
                face_crop = self._crop_face(image, face_boxes[i])

            tensor = self._preprocess(face_crop)
            crop_tensors.append(tensor[0])  # remove batch dim, will re-stack

        # Stack into single batch: (N, 3, 112, 112)
        batch_input = np.stack(crop_tensors, axis=0).astype(np.float32)

        # Run ONNX inference on full batch
        outputs = self._session.run(
            self._output_names, {self._input_name: batch_input}
        )
        raw_embeddings = outputs[0]  # (N, 512)

        # L2 normalize
        embeddings = self._l2_normalize(raw_embeddings)

        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimension (512 for MobileFaceNet)."""
        return 512

    @property
    def input_size(self) -> tuple[int, int]:
        """Expected aligned face input size ``(H, W)``."""
        return (self._input_h, self._input_w)


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


@register_face_embedder("mobilefacenet")
def build_mobilefacenet(config: dict) -> MobileFaceNetModel:
    """Build a MobileFaceNet embedder from config.

    Reads ``config["face_embedder"]`` for:
        - ``model_path``: path to ONNX file (default
          ``pretrained/mobilefacenet_arcface.onnx``)
        - ``input_h``: crop height (default 112)
        - ``input_w``: crop width (default 112)

    Args:
        config: Full config dict with a ``"face_embedder"`` section.

    Returns:
        Configured :class:`MobileFaceNetModel` instance.
    """
    emb_cfg = config.get("face_embedder", {})
    model_path = emb_cfg.get("model_path", _DEFAULT_ONNX_PATH)
    input_h = emb_cfg.get("input_h", 112)
    input_w = emb_cfg.get("input_w", 112)
    return MobileFaceNetModel(model_path=model_path, input_h=input_h, input_w=input_w)


# Register variant alias so "mobilefacenet-arcface" resolves to "mobilefacenet"
_FACE_EMBEDDER_VARIANT_MAP["mobilefacenet-arcface"] = "mobilefacenet"
