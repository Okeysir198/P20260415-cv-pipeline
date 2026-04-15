"""Abstract base classes for face recognition models (inference-only).

All face model implementations must inherit from :class:`FaceDetector` or
:class:`FaceEmbedder` and implement the required abstract methods.

Face detection finds faces within a cropped region (typically a head/person
bounding box from the violation detector). Face embedding extracts a compact
vector representation for identity matching against an enrolled gallery.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ArcFace alignment template — standard 5-point landmark reference positions
# for 112x112 aligned face crops.  Used by FaceEmbedder implementations.
# ---------------------------------------------------------------------------

ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


class FaceDetector(ABC):
    """Abstract base class for face detection models.

    Subclasses must implement :meth:`detect_faces` and :attr:`input_size`.

    This is inference-only — no ``nn.Module`` inheritance, no ``forward()``
    method, no training support. Models load ONNX weights directly.
    """

    @abstractmethod
    def detect_faces(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Detect faces within a bounding box region.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bbox: Region bbox ``[x1, y1, x2, y2]`` to search within.

        Returns:
            Dict with:
                - ``face_boxes``: ``(M, 4)`` float32 ``[x1, y1, x2, y2]``
                  in image coordinates.
                - ``face_scores``: ``(M,)`` float32 detection confidence.
                - ``landmarks``: ``(M, 5, 2)`` float32 five-point landmarks
                  (left_eye, right_eye, nose, left_mouth, right_mouth).
                  Zeros if the detector does not output landmarks.
        """

    def detect_faces_batch(
        self, image: np.ndarray, bboxes: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """Detect faces in multiple bounding box regions.

        Default implementation loops over bboxes. Subclasses may override
        for batched inference.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            bboxes: ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.

        Returns:
            List of *N* detection dicts (same format as
            :meth:`detect_faces`).
        """
        results = []
        for i in range(len(bboxes)):
            results.append(self.detect_faces(image, bboxes[i]))
        return results

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Expected input size ``(H, W)``."""


class FaceEmbedder(ABC):
    """Abstract base class for face embedding models.

    Subclasses must implement :meth:`extract_embedding`, :attr:`embedding_dim`,
    and :attr:`input_size`.

    This is inference-only — no ``nn.Module`` inheritance, no ``forward()``
    method, no training support. Models load ONNX weights directly.
    """

    @abstractmethod
    def extract_embedding(
        self,
        image: np.ndarray,
        face_box: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract a face embedding from a detected face region.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            face_box: ``[x1, y1, x2, y2]`` of the detected face.
            landmarks: Optional ``(5, 2)`` five-point landmarks for alignment.
                When provided, the implementation should use affine alignment
                to the ArcFace reference template for better embedding quality.

        Returns:
            ``(D,)`` float32 L2-normalized embedding vector where *D* is
            :attr:`embedding_dim`.
        """

    def extract_embedding_batch(
        self,
        image: np.ndarray,
        face_boxes: np.ndarray,
        landmarks_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Extract embeddings for multiple detected faces.

        Default implementation loops over faces. Subclasses may override
        for batched ONNX inference.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            face_boxes: ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]``.
            landmarks_list: Optional list of *N* landmark arrays (or None).

        Returns:
            ``(N, D)`` float32 L2-normalized embedding matrix.
        """
        embeddings = []
        for i in range(len(face_boxes)):
            lm = None if landmarks_list is None else landmarks_list[i]
            embeddings.append(self.extract_embedding(image, face_boxes[i], lm))
        if not embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.stack(embeddings, axis=0)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding vector dimension (typically 512)."""

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Expected aligned face input size ``(H, W)``."""
