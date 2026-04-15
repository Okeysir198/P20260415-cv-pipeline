"""Face recognition predictor — identify violators by face.

Given detection results from a violation detector (e.g. helmet model),
crops head regions, detects faces, extracts embeddings, and matches
against an enrolled gallery to identify the person.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p10_inference.face_gallery import FaceGallery
from core.p06_models.face_base import FaceDetector, FaceEmbedder

logger = logging.getLogger(__name__)


class FacePredictor:
    """Identify violators by face recognition on violation detections.

    Args:
        face_detector: FaceDetector for finding faces in head crops.
        face_embedder: FaceEmbedder for extracting face embeddings.
        gallery: FaceGallery with enrolled identities.
        violation_class_ids: Class IDs that trigger face recognition
            (e.g. ``[2]`` for ``head_without_helmet``).
        expand_ratio: Expand violation bbox by this ratio to capture
            the full head region for face detection.
        face_conf_threshold: Minimum face detection confidence.
    """

    def __init__(
        self,
        face_detector: FaceDetector,
        face_embedder: FaceEmbedder,
        gallery: FaceGallery,
        violation_class_ids: List[int],
        expand_ratio: float = 1.5,
        face_conf_threshold: float = 0.5,
    ) -> None:
        self.face_detector = face_detector
        self.face_embedder = face_embedder
        self.gallery = gallery
        self.violation_class_ids = set(violation_class_ids)
        self.expand_ratio = expand_ratio
        self.face_conf_threshold = face_conf_threshold

    def identify(
        self, image: np.ndarray, det_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run face recognition on violation detections.

        Args:
            image: Full BGR image ``(H, W, 3)`` uint8.
            det_results: Detection results dict with keys ``boxes``,
                ``scores``, ``labels`` from the violation detector.

        Returns:
            Dict with:
                - ``identities``: List of str — identity per detection
                  (``None`` if not a violation or no face found).
                - ``identity_scores``: List of float — similarity scores.
                - ``face_boxes``: List of Optional[ndarray] — face bbox
                  per detection (``None`` if not applicable).
        """
        boxes = det_results.get("boxes", np.empty((0, 4)))
        labels = det_results.get("labels", np.empty((0,), dtype=int))
        n_dets = len(boxes)

        identities: List[Optional[str]] = [None] * n_dets
        identity_scores: List[float] = [0.0] * n_dets
        face_boxes: List[Optional[np.ndarray]] = [None] * n_dets

        if n_dets == 0:
            return {
                "identities": identities,
                "identity_scores": identity_scores,
                "face_boxes": face_boxes,
            }

        h, w = image.shape[:2]

        for i in range(n_dets):
            if int(labels[i]) not in self.violation_class_ids:
                continue

            # Expand bbox to capture full head region
            expanded = self._expand_bbox(boxes[i], self.expand_ratio, w, h)

            # Detect faces within the expanded region
            face_result = self.face_detector.detect_faces(image, expanded)
            if len(face_result["face_boxes"]) == 0:
                identities[i] = "unknown"
                continue

            # Take the highest-confidence face
            best_face_idx = int(np.argmax(face_result["face_scores"]))
            if face_result["face_scores"][best_face_idx] < self.face_conf_threshold:
                identities[i] = "unknown"
                continue

            best_face_box = face_result["face_boxes"][best_face_idx]
            best_landmarks = face_result["landmarks"][best_face_idx]
            face_boxes[i] = best_face_box

            # Check if landmarks are valid (non-zero)
            lm = best_landmarks if np.any(best_landmarks) else None

            # Extract embedding and match
            embedding = self.face_embedder.extract_embedding(
                image, best_face_box, lm
            )
            identity, score = self.gallery.match(embedding)
            identities[i] = identity
            identity_scores[i] = score

        return {
            "identities": identities,
            "identity_scores": identity_scores,
            "face_boxes": face_boxes,
        }

    @staticmethod
    def _expand_bbox(
        bbox: np.ndarray, ratio: float, img_w: int, img_h: int
    ) -> np.ndarray:
        """Expand a bounding box by a ratio, clipping to image bounds.

        Args:
            bbox: ``[x1, y1, x2, y2]`` bounding box.
            ratio: Expansion ratio (1.5 = 50% larger).
            img_w: Image width for clipping.
            img_h: Image height for clipping.

        Returns:
            Expanded ``[x1, y1, x2, y2]`` bbox clipped to image.
        """
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * ratio, (y2 - y1) * ratio
        return np.array(
            [
                max(0, cx - bw / 2),
                max(0, cy - bh / 2),
                min(img_w, cx + bw / 2),
                min(img_h, cy + bh / 2),
            ],
            dtype=np.float32,
        )
