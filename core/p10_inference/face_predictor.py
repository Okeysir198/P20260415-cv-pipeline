"""Face recognition predictor — identify violators by face.

Given detection results from any upstream detector, crops the relevant
regions, detects faces, extracts embeddings, and matches against an
enrolled gallery to identify the person.

Batching strategy
-----------------
SCRFD's current Python wrapper (``core.p06_models.scrfd``) accepts a
single bbox per call and crops internally — so face *detection* stays in
a per-detection loop. Embedding extraction and gallery matching, however,
are batched: after looping over detections we stack the best face box and
landmarks for each violator, call
``FaceEmbedder.extract_embedding_batch(...)`` once, then
``FaceGallery.match_batch(...)`` once. Large reduction in ONNX session
overhead and CUDA round-trips when there are several violators per frame.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p06_models.face_base import FaceDetector, FaceEmbedder
from core.p10_inference.face_gallery import FaceGallery


class FacePredictor:
    """Identify violators by face recognition on violation detections.

    Args:
        face_detector: FaceDetector for finding faces in head crops.
        face_embedder: FaceEmbedder for extracting face embeddings.
        gallery: FaceGallery with enrolled identities.
        violation_class_ids: Class IDs that trigger face recognition.
        expand_ratio: Expand violation bbox by this ratio to capture
            the full head region for face detection.
        face_conf_threshold: Minimum face detection confidence.
    """

    def __init__(
        self,
        face_detector: FaceDetector,
        face_embedder: FaceEmbedder,
        gallery: FaceGallery,
        violation_class_ids: list[int],
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
        self, image: np.ndarray, det_results: dict[str, Any]
    ) -> dict[str, Any]:
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

        identities: list[str | None] = [None] * n_dets
        identity_scores: list[float] = [0.0] * n_dets
        face_boxes: list[np.ndarray | None] = [None] * n_dets

        if n_dets == 0:
            return {
                "identities": identities,
                "identity_scores": identity_scores,
                "face_boxes": face_boxes,
            }

        h, w = image.shape[:2]

        # Pass 1 — per-detection face detection. Collect inputs for the
        # batched embedding step; also record which detection each entry
        # maps back to so we can scatter results in pass 3.
        embed_indices: list[int] = []  # det index per face
        embed_boxes: list[np.ndarray] = []
        embed_landmarks: list[np.ndarray | None] = []

        for i in range(n_dets):
            if int(labels[i]) not in self.violation_class_ids:
                continue

            # Expand bbox to capture full head region
            expanded = self._expand_bbox(boxes[i], self.expand_ratio, w, h)

            # SCRFD takes one bbox per call (does its own crop internally),
            # so detection itself is not batched here.
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

            embed_indices.append(i)
            embed_boxes.append(best_face_box)
            embed_landmarks.append(
                best_landmarks if np.any(best_landmarks) else None
            )

        # Pass 2 — batched embedding + gallery match.
        if embed_indices:
            face_boxes_arr = np.stack(embed_boxes, axis=0).astype(np.float32)
            embeddings = self._extract_embeddings_batched(
                image, face_boxes_arr, embed_landmarks
            )
            matches = self.gallery.match_batch(embeddings)
            for det_idx, (identity, score) in zip(embed_indices, matches, strict=True):
                identities[det_idx] = identity
                identity_scores[det_idx] = score

        return {
            "identities": identities,
            "identity_scores": identity_scores,
            "face_boxes": face_boxes,
        }

    def _extract_embeddings_batched(
        self,
        image: np.ndarray,
        face_boxes: np.ndarray,
        landmarks_list: list[np.ndarray | None],
    ) -> np.ndarray:
        """Extract embeddings for *N* faces, preferring a batched API.

        Falls back to a per-face loop if the embedder does not expose
        ``extract_embedding_batch``.

        Returns:
            ``(N, D)`` float32 L2-normalized embedding matrix.
        """
        batch_fn = getattr(self.face_embedder, "extract_embedding_batch", None)
        if callable(batch_fn):
            return batch_fn(image, face_boxes, landmarks_list)

        # Fallback: one call per face
        embeddings = [
            self.face_embedder.extract_embedding(image, face_boxes[j], lm)
            for j, lm in enumerate(landmarks_list)
        ]
        return np.stack(embeddings, axis=0).astype(np.float32)

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
