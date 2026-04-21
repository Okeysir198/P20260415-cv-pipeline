"""Pose estimation predictor — top-down (detector + pose model) pipeline.

Detects persons with any DetectionPredictor, then estimates keypoints on
each person crop using any PoseModel from the registry.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p06_models.pose_base import PoseModel
from core.p10_inference.predictor import DetectionPredictor

logger = logging.getLogger(__name__)


class PosePredictor:
    """Top-down pose estimation: detect persons, then estimate keypoints.

    Args:
        detector: DetectionPredictor for person detection.
        pose_model: PoseModel for keypoint estimation.
        person_conf_threshold: Min confidence to consider a person detection.
        person_class_ids: Class IDs that represent "person" in the detector.
    """

    def __init__(
        self,
        detector: DetectionPredictor,
        pose_model: PoseModel,
        person_conf_threshold: float = 0.5,
        person_class_ids: list[int] | None = None,
    ) -> None:
        self.detector = detector
        self.pose_model = pose_model
        self.person_conf_threshold = person_conf_threshold
        self.person_class_ids = person_class_ids if person_class_ids is not None else [0]

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run detection + pose estimation on a single BGR image.

        Returns:
            Dict with:
                - ``boxes``: (N, 4) float32 [x1,y1,x2,y2]
                - ``scores``: (N,) float32 person detection confidence
                - ``keypoints``: (N, K, 3) float32 [x, y, kpt_score]
                - ``keypoint_scores``: (N,) float32 per-person pose score
                - ``keypoint_names``: List[str] of K names
                - ``skeleton``: List[Tuple[int,int]] bone connections
        """
        det_results = self.detector.predict(image)

        # Filter to person-class detections above threshold
        if det_results["boxes"].shape[0] == 0:
            return self._empty_results()

        mask = np.zeros(len(det_results["labels"]), dtype=bool)
        for cls_id in self.person_class_ids:
            mask |= det_results["labels"] == cls_id
        mask &= det_results["scores"] >= self.person_conf_threshold

        if not mask.any():
            return self._empty_results()

        person_boxes = det_results["boxes"][mask]
        person_scores = det_results["scores"][mask]

        # Run pose on each person
        pose_results = self.pose_model.predict_keypoints_batch(image, person_boxes)

        if not pose_results:
            return self._empty_results()

        keypoints = np.stack([r["keypoints"] for r in pose_results])
        kpt_scores = np.array([r["score"] for r in pose_results], dtype=np.float32)

        return {
            "boxes": person_boxes,
            "scores": person_scores,
            "keypoints": keypoints,
            "keypoint_scores": kpt_scores,
            "keypoint_names": self.pose_model.keypoint_names,
            "skeleton": self.pose_model.skeleton,
        }

    def predict_coco(self, image: np.ndarray) -> dict[str, Any]:
        """Same as predict() but keypoints are always COCO 17-format.

        Calls ``pose_model.to_coco()`` internally for non-COCO models.
        """
        results = self.predict(image)
        if results["keypoints"].shape[0] == 0:
            return results

        if results["keypoints"].shape[1] != 17:
            coco_kpts = np.stack([
                self.pose_model.to_coco(kpts) for kpts in results["keypoints"]
            ])
            results["keypoints"] = coco_kpts
            from core.p06_models.pose_base import COCO_KEYPOINT_NAMES, COCO_SKELETON
            results["keypoint_names"] = COCO_KEYPOINT_NAMES
            results["skeleton"] = COCO_SKELETON

        return results

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        """Load an image from disk and run pose estimation."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")
        return self.predict(image)

    def visualize(
        self,
        image: np.ndarray,
        predictions: dict[str, Any],
        save_path: str | Path | None = None,
        keypoint_threshold: float = 0.3,
    ) -> np.ndarray:
        """Draw skeletons, keypoints, and person boxes on image.

        Args:
            image: Original BGR image.
            predictions: Output from predict() or predict_coco().
            save_path: If provided, save annotated image.
            keypoint_threshold: Min keypoint score to draw.

        Returns:
            Annotated BGR image.
        """
        vis = image.copy()
        boxes = predictions["boxes"]
        keypoints = predictions["keypoints"]
        skeleton = predictions["skeleton"]

        # Color palette for skeleton bones
        bone_color = (0, 255, 0)
        kpt_color = (0, 0, 255)
        box_color = (255, 0, 0)

        for i in range(len(boxes)):
            # Draw person box
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            kpts = keypoints[i]  # (K, 3)

            # Draw skeleton bones
            for idx_a, idx_b in skeleton:
                if idx_a >= len(kpts) or idx_b >= len(kpts):
                    continue
                if kpts[idx_a, 2] < keypoint_threshold or kpts[idx_b, 2] < keypoint_threshold:
                    continue
                pt_a = (int(kpts[idx_a, 0]), int(kpts[idx_a, 1]))
                pt_b = (int(kpts[idx_b, 0]), int(kpts[idx_b, 1]))
                cv2.line(vis, pt_a, pt_b, bone_color, 2)

            # Draw keypoints
            for k in range(len(kpts)):
                if kpts[k, 2] < keypoint_threshold:
                    continue
                pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                cv2.circle(vis, pt, 4, kpt_color, -1)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            logger.info("Saved pose visualization to %s", save_path)

        return vis

    def _empty_results(self) -> dict[str, Any]:
        """Return empty pose results dict."""
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "keypoints": np.empty((0, self.pose_model.num_keypoints, 3), dtype=np.float32),
            "keypoint_scores": np.empty((0,), dtype=np.float32),
            "keypoint_names": self.pose_model.keypoint_names,
            "skeleton": self.pose_model.skeleton,
        }
