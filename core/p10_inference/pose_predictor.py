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
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.p06_models.pose_base import PoseModel
from core.p10_inference.predictor import DetectionPredictor
from utils.viz import VizStyle, annotate_detections, annotate_keypoints

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
        boxes = predictions["boxes"]
        keypoints = predictions["keypoints"]
        skeleton = predictions["skeleton"]

        # Preserve original BGR palette: box=blue, bones=green, vertices=red.
        # Helpers operate in RGB; convert at the boundary.
        box_color_rgb = sv.Color(r=0, g=0, b=255)       # was BGR (255,0,0) = blue
        kpt_color_rgb = sv.Color(r=255, g=0, b=0)       # was BGR (0,0,255) = red
        skeleton_color_rgb = (0, 255, 0)                # was BGR (0,255,0) = green

        style = VizStyle(
            kpt_visibility_threshold=keypoint_threshold,
            skeleton_color_rgb=skeleton_color_rgb,
        )

        vis_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(boxes) > 0:
            dets = sv.Detections(
                xyxy=np.asarray(boxes, dtype=np.float32),
                class_id=np.zeros(len(boxes), dtype=int),
            )
            vis_rgb = annotate_detections(
                vis_rgb, dets, labels=[""] * len(boxes), style=style, color=box_color_rgb,
            )

            # Keypoints: (N, K, 3) → xy (N,K,2) + conf (N,K).
            kpts_arr = np.asarray(keypoints, dtype=np.float32)
            if kpts_arr.size > 0:
                xy = kpts_arr[..., :2]
                conf = kpts_arr[..., 2] if kpts_arr.shape[-1] >= 3 else None
                # Edges drawn in skeleton_color via style; vertices overridden to red.
                vis_rgb = annotate_keypoints(
                    vis_rgb, xy, skeleton_edges=list(skeleton), confidence=conf, style=style,
                )
                # Re-draw vertices in red to preserve original kpt color.
                vis_rgb = annotate_keypoints(
                    vis_rgb, xy, skeleton_edges=None, confidence=conf,
                    style=style, color=kpt_color_rgb,
                )

        vis = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

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
