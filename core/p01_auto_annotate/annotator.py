"""Auto-annotation engine — thin REST client for the auto-label service (s18104).

Delegates all annotation logic (text/auto/hybrid modes, NMS, polygon extraction)
to the auto-label service at ``http://localhost:18104``. The service in turn
calls SAM3 for segmentation.

Also retains the ``mask_to_polygon`` utility for direct use in tests.
"""

import base64
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

import cv2

# Default service URL and timeout
DEFAULT_SERVICE_URL = "http://localhost:18104"
DEFAULT_TIMEOUT = 120


def mask_to_polygon(mask: np.ndarray, img_h: int, img_w: int,
                    simplify_tolerance: float = 2.0,
                    min_vertices: int = 4) -> list[float] | None:
    """Convert a boolean mask to a simplified polygon (normalized vertices).

    Uses cv2.findContours + cv2.approxPolyDP to extract and simplify
    the mask contour.

    Args:
        mask: Boolean array of shape ``(H, W)``.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        simplify_tolerance: Epsilon for cv2.approxPolyDP (in pixels).
        min_vertices: Minimum number of vertices for a valid polygon.

    Returns:
        Flat list of normalized ``[x1, y1, x2, y2, ..., xN, yN]`` or None
        if the mask produces no valid contour.
    """
    if cv2 is None:
        logger.warning("cv2 not available — cannot convert mask to polygon")
        return None

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Use the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Simplify
    epsilon = simplify_tolerance
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) < min_vertices:
        return None

    # Normalize coordinates
    polygon: list[float] = []
    for point in approx:
        x, y = point[0]
        polygon.append(round(float(x) / img_w, 6))
        polygon.append(round(float(y) / img_h, 6))

    return polygon


class Annotator:
    """REST client for the auto-label service (s18104).

    Sends images to ``POST /annotate`` on the auto-label service, which
    handles SAM3 calls, NMS, polygon extraction, and format conversion
    internally.

    Args:
        class_names: Mapping of class_id to class name.
        text_prompts: Mapping of class name to text prompt.
            Falls back to class name if not specified.
        mode: Annotation mode — ``"text"``, ``"auto"``, or ``"hybrid"``.
        confidence_threshold: Minimum score to keep a detection.
        nms_iou_threshold: IoU threshold for NMS (applied server-side).
        service_url: Auto-label service URL (default: ``http://localhost:18104``).
        timeout: Request timeout in seconds (default: 120).
    """

    def __init__(
        self,
        class_names: dict[int, str],
        text_prompts: dict[str, str],
        mode: str = "text",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        service_url: str = DEFAULT_SERVICE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        detection_classes: dict[str, str] | None = None,
        class_rules: list[dict[str, Any]] | None = None,
        vlm_verify: dict[str, Any] | None = None,
    ) -> None:
        self.class_names = class_names
        self.text_prompts = text_prompts
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.detection_classes = detection_classes
        self.class_rules = class_rules
        self.vlm_verify = vlm_verify

    def annotate_image(
        self,
        image_path: Path,
        output_format: str = "bbox",
    ) -> list[dict[str, Any]]:
        """Generate annotations for a single image via the auto-label service.

        Args:
            image_path: Path to the image file.
            output_format: ``"bbox"``, ``"polygon"``, or ``"both"``.

        Returns:
            List of detection dicts with keys: class_id, cx, cy, w, h,
            score, and optionally polygon.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            return []

        # Read and encode image as base64
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        # Map output_format to service format
        # "bbox" -> "yolo", "polygon" -> "yolo_seg", "both" -> "yolo" (polygon in detections)
        svc_output_format = "yolo_seg" if output_format == "polygon" else "yolo"

        # Build request payload
        # Service expects class keys as strings in JSON
        classes_str_keys = {str(k): v for k, v in self.class_names.items()}
        payload: dict[str, Any] = {
            "image": image_b64,
            "classes": classes_str_keys,
            "text_prompts": self.text_prompts,
            "mode": self.mode,
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "output_format": svc_output_format,
            "include_masks": False,
        }

        # Optional: rule-based classification + VLM verification (handled by service)
        if self.detection_classes is not None:
            payload["detection_classes"] = self.detection_classes
        if self.class_rules is not None:
            payload["class_rules"] = self.class_rules
        if self.vlm_verify is not None:
            payload["vlm_verify"] = self.vlm_verify

        try:
            resp = requests.post(
                f"{self.service_url}/annotate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            logger.error(
                "Cannot connect to auto-label service at %s. "
                "Start it with: cd services/s18104_auto_label && docker compose up -d",
                self.service_url,
            )
            return []
        except requests.RequestException as e:
            logger.error("Auto-label service request failed: %s", e)
            return []

        data = resp.json()
        detections_raw = data.get("detections", [])

        # Convert service Detection objects to pipeline detection dicts
        detections: list[dict[str, Any]] = []
        for det in detections_raw:
            bbox_norm = det.get("bbox_norm", [0, 0, 0, 0])
            cx, cy, w, h = bbox_norm

            result: dict[str, Any] = {
                "class_id": det["class_id"],
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "score": det["score"],
            }

            # Include polygon if present and requested
            polygon_data = det.get("polygon", [])
            if output_format in ("polygon", "both") and polygon_data:
                # Service returns polygon as [[x,y], ...] pairs (normalized)
                # Pipeline expects flat list [x1, y1, x2, y2, ...]
                flat_polygon: list[float] = []
                for pt in polygon_data:
                    flat_polygon.append(round(float(pt[0]), 6))
                    flat_polygon.append(round(float(pt[1]), 6))
                result["polygon"] = flat_polygon if len(flat_polygon) >= 6 else None
            else:
                result["polygon"] = None

            detections.append(result)

        return detections

    def unload(self) -> None:
        """No-op — the service manages its own resources."""
        pass
