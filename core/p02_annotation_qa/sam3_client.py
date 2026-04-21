"""SAM3 client — REST wrapper for the SAM3 service (s18100).

Provides the same interface as the previous in-process client but delegates
all inference to the SAM3 REST service.  No torch or transformers dependency.
"""

import base64
import io
import logging
from typing import Any

import httpx
import numpy as np
from PIL import Image

from utils.yolo_io import pil_to_b64

logger = logging.getLogger(__name__)


class SAM3Client:
    """REST client for the SAM3 segmentation service.

    Calls the SAM3 REST API at ``service_url`` instead of loading models
    in-process.  The public API (``segment_with_box``, ``segment_with_text``,
    ``auto_mask``) returns the same types as the previous local implementation.

    Args:
        service_url: Base URL of the SAM3 service (e.g. ``"http://localhost:18100"``).
        timeout: HTTP request timeout in seconds.

    Example::

        client = SAM3Client("http://localhost:18100")
        result = client.segment_with_box(image, [100, 200, 300, 400])
    """

    def __init__(self, service_url: str = "http://localhost:18100", timeout: float = 120) -> None:
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_with_box(self, image: Any, box: list[float]) -> dict:
        """Segment an object using a bounding-box prompt.

        Args:
            image: PIL Image to segment.
            box: Bounding box as ``[x1, y1, x2, y2]`` in absolute pixels.

        Returns:
            Dictionary with keys:
                - ``mask`` (np.ndarray): Boolean mask, shape ``(H, W)``.
                - ``bbox`` (tuple): Tight bounding box ``(cx, cy, w, h)`` normalised to ``[0, 1]``.
                - ``iou_score`` (float): Model-predicted IoU score for the mask.
        """
        payload = {
            "image": pil_to_b64(image),
            "box": [int(c) for c in box],
        }
        data = self._post("/segment_box", payload)

        result = data["result"]
        mask = self._decode_mask(result["mask"])
        img_h, img_w = mask.shape[:2]
        bbox = result["bbox"]
        bbox_norm = self._pixel_bbox_to_norm(bbox, img_w, img_h)

        return {
            "mask": mask,
            "bbox": bbox_norm,
            "iou_score": float(result.get("iou_score", result.get("score", 0.0))),
        }

    def segment_with_text(self, image: Any, text: str) -> list[dict]:
        """Segment objects matching a text prompt.

        Args:
            image: PIL Image to segment.
            text: Natural-language description of the target object(s).

        Returns:
            List of dictionaries, each with:
                - ``mask`` (np.ndarray): Boolean mask, shape ``(H, W)``.
                - ``bbox`` (tuple): Tight bounding box ``(cx, cy, w, h)`` normalised to ``[0, 1]``.
                - ``score`` (float): Detection confidence score.
        """
        payload = {
            "image": pil_to_b64(image),
            "text": text,
        }
        data = self._post("/segment_text", payload)

        detections: list[dict] = []
        for det in data.get("detections", []):
            mask = self._decode_mask(det["mask"])
            img_h, img_w = mask.shape[:2]
            bbox_norm = self._pixel_bbox_to_norm(det["bbox"], img_w, img_h)
            detections.append({
                "mask": mask,
                "bbox": bbox_norm,
                "score": float(det.get("score", 0.0)),
            })

        return detections

    def auto_mask(self, image: Any) -> list[dict]:
        """Generate all masks automatically (no prompt).

        Args:
            image: PIL Image to segment.

        Returns:
            List of dictionaries, each with:
                - ``mask`` (np.ndarray): Boolean mask, shape ``(H, W)``.
                - ``bbox`` (tuple): Tight bounding box ``(cx, cy, w, h)`` normalised to ``[0, 1]``.
                - ``area`` (float): Fraction of image covered by the mask.
                - ``score`` (float): Mask confidence score.
        """
        payload = {"image": pil_to_b64(image)}
        data = self._post("/auto_mask", payload)

        detections: list[dict] = []
        for det in data.get("detections", []):
            mask = self._decode_mask(det["mask"])
            img_h, img_w = mask.shape[:2]
            bbox_norm = self._pixel_bbox_to_norm(det["bbox"], img_w, img_h)
            area = float(det.get("area", 0.0))
            detections.append({
                "mask": mask,
                "bbox": bbox_norm,
                "area": area,
                "score": float(det.get("score", 0.0)),
            })

        return detections

    def unload(self, which: str = "all") -> None:
        """No-op — the service manages its own GPU memory."""
        logger.debug("unload(%s) called — no-op for REST client", which)

    # ------------------------------------------------------------------
    # Static helpers (kept for callers that use them directly)
    # ------------------------------------------------------------------

    @staticmethod
    def mask_to_bbox(mask: np.ndarray, img_h: int, img_w: int) -> tuple[float, float, float, float]:
        """Convert a boolean mask to a normalised bounding box.

        Args:
            mask: Boolean array of shape ``(H, W)``.
            img_h: Image height in pixels.
            img_w: Image width in pixels.

        Returns:
            ``(cx, cy, w, h)`` normalised to ``[0, 1]``.  Returns
            ``(0.0, 0.0, 0.0, 0.0)`` if the mask is empty.
        """
        xyxy = SAM3Client.mask_to_bbox_xyxy(mask)
        if xyxy == (0, 0, 0, 0):
            return (0.0, 0.0, 0.0, 0.0)

        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return (cx, cy, w, h)

    @staticmethod
    def mask_to_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Convert a boolean mask to an absolute-pixel bounding box.

        Args:
            mask: Boolean array of shape ``(H, W)``.

        Returns:
            ``(x1, y1, x2, y2)`` in absolute pixels.  Returns
            ``(0, 0, 0, 0)`` if the mask is empty.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return (0, 0, 0, 0)

        row_indices = np.where(rows)[0]
        y1, y2 = int(row_indices[0]), int(row_indices[-1])
        col_indices = np.where(cols)[0]
        x1, x2 = int(col_indices[0]), int(col_indices[-1])
        return (x1, y1, x2 + 1, y2 + 1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        """Send a POST request to the SAM3 service and return the JSON response."""
        url = f"{self._service_url}{endpoint}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _decode_mask(mask_b64: str) -> np.ndarray:
        """Decode a base64 grayscale PNG mask to a boolean numpy array."""
        mask_bytes = base64.b64decode(mask_b64)
        mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")
        return np.array(mask_pil) > 127

    @staticmethod
    def _pixel_bbox_to_norm(
        bbox: dict, img_w: int, img_h: int,
    ) -> tuple[float, float, float, float]:
        """Convert pixel bbox ``{x1, y1, x2, y2}`` to normalised ``(cx, cy, w, h)``."""
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return (cx, cy, w, h)
