"""VLM verification: use Ollama vision model to verify uncertain detections."""

from __future__ import annotations

import base64
import io
import logging

import requests
from PIL import Image

from src.schemas import Detection

logger = logging.getLogger("auto_label")


def _ask_ollama(ollama_url: str, model: str, image_b64: str, prompt: str, timeout: int) -> str:
    """Call Ollama chat API with image."""
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],  # Ollama native format
                }
            ],
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _crop_detection(img: Image.Image, det: Detection, padding: float = 0.15) -> str:
    """Crop detection bbox with padding, return base64 JPEG."""
    x1, y1, x2, y2 = det.bbox_xyxy
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * padding), int(h * padding)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img.width, x2 + pad_x)
    cy2 = min(img.height, y2 + pad_y)
    cropped = img.crop((cx1, cy1, cx2, cy2))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _compute_priority(
    det: Detection,
    derived_class_ids: set[int],
    img_w: int,
    img_h: int,
) -> float:
    """Compute priority score for VLM verification (higher = more uncertain).

    Factors: low confidence, small area, derived class (from overlap/no_overlap rules).
    """
    # Confidence: lower confidence -> higher priority
    conf_score = 1.0 - det.score

    # Area: smaller detections are more uncertain
    x1, y1, x2, y2 = det.bbox_xyxy
    area_frac = ((x2 - x1) * (y2 - y1)) / (img_w * img_h + 1e-6)
    area_score = 1.0 - min(area_frac * 10, 1.0)  # scale up, cap at 1.0

    # Derived class: overlap/no_overlap rules produce more uncertain labels
    derived_score = 1.0 if det.class_id in derived_class_ids else 0.0

    return 0.4 * conf_score + 0.3 * area_score + 0.3 * derived_score


def verify_detections_vlm(
    image_b64: str,
    detections: list[Detection],
    vlm_config: dict,
    final_classes: dict[int, str],
    derived_class_ids: set[int],
    img_w: int,
    img_h: int,
) -> list[Detection]:
    """Verify detections using a VLM via Ollama. Fail-open on all errors.

    Args:
        image_b64: Base64-encoded full image.
        detections: Current detection list.
        vlm_config: Config dict with keys:
            - model: str — Ollama model name (e.g. "llava")
            - ollama_url: str — Ollama base URL (e.g. "http://localhost:11434")
            - verify_classes: list[int] | None — class IDs to verify (None = all)
            - priority: dict | None — unused (priority is computed automatically)
            - budget: dict — {sample_rate: float, max_samples: int}
            - vlm_min_confidence: float — VLM rejection threshold (default 0.7)
            - timeout: int — per-request timeout in seconds (default 30)
        final_classes: Maps class ID -> class name.
        derived_class_ids: Class IDs produced by overlap/no_overlap rules.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Filtered detection list (VLM-rejected detections removed).
    """
    if not detections:
        return detections

    model = vlm_config.get("model", "llava")
    ollama_url = vlm_config.get("ollama_url", "http://localhost:11434")
    verify_classes = vlm_config.get("verify_classes")  # None means all
    budget = vlm_config.get("budget", {})
    sample_rate = float(budget.get("sample_rate", 0.3))
    max_samples = int(budget.get("max_samples", 10))
    vlm_min_confidence = float(vlm_config.get("vlm_min_confidence", 0.7))
    timeout = int(vlm_config.get("timeout", 30))

    # Filter to verifiable detections
    if verify_classes is not None:
        verify_set = set(verify_classes)
        candidates = [(i, det) for i, det in enumerate(detections) if det.class_id in verify_set]
    else:
        candidates = list(enumerate(detections))

    if not candidates:
        return detections

    # Compute priority and select top N
    priorities = [
        (idx, det, _compute_priority(det, derived_class_ids, img_w, img_h))
        for idx, det in candidates
    ]
    priorities.sort(key=lambda x: x[2], reverse=True)

    n_samples = min(max_samples, max(1, int(len(candidates) * sample_rate)))
    selected = priorities[:n_samples]

    # Decode image once for all crops
    try:
        raw_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        logger.warning("VLM verify: failed to decode image, skipping verification")
        return detections

    # Verify selected detections
    reject_indices: set[int] = set()
    consecutive_errors = 0
    max_consecutive_errors = 3  # circuit breaker

    for idx, det, _priority in selected:
        if consecutive_errors >= max_consecutive_errors:
            logger.warning("VLM verify: circuit breaker triggered after %d consecutive errors", max_consecutive_errors)
            break

        class_name = final_classes.get(det.class_id, det.class_name)

        try:
            crop_b64 = _crop_detection(img, det)
            prompt = (
                f"Is this a {class_name}? Answer only YES or NO. "
                f"If you are not sure, answer YES."
            )
            response = _ask_ollama(ollama_url, model, crop_b64, prompt, timeout)
            consecutive_errors = 0  # reset on success
        except Exception as exc:
            logger.warning("VLM verify: Ollama call failed for detection %d: %s", idx, exc)
            consecutive_errors += 1
            continue  # fail-open: keep detection on error

        # Parse response
        answer = response.strip().upper()
        if answer.startswith("NO"):
            # VLM says this is not the claimed class — reject if detection confidence
            # is below the VLM min confidence threshold
            if det.score < vlm_min_confidence:
                reject_indices.add(idx)
                logger.info(
                    "VLM verify: rejected detection %d (class=%s, score=%.3f) — VLM said NO",
                    idx, class_name, det.score,
                )

    # Filter out rejected detections
    if not reject_indices:
        return detections

    return [det for i, det in enumerate(detections) if i not in reject_indices]
