"""VLM verification for auto-annotation pipeline.

Uses Qwen3.5 via Ollama to verify uncertain detections by cropping
the detection region and asking the VLM if the classification is correct.
Priority scoring focuses VLM budget on detections most likely to be wrong:
low confidence, small area, or derived from overlap rules.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CROP_PROMPT = (
    "Is this image showing a '{class_name}'? "
    "Answer: YES | confidence (0-1) | brief reason "
    "or NO | confidence (0-1) | brief reason"
)

_SCENE_PROMPT = (
    "This image is from a safety monitoring dataset.\n"
    "Available classes: {class_list}\n"
    "Are there any objects of these classes that are NOT annotated? "
    "List only MISSING objects.\n"
    "Format: MISSING: description1, description2 or NONE"
)


class VLMVerifier:
    """Verify detections using a VLM via Ollama with priority-based sampling.

    Args:
        model: Ollama model name (e.g., ``"qwen3.5:9b"``).
        ollama_url: Ollama server URL.
        class_names: Final class_id → class_name mapping.
        verify_classes: Class IDs to verify (others pass through).
        priority: Priority scoring config dict with keys:
            ``low_confidence_threshold`` (float), ``small_box_threshold`` (float),
            ``prioritize_derived`` (bool).
        budget: Sampling budget config dict with keys:
            ``sample_rate`` (float), ``max_samples`` (int).
        crop_padding: Fractional padding around crop.
        request_timeout: Timeout per VLM request in seconds.
        vlm_min_confidence: Minimum VLM confidence to act on its answer.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        verify_classes: list[int] | None = None,
        model: str = "qwen3.5:9b",
        ollama_url: str = "http://localhost:11434",
        priority: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        crop_padding: float = 0.15,
        request_timeout: int = 30,
        vlm_min_confidence: float = 0.7,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.class_names = class_names
        self.verify_classes = set(verify_classes or [])
        self.crop_padding = crop_padding
        self.request_timeout = request_timeout
        self.vlm_min_confidence = vlm_min_confidence

        p = priority or {}
        self.low_conf_threshold = p.get("low_confidence_threshold", 0.5)
        self.small_box_threshold = p.get("small_box_threshold", 0.02)
        self.prioritize_derived = p.get("prioritize_derived", True)

        b = budget or {}
        self.sample_rate = b.get("sample_rate", 0.10)
        self.max_samples = b.get("max_samples", 100)

        self._llm = None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @staticmethod
    def is_ollama_available(
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3.5:9b",
    ) -> bool:
        """Check Ollama is running AND the required model is loaded."""
        import json
        import urllib.request

        try:
            req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                model_names = [m.get("name", "") for m in data.get("models", [])]
                # Check if model (or its base name) is in the list
                base = model.split(":")[0]
                available = any(base in n for n in model_names)
                if not available:
                    logger.warning("Ollama running but model '%s' not found (have: %s)", model, model_names)
                return available
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Priority scoring
    # ------------------------------------------------------------------

    def _compute_priority(self, det: dict, derived_class_ids: set[int]) -> float:
        """Compute verification priority score (higher = verify first).

        Score components (0-1 each, weighted):
            0.4 * confidence_need — how much the detection needs verification
            0.3 * small_area — small boxes are more likely false positives
            0.3 * is_derived — derived classes (overlap/no_overlap) are less certain
        """
        score_val = det.get("score", 1.0)
        conf_need = max(0.0, 1.0 - score_val / self.low_conf_threshold) if self.low_conf_threshold > 0 else 0.0
        conf_need = min(conf_need, 1.0)

        area = det.get("w", 0) * det.get("h", 0)
        small_area = max(0.0, 1.0 - area / self.small_box_threshold) if self.small_box_threshold > 0 else 0.0
        small_area = min(small_area, 1.0)

        is_derived = 1.0 if (self.prioritize_derived and det.get("class_id") in derived_class_ids) else 0.0

        return 0.4 * conf_need + 0.3 * small_area + 0.3 * is_derived

    def _select_samples(
        self, detections: list[dict], derived_class_ids: set[int],
    ) -> list[int]:
        """Select detection indices to verify based on priority + budget.

        Returns indices into the detections list, sorted by priority (highest first).
        """
        candidates: list[tuple[float, int]] = []
        for idx, det in enumerate(detections):
            if det.get("class_id") not in self.verify_classes:
                continue
            priority = self._compute_priority(det, derived_class_ids)
            if priority > 0:
                candidates.append((priority, idx))

        candidates.sort(key=lambda x: -x[0])

        # Apply budget: min(sample_rate * total, max_samples)
        budget = min(
            int(len(detections) * self.sample_rate),
            self.max_samples,
        )
        budget = max(budget, 1)  # verify at least 1 if there are candidates

        return [idx for _, idx in candidates[:budget]]

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_detections(
        self,
        image_path: Path,
        detections: list[dict],
        derived_class_ids: set[int] | None = None,
    ) -> list[dict]:
        """Verify a subset of detections using VLM crop checks.

        Detections not selected for verification pass through unchanged.
        VLM-rejected detections (NO with confidence >= ``vlm_min_confidence``)
        are removed. All others are kept (fail-open).

        Args:
            image_path: Path to the source image.
            detections: List of detection dicts with ``class_id``, ``cx``, ``cy``,
                ``w``, ``h``, ``score``.
            derived_class_ids: Set of class IDs produced by rule-based classification
                (used for priority scoring).

        Returns:
            Filtered detection list.
        """
        if not detections or not self.verify_classes:
            return detections

        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open image %s: %s — keeping all detections", image_path, exc)
            return detections

        derived = derived_class_ids or set()
        verify_indices = set(self._select_samples(detections, derived))

        if not verify_indices:
            return detections

        logger.debug("VLM verifying %d/%d detections for %s", len(verify_indices), len(detections), image_path.name)

        kept: list[dict] = []
        removed = 0
        consecutive_errors = 0
        max_consecutive_errors = 3  # circuit breaker: stop VLM after 3 failures in a row

        for idx, det in enumerate(detections):
            if idx not in verify_indices:
                kept.append(det)
                continue

            # Circuit breaker: too many consecutive failures → skip remaining VLM calls
            if consecutive_errors >= max_consecutive_errors:
                kept.append(det)
                continue

            class_name = self.class_names.get(det["class_id"], f"class_{det['class_id']}")
            try:
                crop_b64 = self._crop_detection(img, det)
                is_correct, vlm_conf, reason = self._ask_vlm(crop_b64, class_name)
                consecutive_errors = 0  # reset on success
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors == max_consecutive_errors:
                    logger.warning("VLM failed %d times in a row — disabling for this image", max_consecutive_errors)
                else:
                    logger.warning("VLM crop verify failed: %s — keeping detection", exc)
                kept.append(det)
                continue

            if is_correct:
                det_copy = dict(det)
                det_copy["score"] = max(det.get("score", 0), vlm_conf)
                det_copy["vlm_verified"] = True
                kept.append(det_copy)
            elif vlm_conf >= self.vlm_min_confidence:
                # High-confidence rejection → remove
                logger.info("VLM rejected %s (vlm_conf=%.2f): %s", class_name, vlm_conf, reason)
                removed += 1
            else:
                # Low-confidence rejection → keep (fail-open)
                kept.append(det)

        if removed:
            logger.info("VLM removed %d false positives from %s", removed, image_path.name)

        return kept

    # ------------------------------------------------------------------
    # Crop + VLM call
    # ------------------------------------------------------------------

    def _crop_detection(self, img: Any, det: dict) -> str:
        """Crop detection bbox (normalized cx,cy,w,h) with padding. Returns base64 JPEG."""
        cx, cy, w, h = det["cx"], det["cy"], det["w"], det["h"]
        img_w, img_h = img.size

        pad_w = w * self.crop_padding
        pad_h = h * self.crop_padding
        x1 = max(0.0, cx - w / 2 - pad_w) * img_w
        y1 = max(0.0, cy - h / 2 - pad_h) * img_h
        x2 = min(1.0, cx + w / 2 + pad_w) * img_w
        y2 = min(1.0, cy + h / 2 + pad_h) * img_h

        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _get_llm(self) -> Any:
        """Lazy-init ChatOpenAI pointing at Ollama."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                base_url=f"{self.ollama_url}/v1",
                api_key="ollama",
                model=self.model,
                temperature=0.0,
                request_timeout=self.request_timeout,
            )
        return self._llm

    def _ask_vlm(self, crop_b64: str, class_name: str) -> tuple[bool, float, str]:
        """Send crop to Ollama VLM and parse YES/NO response."""
        from langchain_core.messages import HumanMessage

        llm = self._get_llm()
        prompt = _CROP_PROMPT.format(class_name=class_name)
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            {"type": "text", "text": prompt},
        ])
        response = llm.invoke([msg])
        return self.parse_response(response.content)

    @staticmethod
    def parse_response(text: str) -> tuple[bool, float, str]:
        """Parse ``YES/NO | confidence (0-1) | reason``. Lenient."""
        text = text.strip()
        upper = text.upper()

        if "YES" in upper:
            is_correct = True
        elif "NO" in upper:
            is_correct = False
        else:
            return True, 0.5, text  # ambiguous → fail-open

        floats = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
        confidence = float(floats[0]) if floats else (0.9 if is_correct else 0.1)

        parts = text.split("|")
        reason = parts[-1].strip() if len(parts) >= 3 else text

        return is_correct, confidence, reason
