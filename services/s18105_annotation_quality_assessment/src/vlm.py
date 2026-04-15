"""VLM verification via ChatOpenAI -> Ollama + async LangGraph flow."""

from __future__ import annotations

import asyncio
import base64
import io
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from PIL import Image

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    VLM_CROP_PROMPT_TEMPLATE,
    VLM_SCENE_PROMPT_TEMPLATE,
    DEFAULT_VLM_CROP_PADDING,
    DEFAULT_VLM_REQUEST_TIMEOUT,
    logger,
)
from src.geometry import decode_image, norm_cxcywh_to_xyxy, norm_xyxy_to_pixel
from src.schemas import (
    VLMCropResult,
    VLMCropVerification,
    VLMSceneVerification,
    VLMVerification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm(cfg: dict) -> ChatOpenAI:
    """Build ChatOpenAI pointing at Ollama's OpenAI-compatible endpoint."""
    url = cfg.get("ollama_url", OLLAMA_URL)
    model = cfg.get("model", OLLAMA_MODEL)
    timeout = cfg.get("request_timeout", DEFAULT_VLM_REQUEST_TIMEOUT)
    return ChatOpenAI(
        base_url=f"{url.rstrip('/')}/v1",
        api_key="ollama",  # required by ChatOpenAI but ignored by Ollama
        model=model,
        temperature=0.0,
        request_timeout=timeout,
    )


def _crop_from_image(
    img: Image.Image,
    bbox_norm: list[float],
    img_w: int,
    img_h: int,
    padding: float,
) -> str:
    """Crop a bbox region from a PIL Image. Returns base64 JPEG."""
    xyxy = norm_cxcywh_to_xyxy(*bbox_norm)
    # Apply padding in normalized space then convert to pixel
    pad_w = (xyxy[2] - xyxy[0]) * padding
    pad_h = (xyxy[3] - xyxy[1]) * padding
    padded = [xyxy[0] - pad_w, xyxy[1] - pad_h, xyxy[2] + pad_w, xyxy[3] + pad_h]
    # Clip to [0, 1]
    padded = [max(0.0, min(1.0, v)) for v in padded]
    px = norm_xyxy_to_pixel(padded, img_w, img_h)

    cropped = img.crop((px[0], px[1], px[2], px[3]))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def crop_annotation(
    image_b64: str,
    bbox_norm: list[float],
    img_w: int,
    img_h: int,
    padding: float = DEFAULT_VLM_CROP_PADDING,
) -> str:
    """Crop a bounding box region from a base64-encoded image.

    Returns:
        Base64-encoded JPEG of the cropped region.
    """
    img = decode_image(image_b64)
    return _crop_from_image(img, bbox_norm, img_w, img_h, padding)


def parse_crop_response(text: str) -> tuple[bool, float, str]:
    """Parse ``YES/NO | confidence (0-1) | brief reason``. Lenient."""
    text = text.strip()
    upper = text.upper()
    if "YES" in upper:
        is_correct = True
    elif "NO" in upper:
        is_correct = False
    else:
        return False, 0.0, text

    floats = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    confidence = float(floats[0]) if floats else (0.9 if is_correct else 0.1)

    parts = text.split("|")
    reason = parts[-1].strip() if len(parts) >= 3 else text

    return is_correct, confidence, reason


def parse_scene_response(text: str) -> tuple[list[int], list[str], float]:
    """Parse INCORRECT/MISSING/QUALITY scene response. Lenient."""
    text = text.strip()

    incorrect: list[int] = []
    inc_match = re.search(r"INCORRECT\s*:\s*(.+?)(?:\n|MISSING|QUALITY|$)", text, re.IGNORECASE)
    if inc_match:
        inc_text = inc_match.group(1).strip()
        if "NONE" not in inc_text.upper():
            incorrect = [int(x) for x in re.findall(r"\d+", inc_text)]

    missing: list[str] = []
    miss_match = re.search(r"MISSING\s*:\s*(.+?)(?:\n|QUALITY|$)", text, re.IGNORECASE)
    if miss_match:
        miss_text = miss_match.group(1).strip()
        if "NONE" not in miss_text.upper():
            bracket = re.search(r"\[(.+?)]", miss_text)
            if bracket:
                items = [s.strip().strip("'\"") for s in bracket.group(1).split(",")]
                missing = [s for s in items if s]
            elif miss_text:
                missing = [miss_text]

    quality = 0.0
    qual_match = re.search(r"QUALITY\s*:\s*(\d*\.?\d+)", text, re.IGNORECASE)
    if qual_match:
        quality = min(1.0, max(0.0, float(qual_match.group(1))))

    return incorrect, missing, quality


# ---------------------------------------------------------------------------
# LangGraph state & nodes
# ---------------------------------------------------------------------------


class VLMState(TypedDict, total=False):
    image_b64: str
    annotations: list[dict]  # [{"class_id": int, "bbox_norm": [cx,cy,w,h]}]
    classes: dict[int, str]
    cfg: dict
    img_w: int
    img_h: int
    crop_results: list[dict]
    scene_result: dict
    verification: dict
    vlm_budget: dict | None
    derived_class_ids: set[int] | None


async def _verify_single_crop(
    llm: ChatOpenAI,
    img: Image.Image,
    idx: int,
    ann: dict,
    classes: dict[int, str],
    img_w: int,
    img_h: int,
    padding: float,
) -> dict:
    """Verify a single annotation crop against the VLM."""
    class_id = ann["class_id"]
    class_name = classes.get(class_id, f"class_{class_id}")

    try:
        crop_b64 = _crop_from_image(img, ann["bbox_norm"], img_w, img_h, padding)
    except Exception:
        logger.warning("Failed to crop annotation %d, skipping", idx)
        # Fail-open: mark as correct with zero confidence so it doesn't
        # penalize the score when VLM infrastructure is unreliable.
        return {
            "annotation_idx": idx, "class_name": class_name,
            "is_correct": True, "confidence": 0.0, "reason": "crop failed",
        }

    prompt = VLM_CROP_PROMPT_TEMPLATE.format(class_name=class_name)
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
        {"type": "text", "text": prompt},
    ])

    try:
        response = await llm.ainvoke([msg])
        is_correct, confidence, reason = parse_crop_response(response.content)
    except Exception as exc:
        logger.warning("VLM crop verify failed for annotation %d: %s", idx, exc)
        # Fail-open: treat as correct with zero confidence so transient VLM
        # errors don't flag good annotations as bad.
        is_correct, confidence, reason = True, 0.0, f"VLM error: {exc}"

    return {
        "annotation_idx": idx, "class_name": class_name,
        "is_correct": is_correct, "confidence": confidence, "reason": reason,
    }


def select_priority_annotations(
    annotations: list[dict],
    vlm_budget: dict,
    derived_class_ids: set[int] | None = None,
) -> list[int]:
    """Return indices of annotations to verify, ranked by priority.

    Priority per annotation:
        0.4 * (1 - score/0.5) + 0.3 * small_area + 0.3 * is_derived

    Args:
        annotations: List of annotation dicts with ``bbox_norm`` and optionally ``score``.
        vlm_budget: Budget config ``{sample_rate, max_samples, priority}``.
        derived_class_ids: Set of class IDs produced by overlap/no_overlap rules.

    Returns:
        Sorted list of annotation indices to verify.
    """
    total = len(annotations)
    if total == 0:
        return []

    sample_rate = vlm_budget.get("sample_rate", 1.0)
    max_samples = vlm_budget.get("max_samples", total)
    n = min(int(sample_rate * total + 0.5), max_samples, total)
    n = max(n, 1)  # always verify at least one

    derived = derived_class_ids or set()

    priorities: list[tuple[float, int]] = []
    for idx, ann in enumerate(annotations):
        score = ann.get("score", 0.5)
        # Low-confidence annotations are more valuable to verify
        score_term = 0.4 * (1.0 - min(score, 0.5) / 0.5)

        # Small objects are harder to annotate
        bbox = ann.get("bbox_norm", [0, 0, 0, 0])
        area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0.0
        small_area = 0.3 * (1.0 if area < 0.01 else 0.0)

        # Derived classes have inherent uncertainty
        is_derived = 0.3 * (1.0 if ann.get("class_id") in derived else 0.0)

        priority = score_term + small_area + is_derived
        priorities.append((priority, idx))

    # Sort descending by priority, pick top-N
    priorities.sort(key=lambda x: x[0], reverse=True)
    selected = sorted([idx for _, idx in priorities[:n]])
    return selected


async def crop_verify_node(state: VLMState) -> dict:
    """Crop each annotated region and ask the VLM if the label is correct.

    Runs all crop verifications concurrently via asyncio.gather.
    When a VLM budget is provided, only selected annotations are verified;
    the rest are passed through as ``is_correct=True``.
    """
    cfg = state["cfg"]
    llm = _build_llm(cfg)
    padding = cfg.get("crop_padding", DEFAULT_VLM_CROP_PADDING)

    vlm_budget: dict = state.get("vlm_budget", {})
    derived_class_ids = state.get("derived_class_ids")

    selected_indices = select_priority_annotations(
        state["annotations"], vlm_budget, derived_class_ids,
    )
    selected_set = set(selected_indices)

    # Decode image once for all crops
    img = decode_image(state["image_b64"])

    # Build pass-through results for non-selected annotations
    passthrough: dict[int, dict] = {}
    for idx, ann in enumerate(state["annotations"]):
        if idx not in selected_set:
            class_name = state["classes"].get(ann["class_id"], f"class_{ann['class_id']}")
            passthrough[idx] = {
                "annotation_idx": idx,
                "class_name": class_name,
                "is_correct": True,
                "confidence": 0.0,
                "reason": "skipped by vlm_budget",
            }

    tasks = [
        _verify_single_crop(llm, img, idx, ann, state["classes"],
                            state["img_w"], state["img_h"], padding)
        for idx, ann in enumerate(state["annotations"])
        if idx in selected_set
    ]
    verified = await asyncio.gather(*tasks)

    # Merge verified + passthrough in original order
    result_map = {r["annotation_idx"]: r for r in verified}
    result_map.update(passthrough)
    results = [result_map[i] for i in range(len(state["annotations"])) if i in result_map]

    return {"crop_results": results}


async def scene_verify_node(state: VLMState) -> dict:
    """Send the full image with annotation list for scene-level verification."""
    cfg = state["cfg"]
    llm = _build_llm(cfg)

    ann_lines: list[str] = []
    for idx, ann in enumerate(state["annotations"]):
        class_name = state["classes"].get(ann["class_id"], f"class_{ann['class_id']}")
        ann_lines.append(f"  [{idx}] {class_name} bbox={ann['bbox_norm']}")

    annotation_list = "\n".join(ann_lines) if ann_lines else "(no annotations)"
    class_list = ", ".join(
        f"{cid}: {cname}" for cid, cname in sorted(state["classes"].items())
    )

    prompt = VLM_SCENE_PROMPT_TEMPLATE.format(
        annotation_list=annotation_list, class_list=class_list,
    )

    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
        {"type": "text", "text": prompt},
    ])

    try:
        response = await llm.ainvoke([msg])
        raw = response.content
        incorrect, missing, quality = parse_scene_response(raw)
    except Exception as exc:
        logger.warning("VLM scene verify failed: %s", exc)
        raw = f"VLM error: {exc}"
        incorrect, missing, quality = [], [], 0.0

    return {
        "scene_result": {
            "incorrect_indices": incorrect,
            "missing_descriptions": missing,
            "quality_score": quality,
            "raw_response": raw,
        }
    }


def combine_node(state: VLMState) -> dict:
    """Merge crop and scene results into a VLMVerification dict."""
    crop_results = state.get("crop_results", [])
    scene_result = state.get("scene_result", {})

    num_checked = len(crop_results)
    num_incorrect = sum(1 for r in crop_results if not r.get("is_correct", True))
    confidences = [r.get("confidence", 0.0) for r in crop_results]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {"verification": {
        "crop_verification": {
            "results": crop_results,
            "num_checked": num_checked,
            "num_incorrect": num_incorrect,
            "mean_confidence": mean_confidence,
        },
        "scene_verification": {
            "incorrect_indices": scene_result.get("incorrect_indices", []),
            "missing_descriptions": scene_result.get("missing_descriptions", []),
            "quality_score": scene_result.get("quality_score", 0.0),
            "raw_response": scene_result.get("raw_response", ""),
        },
        "available": True,
    }}


# ---------------------------------------------------------------------------
# Graph builder & public API
# ---------------------------------------------------------------------------

# Compiled graph singleton — topology is static, only state varies per call.
_compiled_graph = None


def _get_compiled_graph():
    """Return the cached compiled VLM graph (built once)."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = StateGraph(VLMState)
        graph.add_node("crop_verify", crop_verify_node)
        graph.add_node("scene_verify", scene_verify_node)
        graph.add_node("combine", combine_node)

        # crop_verify and scene_verify are independent — fan out from START
        graph.add_edge(START, "crop_verify")
        graph.add_edge(START, "scene_verify")
        graph.add_edge("crop_verify", "combine")
        graph.add_edge("scene_verify", "combine")
        graph.add_edge("combine", END)

        _compiled_graph = graph.compile()
    return _compiled_graph


async def verify_with_vlm(
    image_b64: str,
    annotations: list[dict],
    classes: dict[int, str],
    cfg: dict,
    img_w: int,
    img_h: int,
    vlm_budget: dict | None = None,
    derived_class_ids: set[int] | None = None,
) -> VLMVerification:
    """Run VLM verification on an image with its annotations.

    Args:
        vlm_budget: Optional priority sampling budget ``{sample_rate, max_samples, priority}``.
        derived_class_ids: Class IDs produced by overlap/no_overlap rules.

    Returns ``VLMVerification(available=False)`` on any error.
    """
    try:
        compiled = _get_compiled_graph()
        result = await compiled.ainvoke({
            "image_b64": image_b64,
            "annotations": annotations,
            "classes": classes,
            "cfg": cfg,
            "img_w": img_w,
            "img_h": img_h,
            "crop_results": [],
            "scene_result": {},
            "verification": {},
            "vlm_budget": vlm_budget,
            "derived_class_ids": derived_class_ids,
        })

        v = result["verification"]
        crop_v = v.get("crop_verification", {})
        scene_v = v.get("scene_verification", {})

        return VLMVerification(
            crop_verification=VLMCropVerification(
                results=[VLMCropResult(**r) for r in crop_v.get("results", [])],
                num_checked=crop_v.get("num_checked", 0),
                num_incorrect=crop_v.get("num_incorrect", 0),
                mean_confidence=crop_v.get("mean_confidence", 0.0),
            ),
            scene_verification=VLMSceneVerification(
                incorrect_indices=scene_v.get("incorrect_indices", []),
                missing_descriptions=scene_v.get("missing_descriptions", []),
                quality_score=scene_v.get("quality_score", 0.0),
                raw_response=scene_v.get("raw_response", ""),
            ),
            available=True,
        )
    except Exception as exc:
        logger.warning("VLM verification failed: %s", exc)
        return VLMVerification(available=False)
