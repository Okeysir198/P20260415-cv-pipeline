"""Configuration, constants, and logger setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("annotation_quality_assessment")

# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.environ.get(
    "ANNOTATION_QUALITY_ASSESSMENT_CONFIG",
    str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
)
with open(_CONFIG_PATH, "r") as _f:
    config: dict = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAM3_URL = os.environ.get(
    "SAM3_URL", config.get("services", {}).get("sam3_url", "http://localhost:18100")
).rstrip("/")
REQUEST_TIMEOUT = config.get("processing", {}).get("request_timeout", 120)

# Validation defaults
_val_cfg = config.get("validation", {})
DEFAULT_MIN_BOX_SIZE: float = _val_cfg.get("min_box_size", 0.005)
DEFAULT_MAX_BOX_SIZE: float = _val_cfg.get("max_box_size", 0.95)
DEFAULT_DUPLICATE_IOU_THRESH: float = _val_cfg.get("duplicate_iou_threshold", 0.95)
DEFAULT_MAX_ASPECT_RATIO: float = _val_cfg.get("max_aspect_ratio", 20.0)
DEFAULT_POLY_BBOX_AREA_MIN: float = _val_cfg.get("polygon_bbox_area_ratio_min", 0.1)
DEFAULT_POLY_BBOX_AREA_MAX: float = _val_cfg.get("polygon_bbox_area_ratio_max", 2.0)

# SAM3 config
_sam3_cfg = config.get("sam3", {})
DEFAULT_TEXT_VERIFY_THRESHOLD: float = _sam3_cfg.get("text_verify_threshold", 0.6)
DEFAULT_AUTO_MASK_MIN_AREA: float = _sam3_cfg.get("auto_mask_min_area", 0.001)
DEFAULT_AUTO_MASK_MAX_AREA: float = _sam3_cfg.get("auto_mask_max_area", 0.8)
DEFAULT_MISSING_OVERLAP_THRESH: float = _sam3_cfg.get("missing_overlap_threshold", 0.3)

# Scoring config
_scoring_cfg = config.get("scoring", {})
DEFAULT_SCORING_WEIGHTS: dict = _scoring_cfg.get("weights", {
    "structural": 0.3, "bbox_quality": 0.3, "classification": 0.2, "coverage": 0.2,
})
DEFAULT_SCORING_THRESHOLDS: dict = _scoring_cfg.get("thresholds", {
    "good": 0.8, "review": 0.5,
})

# VLM config
_vlm_cfg = config.get("vlm", {})
OLLAMA_URL = os.environ.get(
    "OLLAMA_URL", config.get("services", {}).get("ollama_url", "http://localhost:11434")
).rstrip("/")
OLLAMA_MODEL: str = _vlm_cfg.get("model", "qwen3.5:9b")
DEFAULT_VLM_TRIGGER: str = _vlm_cfg.get("trigger", "selective")
DEFAULT_VLM_CROP_PADDING: float = _vlm_cfg.get("crop_padding", 0.05)
DEFAULT_VLM_REQUEST_TIMEOUT: int = _vlm_cfg.get("request_timeout", 120)
VLM_CROP_PROMPT_TEMPLATE: str = _vlm_cfg.get("crop_prompt", (
    "Look at this cropped image region from an object detection dataset.\n"
    "Is this a {class_name}?\n"
    "Answer format: YES/NO | confidence (0-1) | brief reason"
))
VLM_SCENE_PROMPT_TEMPLATE: str = _vlm_cfg.get("scene_prompt", (
    "This image has the following annotations:\n"
    "{annotation_list}\n"
    "Available classes: {class_list}\n"
    "Check: 1) Incorrectly labeled? 2) Missing objects? 3) Quality 0-1.\n"
    "Format:\nINCORRECT: [indices] or NONE\nMISSING: [descriptions] or NONE\nQUALITY: [0-1]"
))

# Job config
_jobs_cfg = config.get("jobs", {})
MAX_CONCURRENT_JOBS = _jobs_cfg.get("max_concurrent_jobs", 2)
JOB_TTL_SECONDS = _jobs_cfg.get("ttl_seconds", 3600)

# Reporting config
_report_cfg = config.get("reporting", {})
DEFAULT_WORST_IMAGES_COUNT: int = _report_cfg.get("worst_images_count", 50)
