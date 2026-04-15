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
logger = logging.getLogger("auto_label")

# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.environ.get(
    "AUTO_LABEL_CONFIG",
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
REQUEST_TIMEOUT = config.get("request_timeout", 120)

# Annotation defaults
_annot_cfg = config.get("annotation", {})
DEFAULT_CONFIDENCE = _annot_cfg.get("confidence_threshold", 0.5)
DEFAULT_NMS_IOU = _annot_cfg.get("nms_iou_threshold", 0.5)
DEFAULT_MODE = _annot_cfg.get("mode", "text")
DEFAULT_OUTPUT_FORMAT = _annot_cfg.get("output_format", "coco")
DEFAULT_INCLUDE_MASKS = _annot_cfg.get("include_masks", True)

# Polygon extraction
_poly_cfg = config.get("polygon", {})
SIMPLIFY_TOLERANCE = _poly_cfg.get("simplify_tolerance", 2.0)
MIN_POLYGON_VERTICES = _poly_cfg.get("min_vertices", 4)

# NMS
_nms_cfg = config.get("nms", {})
CROSS_CLASS_NMS_ENABLED = _nms_cfg.get("cross_class_enabled", False)
CROSS_CLASS_NMS_THRESHOLD = _nms_cfg.get("cross_class_threshold", 0.7)

# Jobs
_jobs_cfg = config.get("jobs", {})
MAX_CONCURRENT_JOBS = _jobs_cfg.get("max_concurrent_jobs", 2)
JOB_TTL_SECONDS = _jobs_cfg.get("ttl_seconds", 3600)

# Video sessions
_video_cfg = config.get("video_sessions", {})
MAX_VIDEO_SESSIONS = _video_cfg.get("max_active", 10)
VIDEO_SESSION_TTL = _video_cfg.get("ttl_seconds", 1800)

# Batch processing
_proc_cfg = config.get("processing", {})
BATCH_CONCURRENCY = _proc_cfg.get("batch_concurrency", 4)
USE_BATCH_ENDPOINTS = _proc_cfg.get("use_batch_endpoints", True)
