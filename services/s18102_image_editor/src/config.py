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
logger = logging.getLogger("image_editor")

# YAML config
_CONFIG_PATH = os.environ.get(
    "IMAGE_EDITOR_CONFIG",
    str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
)
with open(_CONFIG_PATH, "r") as _f:
    config: dict = yaml.safe_load(_f)

# Service URLs (override via env)
FLUX_NIM_URL = os.environ.get(
    "FLUX_NIM_URL",
    config.get("services", {}).get("flux_nim_url", "http://localhost:18101"),
).rstrip("/")

SAM3_URL = os.environ.get(
    "SAM3_URL",
    config.get("services", {}).get("sam3_url", "http://localhost:18100"),
).rstrip("/")

# Request timeout
REQUEST_TIMEOUT = config.get("request_timeout", 120)
