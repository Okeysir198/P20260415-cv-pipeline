"""Configuration, constants, and logger setup."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from loguru import logger

# YAML config
_CONFIG_PATH = os.environ.get(
    "SAM3_CONFIG",
    str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
)
with open(_CONFIG_PATH, "r") as _f:
    config: dict = yaml.safe_load(_f)

# Auto-mask prompts
AUTO_MASK_PROMPTS = config["segmentation"].get("auto_mask_prompts", [
    "person. car. truck. dog. cat. bird.",
    "chair. table. laptop. phone. bottle. cup.",
    "tree. building. road. sky. door. window.",
    "fire. smoke. helmet. shoe. bag. sign.",
])
