"""Translate our YAML schema to upstream PaddleDetection / PaddleClas / PaddleSeg configs.

Each task family has its own translator. Returns a dict ready to feed into the
upstream Trainer/Engine. Reads minimum required fields and lets each upstream
default fill the rest — we only override what our schema cares about
(num_classes, dataset paths, epochs, batch_size, lr, save_dir).

Keep this file pure-Python (no paddle import). The translator runs in any venv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_our_yaml(config_path: Path) -> dict[str, Any]:
    """Load our 06_training_paddle_*.yaml + resolve the sibling 05_data.yaml."""
    config = yaml.safe_load(config_path.read_text())
    data_ref = config.get("data", {}).get("dataset_config")
    if data_ref:
        data_path = (config_path.parent / data_ref).resolve()
        config["_data_resolved"] = yaml.safe_load(data_path.read_text())
        config["_data_path"] = data_path
    config["_config_path"] = config_path
    return config


# ---------------------------------------------------------------------------
# Detection (PaddleDetection / ppdet)
# ---------------------------------------------------------------------------

# Maps our `model.arch` key to an upstream ppdet config. The upstream config
# under `ppdet/configs/<...>` carries architecture, head, post-process and the
# default training recipe. We patch the dataset/epochs/lr on top.
_PPDET_BASE_CONFIGS: dict[str, str] = {
    "paddle-picodet-s":   "configs/picodet/picodet_s_416_coco_lcnet.yml",
    "paddle-picodet-m":   "configs/picodet/picodet_m_416_coco_lcnet.yml",
    "paddle-picodet-l":   "configs/picodet/picodet_l_416_coco_lcnet.yml",
    "paddle-ppyoloe-s":   "configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml",
    "paddle-ppyoloe-m":   "configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml",
    "paddle-ppyoloe-l":   "configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml",
    "paddle-ppyoloe-x":   "configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml",
    "paddle-ppyoloe-plus-s": "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
    "paddle-ppyoloe-plus-m": "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml",
    "paddle-ppyoloe-plus-l": "configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml",
    "paddle-ppyoloe-plus-x": "configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml",
}


def detection_overrides(our: dict[str, Any]) -> dict[str, Any]:
    """Build a flat dict of patches to apply on top of the upstream ppdet base config.

    Returns the keys ppdet's `merge_config` expects. The caller (driver in
    train.py) does the actual `load_config(base) + merge_config(patches)`.
    """
    arch = our["model"]["arch"]
    if arch not in _PPDET_BASE_CONFIGS:
        raise ValueError(
            f"Unknown paddle detection arch: {arch!r}. "
            f"Available: {sorted(_PPDET_BASE_CONFIGS)}"
        )

    data = our["_data_resolved"]
    data_dir = (our["_data_path"].parent / data["path"]).resolve()
    num_classes = int(data.get("num_classes") or len(data.get("names") or []))
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0 in 05_data.yaml")

    train_cfg = our.get("training", {})
    data_cfg = our.get("data", {})

    # ppdet expects flat top-level keys; nesting into `TrainDataset` etc. is
    # done by the upstream loader. We hand back what we want to *override*.
    patches: dict[str, Any] = {
        "num_classes": num_classes,
        "epoch": int(train_cfg.get("epochs", 1)),
        "snapshot_epoch": int(train_cfg.get("epochs", 1)),  # save once at end of training
        "log_iter": 10,
        "save_dir": str(_save_dir_from_config(our)),
        # Upstream uses fixed batch sizes per dataset section. Patch both.
        "TrainReader": {"batch_size": int(data_cfg.get("batch_size", 4))},
        "EvalReader": {"batch_size": int(data_cfg.get("batch_size", 4))},
        "TestReader": {"batch_size": 1},
        # Dataset paths — point at our YOLO-format dataset converted to COCO JSON
        # (see _ensure_coco_annotations in train.py).
        "TrainDataset": _coco_dataset_block(data_dir, data, "train"),
        "EvalDataset":  _coco_dataset_block(data_dir, data, "val"),
        "TestDataset":  _coco_dataset_block(data_dir, data, "val"),
    }

    if "lr" in train_cfg:
        patches["LearningRate"] = {"base_lr": float(train_cfg["lr"])}

    return patches


def _coco_dataset_block(data_dir: Path, data_yaml: dict[str, Any], split: str) -> dict[str, Any]:
    """ppdet expects a COCODataSet block. file_name in our COCO JSON is already
    prefixed with `<split>/images/...`, so image_dir is the dataset root.
    """
    return {
        "name": "COCODataSet",
        "image_dir": str(data_dir),
        "anno_path": str((data_dir / f"{split}_paddle_coco.json").resolve()),
        "dataset_dir": str(data_dir),
        "data_fields": ["image", "gt_bbox", "gt_class", "is_crowd"],
    }


def _save_dir_from_config(our: dict[str, Any]) -> Path:
    """Resolve `logging.save_dir` to an absolute path."""
    save_dir = our.get("logging", {}).get("save_dir")
    if save_dir is None:
        raise ValueError("logging.save_dir is required for paddle backend")
    p = Path(save_dir)
    if not p.is_absolute():
        p = (our["_config_path"].parent / p).resolve()
    return p


def ppdet_base_config_path(arch: str) -> str:
    """Return the relative path inside the ppdet repo for a given arch's base config."""
    if arch not in _PPDET_BASE_CONFIGS:
        raise ValueError(f"Unknown paddle detection arch: {arch!r}")
    return _PPDET_BASE_CONFIGS[arch]
