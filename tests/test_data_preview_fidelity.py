"""Synthetic-fixture fidelity tests for data_preview artifacts.

Verifies that ``core.p05_data.run_viz.generate_dataset_stats`` and
``write_dataset_info`` accurately reflect on-disk reality across all four
tasks (detection, classification, segmentation, keypoint).

Goal: lock down current correct behavior so future panel rewrites can be
validated. Uses pytest's ``tmp_path`` and tiny 32x32 / 64x64 images.

Each task gets a hand-crafted dataset with N_TRAIN=5, N_VAL=2 images and
labels whose exact ground-truth stats are computed inline so the assertions
are self-documenting.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.p05_data.run_viz import generate_dataset_stats, write_dataset_info

N_TRAIN = 5
N_VAL = 2
IMG_W = 64
IMG_H = 64


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #

def _write_image(path: Path, w: int = IMG_W, h: int = IMG_H) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_detection_dataset(root: Path) -> dict:
    """Two classes (fire=0, smoke=1).

    Train: 5 images. img0 has 2 fire boxes, img1 has 1 smoke, img2 has
    1 fire + 1 smoke, img3 empty, img4 has 1 fire. Total fire=4, smoke=2,
    annotations=6, n_empty=1.
    Val: 2 images. img0 has 1 fire, img1 has 1 smoke. Total ann=2.
    """
    layout = [
        ("train", [
            [(0, 0.5, 0.5, 0.4, 0.4), (0, 0.2, 0.2, 0.1, 0.1)],
            [(1, 0.5, 0.5, 0.5, 0.5)],
            [(0, 0.3, 0.3, 0.2, 0.2), (1, 0.7, 0.7, 0.2, 0.2)],
            [],
            [(0, 0.5, 0.5, 0.6, 0.6)],
        ]),
        ("val", [
            [(0, 0.5, 0.5, 0.3, 0.3)],
            [(1, 0.4, 0.4, 0.2, 0.2)],
        ]),
    ]
    for split, imgs in layout:
        for i, boxes in enumerate(imgs):
            stem = f"{split}_{i:02d}"
            _write_image(root / split / "images" / f"{stem}.jpg")
            label_path = root / split / "labels" / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w") as f:
                for cid, cx, cy, w, h in boxes:
                    f.write(f"{cid} {cx} {cy} {w} {h}\n")

    data_cfg = {
        "task": "detection",
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "fire", 1: "smoke"},
        "num_classes": 2,
        "input_size": [IMG_H, IMG_W],
    }
    return data_cfg


def _make_classification_dataset(root: Path) -> dict:
    """Folder layout: classes cat=0, dog=1.

    Train: 3 cat + 2 dog (total 5). Val: 1 cat + 1 dog.
    """
    plan = {
        "train": {"cat": 3, "dog": 2},
        "val": {"cat": 1, "dog": 1},
    }
    for split, by_class in plan.items():
        for cname, n in by_class.items():
            for i in range(n):
                _write_image(root / split / cname / f"{cname}_{i:02d}.jpg")

    data_cfg = {
        "task": "classification",
        "path": str(root),
        "train": "train",
        "val": "val",
        "names": {0: "cat", 1: "dog"},
        "num_classes": 2,
        "input_size": [IMG_H, IMG_W],
    }
    return data_cfg


def _make_segmentation_dataset(root: Path) -> dict:
    """Two-class semantic seg (bg=0, fg=1).

    Each train image: a known fg square. Val: smaller fg square.
    """
    train_fg_pixels = [400, 100, 0, 256, 64]   # px counts for class 1 in 5 images
    val_fg_pixels = [16, 0]
    for split, fg_list in [("train", train_fg_pixels), ("val", val_fg_pixels)]:
        for i, n_fg in enumerate(fg_list):
            stem = f"{split}_{i:02d}"
            _write_image(root / split / "images" / f"{stem}.jpg")
            mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            if n_fg > 0:
                # Fill a square of side approx sqrt(n_fg) starting top-left
                side = int(np.sqrt(n_fg))
                mask[:side, :side] = 1
            mask_path = root / split / "masks" / f"{stem}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

    data_cfg = {
        "task": "segmentation",
        "path": str(root),
        "train": "train",
        "val": "val",
        "names": {0: "background", 1: "object"},
        "num_classes": 2,
        "input_size": [IMG_H, IMG_W],
    }
    return data_cfg


def _make_keypoint_dataset(root: Path) -> dict:
    """1 class (person), 3 keypoints. Visibility encodes 0/1/2.

    Train: 5 images each with 1 instance. We hand-craft visibility patterns:
      img0: v=[2,2,2]  (3 visible)
      img1: v=[2,1,0]
      img2: v=[2,2,0]
      img3: v=[0,0,0]  (would be 0 visible joints — file present but no
                        labels: write empty file so it counts as empty)
      img4: v=[2,2,1]
    Val: 2 images:
      img0: v=[2,2,2]
      img1: v=[2,0,0]
    """
    K = 3

    def line(cls: int, box, vis):
        cx, cy, w, h = box
        kp_parts = []
        # 3 keypoints at fixed positions inside the box
        for j, v in enumerate(vis):
            kx = cx + (j - 1) * 0.05
            ky = cy + (j - 1) * 0.05
            kp_parts.extend([f"{kx}", f"{ky}", f"{v}"])
        return f"{cls} {cx} {cy} {w} {h} " + " ".join(kp_parts)

    train_vis = [
        [2, 2, 2],
        [2, 1, 0],
        [2, 2, 0],
        None,            # empty file
        [2, 2, 1],
    ]
    val_vis = [
        [2, 2, 2],
        [2, 0, 0],
    ]
    for split, vis_list in [("train", train_vis), ("val", val_vis)]:
        for i, vis in enumerate(vis_list):
            stem = f"{split}_{i:02d}"
            _write_image(root / split / "images" / f"{stem}.jpg")
            label_path = root / split / "labels" / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "w") as f:
                if vis is not None:
                    f.write(line(0, (0.5, 0.5, 0.4, 0.4), vis) + "\n")

    data_cfg = {
        "task": "keypoint",
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
        "num_classes": 1,
        "num_keypoints": K,
        "input_size": [IMG_H, IMG_W],
    }
    return data_cfg


def _load_stats_json(out_dir: Path) -> dict:
    p = out_dir / "01_dataset_stats.json"
    assert p.exists(), f"stats JSON not written at {p}"
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_detection_stats_fidelity(tmp_path: Path) -> None:
    ds_root = tmp_path / "det_ds"
    data_cfg = _make_detection_dataset(ds_root)
    out_dir = tmp_path / "out_det"

    generate_dataset_stats(
        data_cfg=data_cfg,
        base_dir=str(tmp_path),
        class_names=data_cfg["names"],
        splits=["train", "val"],
        out_dir=out_dir,
    )
    stats = _load_stats_json(out_dir)

    assert stats["task"] == "detection"
    assert stats["splits"]["train"]["n_images"] == N_TRAIN
    assert stats["splits"]["val"]["n_images"] == N_VAL

    # Train: fire=4, smoke=2, total=6, n_empty=1
    train = stats["splits"]["train"]
    assert train["n_annotations"] == 6
    assert train["n_empty"] == 1
    assert train["class_counts"] == {"fire": 4, "smoke": 2}

    # Val: fire=1, smoke=1, total=2, n_empty=0
    val = stats["splits"]["val"]
    assert val["n_annotations"] == 2
    assert val["n_empty"] == 0
    assert val["class_counts"] == {"fire": 1, "smoke": 1}


def test_classification_stats_fidelity(tmp_path: Path) -> None:
    ds_root = tmp_path / "cls_ds"
    data_cfg = _make_classification_dataset(ds_root)
    out_dir = tmp_path / "out_cls"

    generate_dataset_stats(
        data_cfg=data_cfg,
        base_dir=str(tmp_path),
        class_names=data_cfg["names"],
        splits=["train", "val"],
        out_dir=out_dir,
    )
    stats = _load_stats_json(out_dir)

    assert stats["task"] == "classification"
    assert stats["splits"]["train"]["n_images"] == N_TRAIN
    assert stats["splits"]["val"]["n_images"] == N_VAL

    assert stats["splits"]["train"]["class_counts"] == {"cat": 3, "dog": 2}
    assert stats["splits"]["val"]["class_counts"] == {"cat": 1, "dog": 1}

    # Imbalance ratio: train max/min = 3/2 = 1.5
    assert stats["splits"]["train"]["imbalance_ratio_max_over_min"] == pytest.approx(1.5)


def test_segmentation_stats_fidelity(tmp_path: Path) -> None:
    ds_root = tmp_path / "seg_ds"
    data_cfg = _make_segmentation_dataset(ds_root)
    out_dir = tmp_path / "out_seg"

    generate_dataset_stats(
        data_cfg=data_cfg,
        base_dir=str(tmp_path),
        class_names=data_cfg["names"],
        splits=["train", "val"],
        out_dir=out_dir,
    )
    stats = _load_stats_json(out_dir)

    assert stats["task"] == "segmentation"
    assert stats["splits"]["train"]["n_images"] == N_TRAIN
    assert stats["splits"]["val"]["n_images"] == N_VAL

    # Train: object pixels were drawn as squares of side floor(sqrt(n));
    # actual fg pixels: floor(sqrt(400))^2 = 400, sqrt(100)^2=100, 0,
    # sqrt(256)^2=256, sqrt(64)^2=64 -> total fg = 820
    expected_fg = 20 * 20 + 10 * 10 + 0 + 16 * 16 + 8 * 8  # 820
    train_pc = stats["splits"]["train"]["pixel_counts_by_class"]
    assert train_pc["object"] == expected_fg
    total_px = N_TRAIN * IMG_W * IMG_H
    assert train_pc["background"] == total_px - expected_fg

    # Val: sqrt(16)^2=16, 0 -> total fg = 16
    val_pc = stats["splits"]["val"]["pixel_counts_by_class"]
    assert val_pc["object"] == 16


def test_keypoint_stats_fidelity(tmp_path: Path) -> None:
    ds_root = tmp_path / "kpt_ds"
    data_cfg = _make_keypoint_dataset(ds_root)
    out_dir = tmp_path / "out_kpt"

    generate_dataset_stats(
        data_cfg=data_cfg,
        base_dir=str(tmp_path),
        class_names=data_cfg["names"],
        splits=["train", "val"],
        out_dir=out_dir,
    )
    stats = _load_stats_json(out_dir)

    assert stats["task"] == "keypoint"
    assert stats["num_keypoints"] == 3
    assert stats["splits"]["train"]["n_images"] == N_TRAIN
    assert stats["splits"]["val"]["n_images"] == N_VAL

    # Train: 4 instances (img3 is empty)
    assert stats["splits"]["train"]["n_instances"] == 4
    # Visibility per joint (v > 0 across instances): from
    #  [2,2,2], [2,1,0], [2,2,0], [2,2,1]
    # joint 0: 4 visible, joint 1: 4, joint 2: 2
    assert stats["splits"]["train"]["vis_counts"] == [4, 4, 2]
    # labeled_counts == n_instances per joint (4 each)
    assert stats["splits"]["train"]["labeled_counts"] == [4, 4, 4]

    # Val: 2 instances [2,2,2], [2,0,0]
    assert stats["splits"]["val"]["n_instances"] == 2
    assert stats["splits"]["val"]["vis_counts"] == [2, 1, 1]
    assert stats["splits"]["val"]["labeled_counts"] == [2, 2, 2]

    assert stats["total_instances"] == 6


def test_write_dataset_info_no_absolute_paths(tmp_path: Path) -> None:
    ds_root = tmp_path / "det_ds"
    data_cfg = _make_detection_dataset(ds_root)
    out_dir = tmp_path / "out_info"

    data_config_path = tmp_path / "configs" / "05_data.yaml"
    training_config_path = tmp_path / "configs" / "06_training.yaml"
    data_config_path.parent.mkdir(parents=True, exist_ok=True)
    data_config_path.write_text("# stub")
    training_config_path.write_text("# stub")

    write_dataset_info(
        out_dir=out_dir,
        feature_name="test_feature",
        data_config_path=data_config_path,
        training_config_path=training_config_path,
        data_cfg=data_cfg,
        training_cfg={"training": {"backend": "pytorch"}},
        class_names=data_cfg["names"],
        split_sizes={"train": N_TRAIN, "val": N_VAL},
    )

    info_path = out_dir / "00_dataset_info.json"
    assert info_path.exists()
    info = json.loads(info_path.read_text())

    # 1) data_config / training_config are basenames only.
    assert info["data_config"] == "05_data.yaml"
    assert info["training_config"] == "06_training.yaml"

    # 2) dataset_relpath is relative (does not start with /).
    assert info["dataset_relpath"] is not None
    assert not info["dataset_relpath"].startswith("/")

    # 3) No absolute path strings should leak into the JSON. The most
    #    common failure mode is the raw resolved dataset root appearing
    #    verbatim somewhere in the file.
    raw_text = info_path.read_text()
    abs_root = str(ds_root.resolve())
    assert abs_root not in raw_text, (
        f"Absolute dataset root leaked into 00_dataset_info.json: {abs_root}"
    )
    assert str(tmp_path.resolve()) not in raw_text or info["dataset_relpath"] is None

    # 4) Splits block reflects the input sizes.
    assert info["splits"]["train"]["n_images"] == N_TRAIN
    assert info["splits"]["val"]["n_images"] == N_VAL
    # No subset in this call -> n_images_full / n_images_used absent.
    assert "n_images_full" not in info["splits"]["train"]
    assert "n_images_used" not in info["splits"]["train"]


def test_write_dataset_info_subset_emits_full_and_used(tmp_path: Path) -> None:
    """When ``full_sizes`` is passed and differs from split_sizes, the JSON
    must surface both ``n_images_full`` and ``n_images_used``.
    """
    ds_root = tmp_path / "det_ds"
    data_cfg = _make_detection_dataset(ds_root)
    out_dir = tmp_path / "out_info_subset"

    write_dataset_info(
        out_dir=out_dir,
        feature_name="test_feature",
        data_config_path=tmp_path / "05_data.yaml",
        training_config_path=tmp_path / "06_training.yaml",
        data_cfg=data_cfg,
        training_cfg={"training": {"backend": "pytorch"}},
        class_names=data_cfg["names"],
        split_sizes={"train": 2, "val": 1},        # subset-active sizes
        full_sizes={"train": N_TRAIN, "val": N_VAL},
    )

    info = json.loads((out_dir / "00_dataset_info.json").read_text())
    train = info["splits"]["train"]
    assert train["n_images"] == 2
    assert train["n_images_full"] == N_TRAIN
    assert train["n_images_used"] == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
