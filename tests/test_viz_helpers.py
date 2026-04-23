"""Smoke tests for utils.viz helpers.

Synthetic-data only (no real dataset, no GPU). Fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.viz import (
    COCO_SKELETON_EDGES,
    VizStyle,
    annotate_detections,
    annotate_keypoints,
    annotate_polygons,
    apply_plot_style,
    classification_banner,
    save_image_grid,
)


def _blank(h: int = 256, w: int = 256) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_annotate_detections_smoke():
    img = _blank()
    dets = sv.Detections(
        xyxy=np.array([[20, 20, 80, 80], [120, 40, 200, 150]], dtype=np.float32),
        confidence=np.array([0.9, 0.75], dtype=np.float32),
        class_id=np.array([0, 1], dtype=int),
    )
    out = annotate_detections(img, dets, class_names={0: "cat", 1: "dog"})
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # Pixels inside a box should have changed from the uniform background.
    assert not np.array_equal(out[20:80, 20:80], img[20:80, 20:80])


def test_annotate_keypoints_smoke():
    img = _blank()
    # Single person, 17 COCO keypoints spread across image.
    kpts = np.stack(
        [
            np.linspace(40, 200, 17),
            np.linspace(40, 200, 17),
        ],
        axis=-1,
    ).astype(np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    conf[3] = 0.1  # below threshold — should be hidden
    out = annotate_keypoints(
        img,
        kpts,
        skeleton_edges=COCO_SKELETON_EDGES,
        confidence=conf,
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_annotate_polygons_smoke():
    img = _blank()
    tri = np.array([[60, 60], [180, 70], [120, 200]], dtype=np.int32)
    out = annotate_polygons(img, [tri], labels=["zone_a"])
    assert out.shape == img.shape
    # Alpha blend inside the triangle must change mean pixel value.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [tri], 1)
    assert out[mask.astype(bool)].mean() != img[mask.astype(bool)].mean()


def test_classification_banner_smoke():
    img = _blank(128, 256)
    style = VizStyle()
    out = classification_banner(img, "fire 0.92", style=style, position="top")
    assert out.shape[0] == img.shape[0] + style.banner_height
    assert out.shape[1] == img.shape[1]


def test_save_image_grid_smoke(tmp_path):
    imgs = [_blank(100 + i * 10, 150 + i * 5) for i in range(5)]
    out_path = tmp_path / "grid.png"
    save_image_grid(imgs, out_path, cols=3, titles=[f"i{i}" for i in range(5)])
    assert out_path.exists()
    loaded = cv2.imread(str(out_path))
    assert loaded is not None
    assert loaded.shape[0] > 0 and loaded.shape[1] > 0


def test_vizstyle_auto_thickness():
    style = VizStyle()
    t_small = style.auto_box_thickness(400, 400)
    t_large = style.auto_box_thickness(2000, 2000)
    assert t_large > t_small
    assert style.auto_keypoint_radius(2000, 2000) > style.auto_keypoint_radius(400, 400)
    assert style.auto_skeleton_thickness(2000, 2000) >= style.auto_skeleton_thickness(400, 400)


def test_vizstyle_from_config_new_keys():
    cfg = {
        "visualization": {
            "palette": "pastel",
            "box_thickness": 7,
            "label_text_scale": 0.9,
            "label_position": "bottom_left",
            "keypoint_radius": 11,
            "kpt_visibility_threshold": 0.5,
            "zone_fill_alpha": 0.33,
            "grid_cell_size": 640,
            "grid_cols": 5,
            "banner_height": 40,
            "banner_bg_rgb": [10, 20, 30],
            "error_colors_rgb": {"tp": [1, 2, 3], "fp": [4, 5, 6]},
        }
    }
    s = VizStyle.from_config(cfg)
    assert s.palette == "pastel"
    assert s.box_thickness == 7
    assert s.label_text_scale == 0.9
    assert s.label_position == "bottom_left"
    assert s.sv_label_position() == sv.Position.BOTTOM_LEFT
    assert s.keypoint_radius == 11
    assert s.kpt_visibility_threshold == 0.5
    assert s.zone_fill_alpha == 0.33
    assert s.grid_cell_size == 640
    assert s.grid_cols == 5
    assert s.banner_height == 40
    assert s.banner_bg_rgb == (10, 20, 30)
    assert s.error_colors_rgb["tp"] == (1, 2, 3)
    # Alias 'viz:' also works
    s2 = VizStyle.from_config({"viz": {"box_thickness": 9}})
    assert s2.box_thickness == 9


def test_apply_plot_style_idempotent():
    import matplotlib as mpl

    apply_plot_style()
    dpi1 = mpl.rcParams["figure.dpi"]
    grid1 = mpl.rcParams["axes.grid"]
    apply_plot_style()
    assert mpl.rcParams["figure.dpi"] == dpi1
    assert mpl.rcParams["axes.grid"] == grid1
    assert mpl.rcParams["figure.dpi"] == 110


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
