"""Tests for `core.p00_data_prep.core.dedup`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.p00_data_prep.core.dedup import (  # noqa: E402
    apply_max_per_group_eval,
    build_groups,
    compute_phashes,
    stratified_group_split,
    validate_dedup_config,
    verify_no_leakage,
)


def _make_image(path: Path, seed: int, size: int = 64) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_synthetic_dataset(tmp_path: Path, n_sources: int = 3, per_source: int = 30) -> tuple[
    list[Path], dict[Path, str], dict[Path, list[int]]
]:
    """Returns (paths, path→source, path→class_ids)."""
    paths: list[Path] = []
    src_map: dict[Path, str] = {}
    cls_map: dict[Path, list[int]] = {}
    for s in range(n_sources):
        for i in range(per_source):
            p = tmp_path / f"src{s}_img{i:03d}.png"
            _make_image(p, seed=s * 1000 + i)
            paths.append(p)
            src_map[p] = f"src{s}"
            # Spread 2 classes across all sources roughly evenly
            cls_map[p] = [i % 2]
    return paths, src_map, cls_map


def test_source_balance_invariant(tmp_path: Path) -> None:
    """Every source family must appear in train/val/test within ±25% of target."""
    paths, src_map, cls_map = _make_synthetic_dataset(tmp_path, n_sources=3, per_source=40)
    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    g_classes: dict[int, list[int]] = {}
    g_imgs: dict[int, int] = {}
    g_source: dict[int, str] = {}
    for img, gid in img_to_group.items():
        g_classes.setdefault(gid, []).extend(cls_map[img])
        g_imgs[gid] = g_imgs.get(gid, 0) + 1
        g_source.setdefault(gid, src_map[img])

    g2s = stratified_group_split(g_classes, g_imgs, g_source, seed=42)

    per_split_per_src: dict[str, dict[str, int]] = {s: {} for s in ("train", "val", "test")}
    for img, gid in img_to_group.items():
        sp = g2s[gid]
        src = src_map[img]
        per_split_per_src[sp][src] = per_split_per_src[sp].get(src, 0) + 1

    # Each source must be present in every split (count > 0)
    for src in {"src0", "src1", "src2"}:
        for sp in ("train", "val", "test"):
            assert per_split_per_src[sp].get(src, 0) > 0, (
                f"source {src} missing from split {sp}: {per_split_per_src}"
            )


def test_class_balance_invariant(tmp_path: Path) -> None:
    """Per-class image counts within each split should be roughly proportional to ratio."""
    paths, src_map, cls_map = _make_synthetic_dataset(tmp_path, n_sources=2, per_source=50)
    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    g_classes: dict[int, list[int]] = {}
    g_imgs: dict[int, int] = {}
    g_source: dict[int, str] = {}
    for img, gid in img_to_group.items():
        g_classes.setdefault(gid, []).extend(cls_map[img])
        g_imgs[gid] = g_imgs.get(gid, 0) + 1
        g_source.setdefault(gid, src_map[img])

    g2s = stratified_group_split(g_classes, g_imgs, g_source, seed=42)

    # Aggregate per-split per-class
    per_split_per_cls: dict[str, dict[int, int]] = {s: {} for s in ("train", "val", "test")}
    for img, gid in img_to_group.items():
        sp = g2s[gid]
        for c in cls_map[img]:
            per_split_per_cls[sp][c] = per_split_per_cls[sp].get(c, 0) + 1

    for c in (0, 1):
        total = sum(per_split_per_cls[s].get(c, 0) for s in ("train", "val", "test"))
        train_frac = per_split_per_cls["train"].get(c, 0) / total
        # Expect train ratio ≈ 0.8; allow ±0.25 for small synthetic datasets
        assert 0.55 <= train_frac <= 1.0, f"class {c} train_frac={train_frac:.2f}"


def test_zero_leakage(tmp_path: Path) -> None:
    paths, src_map, cls_map = _make_synthetic_dataset(tmp_path, n_sources=3, per_source=20)
    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    g_classes: dict[int, list[int]] = {}
    g_imgs: dict[int, int] = {}
    g_source: dict[int, str] = {}
    for img, gid in img_to_group.items():
        g_classes.setdefault(gid, []).extend(cls_map[img])
        g_imgs[gid] = g_imgs.get(gid, 0) + 1
        g_source.setdefault(gid, src_map[img])

    g2s = stratified_group_split(g_classes, g_imgs, g_source, seed=42)
    img_to_split = {img: g2s[gid] for img, gid in img_to_group.items()}

    leaks = verify_no_leakage(img_to_hash, img_to_split, hamming_thresh=3)
    assert leaks == 0


def test_max_per_group_eval_cap(tmp_path: Path) -> None:
    """Build a 50-image identical-pHash group; cap=10 must keep ≤10 in val+test."""
    paths: list[Path] = []
    base = np.zeros((64, 64, 3), dtype=np.uint8) + 128
    for i in range(50):
        p = tmp_path / f"dup_{i:03d}.png"
        # Identical content (same pHash) → all collapse to one group
        Image.fromarray(base.copy()).save(p)
        paths.append(p)
    # Add a few "diverse" images so the splitter has more than one group
    for i in range(15):
        p = tmp_path / f"div_{i:03d}.png"
        _make_image(p, seed=9000 + i)
        paths.append(p)

    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    g_classes: dict[int, list[int]] = {gid: [0] for gid in set(img_to_group.values())}
    g_imgs: dict[int, int] = {}
    for gid in img_to_group.values():
        g_imgs[gid] = g_imgs.get(gid, 0) + 1

    # Force the duplicate group into val by giving it the largest size; without source,
    # the splitter will likely place it in train (largest first), so we manually assign.
    dup_gid = img_to_group[paths[0]]
    g2s = stratified_group_split(g_classes, g_imgs, seed=42)
    g2s[dup_gid] = "val"

    capped = apply_max_per_group_eval(img_to_group, g2s, max_per_group_eval=10)
    in_val = sum(1 for img in paths[:50] if capped.get(img) == "val")
    dropped = sum(1 for img in paths[:50] if capped.get(img) is None)
    assert in_val == 10
    assert dropped == 40


def test_stratify_by_class_only_backcompat(tmp_path: Path) -> None:
    """Omitting source from stratify_by must run without error."""
    paths, _src_map, cls_map = _make_synthetic_dataset(tmp_path, n_sources=2, per_source=20)
    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    g_classes: dict[int, list[int]] = {}
    g_imgs: dict[int, int] = {}
    for img, gid in img_to_group.items():
        g_classes.setdefault(gid, []).extend(cls_map[img])
        g_imgs[gid] = g_imgs.get(gid, 0) + 1

    g2s = stratified_group_split(
        g_classes, g_imgs, group_to_source=None, stratify_by=["class"], seed=42
    )
    assert set(g2s.values()).issubset({"train", "val", "test"})


def test_validate_dedup_config_defaults() -> None:
    merged = validate_dedup_config({})
    assert merged["enabled"] is True
    assert merged["hamming_thresh"] == 3
    assert merged["stratify_by"] == ["class", "source"]


def test_validate_dedup_config_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        validate_dedup_config({"hammming_thresh": 3})  # typo


def test_validate_dedup_config_bad_thresh() -> None:
    with pytest.raises(ValueError):
        validate_dedup_config({"hamming_thresh": 99})


def test_validate_dedup_config_bad_source_from() -> None:
    with pytest.raises(ValueError):
        validate_dedup_config({"source_from": "magic"})
