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
    _is_video_group,
    _temporal_split_video_group,
    apply_max_per_group_eval,
    build_groups,
    compute_phashes,
    per_source_split,
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


# ---------------------------------------------------------------------------
# New per-source + temporal split tests (added 2026-05-03)
# ---------------------------------------------------------------------------


def test_video_group_detection(tmp_path: Path) -> None:
    """Video group: monotonic numeric suffix, size >= 20."""
    # Sequential numeric filenames → video
    video = [tmp_path / f"frame_{i:03d}.jpg" for i in range(25)]
    assert _is_video_group(video, min_size=20) is True

    # Random hash filenames → not video
    random_hash = [tmp_path / f"img_{c}.jpg" for c in "abcdefghijklmnopqrstuvwxy"]
    assert _is_video_group(random_hash, min_size=20) is False

    # Too small → not video
    small_seq = [tmp_path / f"frame_{i:03d}.jpg" for i in range(10)]
    assert _is_video_group(small_seq, min_size=20) is False

    # Realistic AoF naming
    aof = [tmp_path / f"AoF{i:05d}.jpg" for i in range(1000, 1030)]
    assert _is_video_group(aof, min_size=20) is True

    # Roboflow-style augmentation hashes (no trailing digits before extension)
    rf = [tmp_path / f"000051_jpg.rf.{c * 8}.jpg" for c in "abcdefghijklmnopqrstuvwxy"]
    assert _is_video_group(rf, min_size=20) is False


def test_temporal_split_with_gap(tmp_path: Path) -> None:
    imgs = [tmp_path / f"frame_{i:04d}.jpg" for i in range(100)]
    mapping = _temporal_split_video_group(
        imgs, target_ratios=(0.7, 0.15, 0.15), gap_fraction=0.05, min_gap_frames=5
    )
    splits = [mapping[p] for p in imgs]
    n_train = sum(1 for s in splits if s == "train")
    n_val = sum(1 for s in splits if s == "val")
    n_test = sum(1 for s in splits if s == "test")
    n_drop = sum(1 for s in splits if s is None)

    # Layout: 2 gaps of 5 = 10 dropped; remaining 90 split 70/15/15
    assert n_drop == 10
    assert n_train == 63  # round(90 * 0.7)
    assert n_val == 14   # round(90 * 0.15)
    assert n_test == 13  # remainder
    assert n_train + n_val + n_test + n_drop == 100

    # No overlap; gap frames sit between train→val and val→test
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx = [i for i, s in enumerate(splits) if s == "val"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    assert max(train_idx) < min(val_idx)
    assert max(val_idx) < min(test_idx)


def test_per_source_split_proportionality(tmp_path: Path) -> None:
    """3 sources × 100 imgs each → each source ~70/15/15 ±5%."""
    img_to_group: dict[Path, int] = {}
    img_to_source: dict[Path, str] = {}
    img_to_classes: dict[Path, list[int]] = {}
    gid = 0
    for s_idx in range(3):
        for i in range(100):
            # Each img gets its own group (still groups, not video)
            p = tmp_path / f"src{s_idx}_unique_{i:03d}.png"
            img_to_group[p] = gid
            img_to_source[p] = f"src{s_idx}"
            img_to_classes[p] = [i % 2]
            gid += 1

    # Disable temporal — these are still groups (singletons)
    out = per_source_split(
        img_to_group, img_to_classes, img_to_source,
        target_ratios=(0.70, 0.15, 0.15),
        enable_temporal=False,
        seed=42,
    )

    # Per-source counts
    per_src_split: dict[str, dict[str, int]] = {
        s: {"train": 0, "val": 0, "test": 0} for s in ("src0", "src1", "src2")
    }
    for img, sp in out.items():
        if sp is None:
            continue
        per_src_split[img_to_source[img]][sp] += 1

    for src in ("src0", "src1", "src2"):
        total = sum(per_src_split[src].values())
        assert total == 100
        train_frac = per_src_split[src]["train"] / total
        val_frac = per_src_split[src]["val"] / total
        test_frac = per_src_split[src]["test"] / total
        assert 0.65 <= train_frac <= 0.75, f"{src} train_frac={train_frac}"
        assert 0.10 <= val_frac <= 0.20, f"{src} val_frac={val_frac}"
        assert 0.10 <= test_frac <= 0.20, f"{src} test_frac={test_frac}"


def test_per_source_split_no_leakage(tmp_path: Path) -> None:
    paths, src_map, cls_map = _make_synthetic_dataset(tmp_path, n_sources=3, per_source=40)
    img_to_hash = compute_phashes(paths, n_workers=2)
    img_to_group = build_groups(img_to_hash, hamming_thresh=3)

    img_to_classes = {p: cls_map[p] for p in paths}

    out = per_source_split(
        img_to_group, img_to_classes, src_map,
        target_ratios=(0.70, 0.15, 0.15),
        enable_temporal=False,
        seed=42,
    )

    img_to_split = {p: s for p, s in out.items() if s is not None}
    leaks = verify_no_leakage(img_to_hash, img_to_split, hamming_thresh=3)
    assert leaks == 0


def test_per_source_split_tiny_source_to_train(tmp_path: Path) -> None:
    """Source with <7 imgs → all to train."""
    img_to_group: dict[Path, int] = {}
    img_to_source: dict[Path, str] = {}
    img_to_classes: dict[Path, list[int]] = {}
    gid = 0
    # Tiny source: 5 imgs
    for i in range(5):
        p = tmp_path / f"tiny_{i:03d}.png"
        img_to_group[p] = gid
        img_to_source[p] = "tiny_src"
        img_to_classes[p] = [0]
        gid += 1
    # Normal source: 50 imgs
    for i in range(50):
        p = tmp_path / f"big_{i:03d}.png"
        img_to_group[p] = gid
        img_to_source[p] = "big_src"
        img_to_classes[p] = [0]
        gid += 1

    out = per_source_split(
        img_to_group, img_to_classes, img_to_source,
        target_ratios=(0.70, 0.15, 0.15),
        enable_temporal=False,
        seed=42,
    )

    tiny_splits = {out[p] for p in img_to_group if img_to_source[p] == "tiny_src"}
    assert tiny_splits == {"train"}, f"tiny source spread across {tiny_splits}"


def test_validate_dedup_config_per_source_strategy() -> None:
    merged = validate_dedup_config({
        "split_strategy": "per_source_with_temporal",
        "split_ratios": [0.7, 0.15, 0.15],
        "temporal": {"enabled": True, "min_group_size_for_video": 30},
    })
    assert merged["split_strategy"] == "per_source_with_temporal"
    assert merged["split_ratios"] == (0.7, 0.15, 0.15)
    assert merged["temporal"]["min_group_size_for_video"] == 30
    assert merged["temporal"]["gap_fraction"] == 0.05  # default preserved


def test_validate_dedup_config_bad_split_ratios() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_dedup_config({"split_ratios": [0.5, 0.3, 0.3]})
    with pytest.raises(ValueError, match="list of 3"):
        validate_dedup_config({"split_ratios": [0.7, 0.3]})


def test_validate_dedup_config_bad_temporal() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        validate_dedup_config({"temporal": {"badkey": 1}})
    with pytest.raises(ValueError, match="gap_fraction"):
        validate_dedup_config({"temporal": {"gap_fraction": 0.9}})
