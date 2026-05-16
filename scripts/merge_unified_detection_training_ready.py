"""Merge Phase 1 + Phase 2 raw / training_ready sources into a single unified
detection training_ready dataset.

Output: dataset_store/training_ready/unified_detection/{train,val,test}/{images,labels}/
        + valid_classes.json (per-source loss masking)
        + DATASET_REPORT.md (counts per class & per source)

Phase 1: read existing training_ready/<src>/{train,val,test}/  (already split)
Phase 2: read raw/<src>/<subdir>/{train,valid,test}/           (raw with native splits)

Files are COPIED (not symlinked). Per-image filenames are prefixed with the
source key (e.g. "fire__abc.jpg") to avoid collisions across sources.

Run from project root:
    uv run scripts/merge_unified_detection_training_ready.py
    uv run scripts/merge_unified_detection_training_ready.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_store"
OUT_ROOT     = DATASET_ROOT / "training_ready/unified_detection"

# ---------------------------------------------------------------------------
# Unified taxonomy — 19 active classes, sequential IDs (no reserved gaps).
# Phase 2 classes added later via HF num_labels resize + ignore_mismatched_sizes.
# ---------------------------------------------------------------------------
UNIFIED_NAMES: dict[int, str] = {
     0: "person",
     1: "fallen_person",
     2: "fire",
     3: "smoke",
     4: "phone_usage",
     5: "helmet",
     6: "no_helmet",
     7: "nitto_hat",
     8: "safety_shoes",
     9: "no_safety_shoes",
    10: "mask",
    11: "no_mask",
    12: "n95",
    13: "gloves",
    14: "apron",
    15: "harness",
    16: "no_harness",
    17: "harness_hooked",
    18: "harness_unhooked",
}

# ---------------------------------------------------------------------------
# Source registry. Each source declares:
#   src       — root dir
#   layout    — "training_ready" (has train/val/test/{images,labels})
#               | "yolo_raw"     (has train/valid/test/{images,labels})
#   remap     — {source_class_id: unified_class_id} (drop entries to skip class)
#   valid     — set of unified IDs that this source actually annotates
#               (for per-source loss masking; non-listed classes get loss=0)
# ---------------------------------------------------------------------------
SOURCES = {
    # ---- Phase 1 ---------------------------------------------------------
    "fire": {
        "src":    DATASET_ROOT / "training_ready/fire_detection",
        "layout": "training_ready",
        "remap":  {0: 2, 1: 3},                  # fire, smoke
        "valid":  {2, 3},
    },
    "fall": {
        "src":    DATASET_ROOT / "training_ready/fall_detection",
        "layout": "training_ready",
        "remap":  {0: 0, 1: 1},                  # person, fallen_person
        "valid":  {0, 1},
    },
    "phone": {
        "src":    DATASET_ROOT / "training_ready/safety_poketenashi_phone_usage",
        "layout": "training_ready",
        "remap":  {0: 4},                        # phone_usage
        "valid":  {4},
    },
    "helmet": {
        "src":    DATASET_ROOT / "training_ready/helmet_detection",
        "layout": "training_ready",
        "remap":  {0: 0, 1: 5, 2: 6, 3: 7},      # person, helmet, no_helmet, nitto_hat
        "valid":  {0, 5, 6, 7},
    },
    "shoes": {
        "src":    DATASET_ROOT / "training_ready/shoes_detection",
        "layout": "training_ready",
        "remap":  {0: 0, 1: 8, 2: 9},            # person, safety_shoes, no_safety_shoes
        "valid":  {0, 8, 9},
    },
    # ---- Phase 2 (raw) ---------------------------------------------------
    "mask3": {
        "src":    DATASET_ROOT / "raw/mask_detection/rf_mask_3class",
        "layout": "yolo_raw",
        "remap":  {0: 11, 1: 10, 2: 11},         # incorrect→no_mask, with_mask→mask, without→no_mask
        "valid":  {10, 11},
    },
    "mask2": {
        "src":    DATASET_ROOT / "raw/mask_detection/rf_mask_3k",
        "layout": "yolo_raw",
        "remap":  {0: 10, 1: 11},                # with_mask→mask, without_mask→no_mask
        "valid":  {10, 11},
    },
    "n95": {
        "src":    DATASET_ROOT / "raw/mask_detection/rf_n95",
        "layout": "yolo_raw",
        "remap":  {0: 12},                       # N95→n95; drop ear, earplug
        "valid":  {12},
    },
    "gloves": {
        "src":    DATASET_ROOT / "raw/glove_detection/rf_hand_gloves",
        "layout": "yolo_raw",
        "remap":  {0: 13},                       # HAND-GLOVES→gloves
        "valid":  {13},
    },
    "apron": {
        "src":    DATASET_ROOT / "raw/apron_detection/rf_apron",
        "layout": "yolo_raw",
        "remap":  {0: 14},                       # Wearing-Apron→apron
        "valid":  {14},
    },
    "harness1": {
        "src":    DATASET_ROOT / "raw/harness_detection/rf_body_harness",
        "layout": "yolo_raw",
        "remap":  {0: 15},                       # safety_harness→harness; drop "worker" (unreliable)
        "valid":  {15},
    },
    "harness2": {
        "src":    DATASET_ROOT / "raw/harness_detection/rf_safety_harness_v2",
        "layout": "yolo_raw",
        "remap":  {0: 17, 1: 16, 2: 18},         # anchored→hooked, no_safety_harness→no_harness, non_anchored→unhooked
        "valid":  {16, 17, 18},
    },
}

SPLIT_MAP = {
    "training_ready": {"train": "train", "val": "val",   "test": "test"},
    "yolo_raw":       {"train": "train", "val": "valid", "test": "test"},
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_split(src_root: Path, layout: str, split: str) -> list[Path]:
    """Return image paths for a given split in either layout."""
    sub = SPLIT_MAP[layout][split]
    img_dir = src_root / sub / "images"
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def _label_path_for(img_path: Path) -> Path:
    """Find sibling label file (replace 'images' dir with 'labels' + .txt extension)."""
    return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"


def _remap_label_file(src_label: Path, dst_label: Path,
                      remap: dict[int, int]) -> tuple[int, int]:
    """Read source YOLO label, remap class IDs (drop unmapped lines), write to dst.
    Returns (n_lines_written, n_lines_dropped)."""
    if not src_label.exists():
        dst_label.write_text("")
        return 0, 0

    out_lines: list[str] = []
    dropped = 0
    for line in src_label.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(parts[0])
        except ValueError:
            continue
        if src_id not in remap:
            dropped += 1
            continue
        out_lines.append(f"{remap[src_id]} " + " ".join(parts[1:]))

    dst_label.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines), dropped


def _process_source(
    key: str, cfg: dict, dst_root: Path, dry_run: bool,
) -> dict:
    """Copy + remap one source. Returns per-split image counts + per-class counts."""
    src_root: Path = cfg["src"]
    layout = cfg["layout"]
    remap = cfg["remap"]

    if not src_root.exists():
        logger.warning("Source missing: {} → skipped", src_root)
        return {"missing": True}

    split_counts: dict[str, int] = {}
    class_counts: dict[int, int] = defaultdict(int)
    valid_classes_per_image: dict[str, list[int]] = {}

    for split in ("train", "val", "test"):
        img_paths = _iter_split(src_root, layout, split)
        if not img_paths:
            split_counts[split] = 0
            continue

        dst_img_dir = dst_root / split / "images"
        dst_lbl_dir = dst_root / split / "labels"
        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        for img in img_paths:
            new_name = f"{key}__{img.name}"
            dst_img = dst_img_dir / new_name
            dst_lbl = dst_lbl_dir / f"{Path(new_name).stem}.txt"
            src_lbl = _label_path_for(img)

            if not dry_run:
                shutil.copy2(img, dst_img)
                n_kept, _ = _remap_label_file(src_lbl, dst_lbl, remap)
                if n_kept > 0:
                    for line in dst_lbl.read_text().splitlines():
                        if line.strip():
                            class_counts[int(line.split()[0])] += 1
            else:
                # dry-run still counts class lines from source for reporting
                if src_lbl.exists():
                    for line in src_lbl.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sid = int(line.split()[0])
                        except ValueError:
                            continue
                        if sid in remap:
                            class_counts[remap[sid]] += 1

            valid_classes_per_image[new_name] = sorted(cfg["valid"])
            kept += 1
        split_counts[split] = kept
        logger.info("  [{}/{}] copied {} imgs", key, split, kept)

    return {
        "missing": False,
        "split_counts": split_counts,
        "class_counts": dict(class_counts),
        "valid_classes_per_image": valid_classes_per_image,
    }


def _write_data_yaml(dst_root: Path) -> None:
    lines = [
        "# Generated by scripts/merge_unified_detection_training_ready.py",
        "names:",
    ]
    for cid in sorted(UNIFIED_NAMES):
        lines.append(f"  {cid}: {UNIFIED_NAMES[cid]}")
    lines += [
        f"num_classes: {max(UNIFIED_NAMES) + 1}",
        "input_size: [640, 640]",
        "",
        "mean: [0.485, 0.456, 0.406]",
        "std:  [0.229, 0.224, 0.225]",
        "",
        "path: " + str(dst_root.resolve()),
        "train: train/images",
        "val:   val/images",
        "test:  test/images",
        "",
        "# Per-source loss masking — load valid_classes.json at runtime",
        "valid_classes_json: valid_classes.json",
    ]
    (dst_root / "data.yaml").write_text("\n".join(lines) + "\n")


def _write_report(dst_root: Path, results: dict[str, dict]) -> None:
    lines = ["# Unified Detection — DATASET_REPORT", ""]
    lines.append("## Per-source")
    lines.append("")
    lines.append("| source | train | val | test | layout |")
    lines.append("|---|---:|---:|---:|---|")
    totals = defaultdict(int)
    for key, r in results.items():
        if r.get("missing"):
            continue
        sc = r["split_counts"]
        lines.append(f"| {key} | {sc.get('train', 0)} | {sc.get('val', 0)} | {sc.get('test', 0)} | {SOURCES[key]['layout']} |")
        for split, n in sc.items():
            totals[split] += n
    lines.append(f"| **TOTAL** | **{totals['train']}** | **{totals['val']}** | **{totals['test']}** | |")
    lines.append("")

    # per-class counts
    class_counts: dict[int, int] = defaultdict(int)
    for r in results.values():
        if r.get("missing"):
            continue
        for cid, n in r["class_counts"].items():
            class_counts[cid] += n
    lines.append("## Per-class instance counts (across all splits)")
    lines.append("")
    lines.append("| ID | name | count | status |")
    lines.append("|---:|---|---:|---|")
    for cid in sorted(UNIFIED_NAMES):
        n = class_counts.get(cid, 0)
        status = "✅" if n > 0 else "🅁 reserved"
        lines.append(f"| {cid} | `{UNIFIED_NAMES[cid]}` | {n:,} | {status} |")
    lines.append("")
    (dst_root / "DATASET_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk sources and count without copying anything.")
    parser.add_argument("--out", default=str(OUT_ROOT),
                        help=f"Output root (default: {OUT_ROOT})")
    args = parser.parse_args()

    dst_root = Path(args.out)
    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output root: {} | dry_run={}", dst_root, args.dry_run)

    results: dict[str, dict] = {}
    valid_classes_per_image: dict[str, list[int]] = {}
    for key, cfg in SOURCES.items():
        logger.info("=== source: {} ===", key)
        r = _process_source(key, cfg, dst_root, args.dry_run)
        results[key] = r
        if not r.get("missing"):
            valid_classes_per_image.update(r["valid_classes_per_image"])

    if not args.dry_run:
        (dst_root / "valid_classes.json").write_text(
            json.dumps(valid_classes_per_image, indent=0)
        )
        _write_data_yaml(dst_root)
        _write_report(dst_root, results)
        logger.info("Wrote valid_classes.json ({} entries)", len(valid_classes_per_image))
        logger.info("Wrote data.yaml + DATASET_REPORT.md")
    else:
        # dry-run: print summary
        for key, r in results.items():
            if r.get("missing"):
                logger.warning("  {}: MISSING", key)
                continue
            sc = r["split_counts"]
            logger.info(
                "  {}: train={} val={} test={} | classes={}",
                key, sc.get("train", 0), sc.get("val", 0), sc.get("test", 0),
                sorted(r["class_counts"].keys()),
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
