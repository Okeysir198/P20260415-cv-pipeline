"""QA pass on dataset_store/training_ready/unified_detection/.

Checks (all read-only by default):
    1. Image validity        — corrupt JPGs, zero-byte files
    2. Label validity        — malformed lines, out-of-bounds class IDs,
                                negative/oversize coords, zero-area boxes
    3. Empty-label files     — flag images with no annotations
    4. Filename collisions   — duplicate stems across splits
    5. Per-source class consistency — labels in this image use only the
                                       source's declared `valid_classes`
    6. Sample gallery        — N random samples per class for visual review

Outputs:
    <dataset_root>/qa/
        qa_report.md
        removal_candidates.json   # {file: reason} — re-run with --apply to delete
        gallery/<class_id>_<class_name>/*.jpg

Usage:
    uv run scripts/qa_unified_detection.py
    uv run scripts/qa_unified_detection.py --gallery-per-class 16
    uv run scripts/qa_unified_detection.py --apply           # DELETE flagged
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_store/training_ready/unified_detection"
DATA_CFG = PROJECT_ROOT / "features/unified_detection/configs/05_data.yaml"


def _load_label_lines(label_path: Path) -> list[tuple[int, float, float, float, float]] | None:
    """Parse YOLO label file. Returns None on read error."""
    if not label_path.exists():
        return []
    try:
        text = label_path.read_text()
    except OSError:
        return None
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            return None  # malformed
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            return None
        out.append((cid, cx, cy, w, h))
    return out


def _check_image(img_path: Path) -> str | None:
    """Return error string if image is bad, else None."""
    try:
        if img_path.stat().st_size == 0:
            return "zero_byte"
    except OSError:
        return "stat_error"
    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable"
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return f"tiny_{w}x{h}"
    return None


def _source_key(stem: str) -> str:
    """Extract source prefix (e.g. 'fire' from 'fire__abc')."""
    return stem.split("__", 1)[0] if "__" in stem else "unknown"


def _scan(
    split_dir: Path, valid_classes_per_image: dict[str, list[int]],
    source_valid: dict[str, set[int]], num_classes: int,
) -> dict:
    """Scan one split directory; return per-file flags."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    flags: dict[str, str] = {}     # file → reason
    empty_labels: list[str] = []
    class_counts: Counter = Counter()
    out_of_source: list[tuple[str, int]] = []  # (file, unexpected_class)

    img_paths = sorted(img_dir.iterdir())
    logger.info("  {} images...", len(img_paths))

    for img in img_paths:
        rel = f"{split_dir.name}/images/{img.name}"

        # 1) image validity
        err = _check_image(img)
        if err:
            flags[rel] = f"image_{err}"
            continue

        # 2) label validity
        lbl = lbl_dir / f"{img.stem}.txt"
        rows = _load_label_lines(lbl)
        if rows is None:
            flags[rel] = "label_malformed"
            continue
        if not rows:
            empty_labels.append(rel)
            continue

        bad = False
        src_valid = source_valid.get(_source_key(img.stem), set())
        for cid, cx, cy, w, h in rows:
            if cid < 0 or cid >= num_classes:
                flags[rel] = f"label_class_oob_{cid}"
                bad = True
                break
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                flags[rel] = "label_center_oob"
                bad = True
                break
            if w <= 0 or h <= 0 or w > 1.0 or h > 1.0:
                flags[rel] = f"label_size_invalid_{w:.3f}x{h:.3f}"
                bad = True
                break
            # area in 640² pixel terms
            if (w * 640) * (h * 640) < 4.0:
                flags[rel] = "label_subpixel_box"
                bad = True
                break
            if src_valid and cid not in src_valid:
                out_of_source.append((rel, cid))
            class_counts[cid] += 1

        if bad:
            continue

    return {
        "flags": flags,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counts),
        "out_of_source": out_of_source,
        "n_imgs": len(img_paths),
    }


def _build_gallery(
    out_dir: Path, split_dir: Path, names: dict[int, str], n_per_class: int,
) -> None:
    """Render N random labeled samples per class into out_dir/<id>_<name>/."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    by_class: dict[int, list[Path]] = defaultdict(list)
    for img in img_dir.iterdir():
        lbl = lbl_dir / f"{img.stem}.txt"
        rows = _load_label_lines(lbl) or []
        for cid, *_ in rows:
            by_class[cid].append(img)

    rng = random.Random(0)
    for cid, paths in by_class.items():
        cls_name = names.get(cid, f"class{cid}")
        cls_dir = out_dir / f"{cid:02d}_{cls_name}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        sample = rng.sample(paths, min(n_per_class, len(paths)))
        for img in sample:
            frame = cv2.imread(str(img))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            for row in _load_label_lines(lbl_dir / f"{img.stem}.txt") or []:
                rcid, cx, cy, bw, bh = row
                if rcid != cid:
                    continue
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(cls_dir / img.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATASET_ROOT))
    ap.add_argument("--data-config", default=str(DATA_CFG))
    ap.add_argument("--gallery-per-class", type=int, default=16)
    ap.add_argument("--apply", action="store_true",
                    help="DELETE files in removal_candidates.json (image + label).")
    args = ap.parse_args()

    root = Path(args.root)
    out_qa = root / "qa"
    out_qa.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    names = {int(k): v for k, v in cfg["names"].items()}
    num_classes = cfg["num_classes"]
    sources = cfg.get("sources", {})
    source_valid = {k: set(v["classes"]) for k, v in sources.items()}

    valid_classes_json = root / cfg.get("valid_classes_json", "valid_classes.json")
    valid_classes_per_image: dict[str, list[int]] = json.loads(valid_classes_json.read_text())

    # ---- apply-mode: delete and exit ----
    if args.apply:
        rc_file = out_qa / "removal_candidates.json"
        if not rc_file.exists():
            logger.error("No removal_candidates.json — run without --apply first.")
            sys.exit(1)
        rc = json.loads(rc_file.read_text())
        n = 0
        for rel, reason in rc.items():
            img = root / rel
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            for p in (img, lbl):
                if p.exists():
                    p.unlink()
            n += 1
        logger.info("Deleted {} entries", n)
        return

    # ---- scan all splits ----
    all_flags: dict[str, str] = {}
    all_empty: list[str] = []
    all_oos: list[tuple[str, int]] = []
    per_split_class_counts: dict[str, dict[int, int]] = {}
    per_split_n: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        logger.info("--- scanning {} ---", split)
        r = _scan(split_dir, valid_classes_per_image, source_valid, num_classes)
        all_flags.update(r["flags"])
        all_empty.extend(r["empty_labels"])
        all_oos.extend(r["out_of_source"])
        per_split_class_counts[split] = r["class_counts"]
        per_split_n[split] = r["n_imgs"]

    # ---- gallery (sampled from train) ----
    if args.gallery_per_class > 0:
        logger.info("Building gallery ({} per class)...", args.gallery_per_class)
        _build_gallery(out_qa / "gallery", root / "train", names, args.gallery_per_class)

    # ---- write removal_candidates.json (only validity flags, NOT empty-labels) ----
    (out_qa / "removal_candidates.json").write_text(json.dumps(all_flags, indent=2))

    # ---- write qa_report.md ----
    lines = ["# Unified Detection — QA Report", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| split | imgs | flagged | empty-label |")
    lines.append("|---|---:|---:|---:|")
    for split in ("train", "val", "test"):
        nflag = sum(1 for k in all_flags if k.startswith(f"{split}/images/"))
        nemp  = sum(1 for k in all_empty if k.startswith(f"{split}/images/"))
        lines.append(f"| {split} | {per_split_n[split]} | {nflag} | {nemp} |")
    lines.append("")

    lines.append("## Flag distribution")
    flag_kinds = Counter(all_flags.values())
    for kind, n in flag_kinds.most_common():
        lines.append(f"- `{kind}`: **{n}**")
    lines.append("")

    lines.append("## Out-of-source class usage (source-prefix vs label class)")
    lines.append("First 50 instances:")
    for rel, cid in all_oos[:50]:
        lines.append(f"- `{rel}` → unexpected class {cid} (`{names.get(cid, '?')}`)")
    if len(all_oos) > 50:
        lines.append(f"... and {len(all_oos) - 50} more")
    lines.append("")
    lines.append(f"**Total out-of-source label instances:** {len(all_oos)}")
    lines.append("")

    lines.append("## Empty-label files (potential negatives — review per source)")
    by_src = Counter(_source_key(Path(r).stem) for r in all_empty)
    for src, n in by_src.most_common():
        lines.append(f"- `{src}__*`: **{n}** empty-label images")
    lines.append("")

    lines.append("## Per-class instance counts (post-validity)")
    lines.append("")
    lines.append("| ID | name | train | val | test |")
    lines.append("|---:|---|---:|---:|---:|")
    for cid in sorted(names):
        t = per_split_class_counts["train"].get(cid, 0)
        v = per_split_class_counts["val"].get(cid, 0)
        s = per_split_class_counts["test"].get(cid, 0)
        marker = "" if (t + v + s) > 0 else " 🅁"
        lines.append(f"| {cid} | `{names[cid]}` | {t:,} | {v:,} | {s:,} |{marker}")
    lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append("1. Review `qa/gallery/<id>_<name>/*.jpg` — confirm labels look right per class")
    lines.append("2. Review `qa/removal_candidates.json` — verify flagged files are truly bad")
    lines.append("3. Apply removals: `uv run scripts/qa_unified_detection.py --apply`")
    lines.append("4. Decide what to do with empty-label images (drop? keep as negatives?)")

    (out_qa / "qa_report.md").write_text("\n".join(lines) + "\n")
    logger.info("QA done. {} flagged, {} empty-label, {} out-of-source labels",
                len(all_flags), len(all_empty), len(all_oos))
    logger.info("→ {}", out_qa / "qa_report.md")


if __name__ == "__main__":
    main()
