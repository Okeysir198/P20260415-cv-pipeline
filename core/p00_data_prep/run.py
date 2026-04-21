#!/usr/bin/env python3
"""
Generic CV Data Preparation Tool - CLI Entry Point

Combines multiple source datasets into training-ready datasets.
Supports: detection, classification, segmentation, pose estimation.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.p00_data_prep.adapters.detection import DetectionAdapter
from core.p00_data_prep.core.splitter import SplitGenerator
from core.p00_data_prep.utils.file_ops import FileOps
from utils.config import load_config, resolve_path


TASK_ADAPTERS = {
    "detection": DetectionAdapter,
    # "classification": ClassificationAdapter,
    # "segmentation": SegmentationAdapter,
    # "pose": PoseAdapter,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generic CV Data Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine helmet detection datasets
  python core/p00_data_prep/run.py --config features/ppe-helmet_detection/configs/00_data_preparation.yaml

  # Dry run to preview
  python core/p00_data_prep/run.py --config features/ppe-helmet_detection/configs/00_data_preparation.yaml --dry-run

  # Re-split only (change ratios without recombining)
  python core/p00_data_prep/run.py --config features/ppe-helmet_detection/configs/00_data_preparation.yaml --resplit-only --splits 0.85 0.1 0.05

  # Force overwrite existing dataset
  python core/p00_data_prep/run.py --config features/ppe-helmet_detection/configs/00_data_preparation.yaml --force
        """
    )

    parser.add_argument("--config", type=str, required=True, help="Path to data preparation config file")
    parser.add_argument("--splits", type=float, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"),
                        help="Split ratios (default: from config or 0.8 0.1 0.1)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    parser.add_argument("--resplit-only", action="store_true", help="Only regenerate splits.json, don't recombine data")

    return parser.parse_args()


def load_prep_config(config_path: str) -> dict:
    """Load data preparation config, resolving all paths relative to the config file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base_dir = path.parent
    config = load_config(str(path))

    if "output_dir" in config:
        config["output_dir"] = resolve_path(config["output_dir"], base_dir)

    for source in config.get("sources", []):
        if "path" in source:
            source["resolved_path"] = resolve_path(source["path"], base_dir)

    # Store config dir and path so run_data_prep can pass the right base_dir to parsers
    config["_config_dir"] = str(base_dir.resolve())
    config["_config_path"] = str(path.resolve())

    return config


def run_data_prep(config: dict, args) -> None:
    """Run data preparation pipeline."""
    print(f"📦 Data Preparation: {config['dataset_name']}")
    print(f"   Task: {config.get('task', 'detection')}")
    print(f"   Output: {config['output_dir']}")

    task_type = config.get("task", "detection")
    adapter_class = TASK_ADAPTERS.get(task_type)
    if adapter_class is None:
        raise ValueError(f"Unsupported task type: {task_type}")

    adapter = adapter_class(config)

    if args.splits:
        ratios = tuple(args.splits)
    else:
        splits_config = config.get("splits", {})
        ratios = (
            splits_config.get("train", 0.8),
            splits_config.get("val", 0.1),
            splits_config.get("test", 0.1)
        )

    seed = config.get("splits", {}).get("seed", 42)

    if args.resplit_only:
        print("\n🔄 Resplit mode (reusing combined data)")
        resplit_only(Path(config["output_dir"]), ratios, seed, task_type)
        return

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        if not args.force:
            print(f"\n⚠️  Output directory exists: {output_dir}")
            print("   Use --force to overwrite")
            return
        import shutil
        print(f"\n🧹 --force: wiping existing output dir {output_dir}")
        shutil.rmtree(output_dir)

    print(f"\n📥 Merging {len(config.get('sources', []))} source datasets...")

    base_dir = Path(config["_config_dir"])
    samples = adapter.merge_sources(base_dir)

    print(f"   Found {len(samples)} samples")

    if not samples:
        print("❌ No samples found!")
        return

    stats = adapter.get_class_statistics(samples)
    print("\n📊 Class distribution:")
    total = sum(stats.values())
    for class_name, count in stats.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"   {class_name}: {count} ({pct:.1f}%)")

    if args.dry_run:
        print(f"\n🔍 Dry run mode - {len(samples)} samples would be processed")
        return

    # Assign splits BEFORE copying so we can write each sample directly into its
    # split subdir. No flat images/ or labels/ dir, no symlinks.
    from core.p00_data_prep.core.splitter import (
        ensure_split_dirs,
        write_audit_snapshot,
    )

    # Separate pre-split sources (has_splits=True) from flat sources needing random splitting.
    pre_assigned: list[tuple[int, str]] = []  # (idx, split)
    to_split_input: list[dict] = []
    _VALID_SPLITS = {"train", "val", "test"}
    _SPLIT_ALIASES = {"valid": "val"}

    for i, s in enumerate(samples):
        if "original_split" in s:
            raw = s["original_split"]
            mapped = _SPLIT_ALIASES.get(raw, raw)
            pre_assigned.append((i, mapped if mapped in _VALID_SPLITS else "train"))
        else:
            to_split_input.append({"_idx": i, "filename": Path(s["image_path"]).name, "labels": s.get("labels", [])})

    n_pre = len(pre_assigned)
    n_random = len(to_split_input)
    print(f"\n✂️  Assigning splits ({ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%})...")
    print(f"   Pre-split (scene-aware): {n_pre}  |  Random-split: {n_random}")

    idx_to_split: dict[int, str] = {i: split for i, split in pre_assigned}

    if to_split_input:
        splitter = SplitGenerator(ratios=ratios, seed=seed, stratified=True)
        assignment = splitter.assign_splits(to_split_input)  # {split: [dict, ...]}
        idx_to_split.update({item["_idx"]: split for split, items in assignment.items() for item in items})

    ensure_split_dirs(output_dir)
    split_dirs = {
        split: {"images": output_dir / split / "images", "labels": output_dir / split / "labels"}
        for split in ("train", "val", "test")
    }

    print("\n📝 Processing samples...")
    file_ops = FileOps(handle_duplicates="rename")

    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        split = idx_to_split.get(i, "train")
        output_path = file_ops.copy_file(
            sample["image_path"], split_dirs[split]["images"], source_name=sample["source"]
        )
        if output_path is None:
            continue

        stem = output_path.stem  # Use actual output filename (may be renamed)
        label_path = split_dirs[split]["labels"] / f"{stem}.txt"
        with open(label_path, "w") as f:
            for obj in sample.get("objects", []):
                f.write(f"{obj['class_id']} {obj['cx']:.6f} {obj['cy']:.6f} {obj['w']:.6f} {obj['h']:.6f}\n")

        sample["filename"] = output_path.name
        sample["_split"] = split

    # Audit snapshot (counts + ratios + seed), not a filename list.
    counts = {split: sum(1 for s in samples if s.get("_split") == split) for split in ("train", "val", "test")}
    splits_file = output_dir / "splits.json"
    write_audit_snapshot(
        splits_file,
        counts=counts,
        ratios=ratios,
        seed=seed,
        task_type=task_type,
        total=sum(counts.values()),
    )

    # Count images actually written per source
    source_counts: Dict[str, int] = {}
    for sample in samples:
        if "_split" in sample:
            src = sample.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

    # Generate dataset report
    report_path = output_dir / "DATASET_REPORT.md"
    _write_dataset_report(
        report_path, config, stats, splits_file, ratios, source_counts,
        config_path=config.get("_config_path", ""),
    )

    print(f"\n✅ Done! Dataset created at: {output_dir}")
    for split, d in split_dirs.items():
        print(f"   {split}/: {counts[split]} imgs ({d['images']})")
    print(f"   Snapshot: {splits_file}")
    print(f"   Report: {report_path}")


def resplit_only(output_dir: Path, ratios: tuple, seed: int, task_type: str) -> None:
    """Rescan existing split subdirs, reassign samples, physically move files."""
    from core.p00_data_prep.core.splitter import (
        IMAGE_EXTS,
        SPLIT_NAMES,
        ensure_split_dirs,
        move_sample,
        rescan_splits,
        write_audit_snapshot,
    )

    output_dir = Path(output_dir)
    ensure_split_dirs(output_dir)

    # Collect every existing sample from all split subdirs.
    samples = []
    for split in SPLIT_NAMES:
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        if not img_dir.is_dir():
            continue
        for ext in IMAGE_EXTS:
            for img_path in img_dir.glob(f"*{ext}"):
                stem = img_path.stem
                label_path = lbl_dir / f"{stem}.txt"
                labels = []
                if label_path.exists():
                    try:
                        for line in label_path.read_text().splitlines():
                            parts = line.strip().split()
                            if parts:
                                labels.append(parts[0])
                    except (OSError, IOError):
                        pass
                samples.append({
                    "stem": stem,
                    "filename": img_path.name,
                    "current_split": split,
                    "labels": labels,
                })

    if not samples:
        print(f"❌ No samples found under {output_dir}/{{train,val,test}}/images")
        return

    print(f"   Found {len(samples)} samples across existing splits")

    # Reassign.
    splitter = SplitGenerator(ratios=ratios, seed=seed, stratified=True)
    assignment = splitter.assign_splits(samples)
    new_split_of: Dict[str, str] = {}
    for split, items in assignment.items():
        for item in items:
            new_split_of[item["stem"]] = split

    # Physically move samples whose split changed.
    n_moved = 0
    for s in samples:
        new_split = new_split_of[s["stem"]]
        if new_split != s["current_split"]:
            move_sample(output_dir, s["stem"], s["current_split"], new_split)
            n_moved += 1

    counts = rescan_splits(output_dir)
    write_audit_snapshot(
        output_dir / "splits.json",
        counts={k: len(v) for k, v in counts.items()},
        ratios=ratios,
        seed=seed,
        task_type=task_type,
        total=sum(len(v) for v in counts.values()),
    )
    print(f"✅ Resplit done: {n_moved} samples moved. Snapshot: {output_dir / 'splits.json'}")


def _ascii_bar(count: int, max_count: int, width: int = 12) -> str:
    """Fixed-width ASCII progress bar scaled to count/max_count."""
    filled = round(width * count / max_count) if max_count > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _scan_label_files(output_dir: Path, classes: list) -> Tuple[dict, dict]:
    """One-pass scan of all label files in output_dir/{train,val,test}/labels/.

    Returns:
        per_split: {split: {class_name: annotation_count}}
        size_tiers: {"tiny": n, "small": n, "medium": n, "large": n}

    Size tiers use relative bbox area (w×h), calibrated for ~640 px images:
        tiny   < 0.000479  (≈ < 14² px)
        small  0.000479–0.0025   (≈ 14²–32² px)
        medium  0.0025–0.0225   (≈ 32²–96² px)
        large  ≥ 0.0225   (≈ > 96² px)
    """
    class_id_to_name = {i: name for i, name in enumerate(classes)}
    per_split = {s: {c: 0 for c in classes} for s in ("train", "val", "test")}
    size_tiers = {"tiny": 0, "small": 0, "medium": 0, "large": 0}

    for split in ("train", "val", "test"):
        label_dir = output_dir / split / "labels"
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            try:
                text = label_file.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                area = w * h
                name = class_id_to_name.get(class_id)
                if name:
                    per_split[split][name] += 1
                if area < 0.000479:
                    size_tiers["tiny"] += 1
                elif area < 0.0025:
                    size_tiers["small"] += 1
                elif area < 0.0225:
                    size_tiers["medium"] += 1
                else:
                    size_tiers["large"] += 1

    return per_split, size_tiers


def _write_dataset_report(
    report_path: Path,
    config: dict,
    stats: dict,
    splits_file: Path,
    ratios: tuple,
    source_counts: Dict[str, int],
    config_path: str = "",
) -> None:
    """Generate DATASET_REPORT.md summarizing the prepared dataset."""
    dataset_name = config["dataset_name"]
    classes = config.get("classes", [])
    sources = config.get("sources", [])
    output_dir = Path(config["output_dir"])

    # ── Split counts from audit snapshot ────────────────────────────────────
    split_counts = {"train": 0, "val": 0, "test": 0}
    if splits_file.exists():
        with open(splits_file) as f:
            splits_data = json.load(f)
        for split_name in split_counts:
            split_counts[split_name] = splits_data.get("counts", {}).get(split_name, 0)

    total_images = sum(split_counts.values())
    total_annotations = sum(stats.values())
    avg_ann = total_annotations / total_images if total_images > 0 else 0.0

    # ── Per-split per-class counts + bbox size tiers ─────────────────────────
    per_split, size_tiers = _scan_label_files(output_dir, classes)
    per_split_ann = {s: sum(per_split[s].values()) for s in per_split}
    total_size_anns = sum(size_tiers.values())

    # ── Class imbalance ──────────────────────────────────────────────────────
    counts_list = [stats.get(c, 0) for c in classes]
    max_count = max(counts_list) if counts_list else 0
    nonzero = [c for c in counts_list if c > 0]
    min_count = min(nonzero) if nonzero else 1
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # ── Config path relative to project root ────────────────────────────────
    if config_path:
        try:
            rel_config = str(Path(config_path).relative_to(output_dir.parent.parent.parent))
        except ValueError:
            rel_config = config_path
    else:
        rel_config = f"features/.../configs/00_data_preparation.yaml"

    # ════════════════════════════════════════════════════════════════════════
    title = dataset_name.replace("_", " ").title()
    lines = [
        f"# Dataset Report — {title}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Config:** `{rel_config}`",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total images | {total_images:,} |",
        f"| Total annotations | {total_annotations:,} |",
        f"| Avg annotations/image | {avg_ann:.2f} |",
        f"| Classes | {len(classes)} ({', '.join(classes)}) |",
        f"| Sources used | {len(sources)} |",
        f"| Split ratio | {ratios[0]:.0%} / {ratios[1]:.0%} / {ratios[2]:.0%} |",
        "",
        "---",
        "",
        "## Class Distribution",
        "",
        "| ID | Class | Annotations | % | Avg/img | Bar |",
        "|----|-------|------------:|--:|--------:|-----|",
    ]

    for i, name in enumerate(classes):
        count = stats.get(name, 0)
        pct = 100 * count / total_annotations if total_annotations > 0 else 0.0
        avg = count / total_images if total_images > 0 else 0.0
        bar = _ascii_bar(count, max_count)
        lines.append(f"| {i} | {name} | {count:,} | {pct:.1f}% | {avg:.2f} | {bar} |")

    imbalance_flag = "✅" if imbalance_ratio <= 3.0 else "⚠️"
    lines += [
        "",
        f"**Imbalance ratio:** {imbalance_ratio:.2f}× {imbalance_flag} (threshold 3×)",
        "",
        "---",
        "",
        "## Split Distribution",
        "",
    ]

    class_cols = " | ".join(classes)
    sep_cols = " | ".join("------:" for _ in classes)
    lines += [
        f"| Split | Images | Annotations | Ann/img | {class_cols} |",
        f"|-------|-------:|------------:|--------:|{sep_cols}|",
    ]
    for split in ("train", "val", "test"):
        n_img = split_counts[split]
        n_ann = per_split_ann[split]
        a_img = n_ann / n_img if n_img > 0 else 0.0
        cls_vals = " | ".join(f"{per_split[split].get(c, 0):,}" for c in classes)
        lines.append(f"| {split.title()} | {n_img:,} | {n_ann:,} | {a_img:.2f} | {cls_vals} |")
    cls_totals = " | ".join(f"{stats.get(c, 0):,}" for c in classes)
    lines += [
        f"| **Total** | **{total_images:,}** | **{total_annotations:,}** | **{avg_ann:.2f}** | {cls_totals} |",
        "",
        "---",
        "",
        "## Split Strategy",
        "",
        "> **Author-defined** — original train/val/test dirs from the source dataset are honored as-is.",
        "> Scenes are guaranteed not to leak across splits if the dataset authors split by scene (e.g. video-derived datasets).",
        "> **Random stratified** — shuffled by class label; no scene-separation guarantee.",
        "",
        "| Source | Strategy | Basis |",
        "|--------|----------|-------|",
    ]
    for src in sources:
        src_name = src.get("name", "?")
        if src.get("has_splits"):
            strategy = "Author-defined ✅"
            basis = f"splits: {', '.join(src.get('splits_to_use', []))}"
        else:
            strategy = "Random stratified"
            basis = f"flat source — {ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%} seed={config.get('splits', {}).get('seed', 42)}"
        lines.append(f"| {src_name} | {strategy} | {basis} |")

    lines += [
        "",
        "---",
        "",
        "## Class Mapping — Raw → Training Ready",
        "",
    ]

    for src in sources:
        src_name = src.get("name", "?")
        class_map = src.get("class_map", {})
        dropped = src.get("dropped_classes", [])
        lines += [
            f"### {src_name}",
            "",
            "| Source class | → | Target class | Action |",
            "|-------------|---|-------------|--------|",
        ]
        for raw_cls, target_cls in class_map.items():
            stripped = raw_cls.strip().lstrip("-")
            if stripped.isdigit():
                action = "numeric ID → name"
            elif raw_cls == target_cls:
                action = "identity"
            else:
                action = "renamed"
            lines.append(f"| `{raw_cls}` | → | `{target_cls}` | {action} |")
        for dc in dropped:
            lines.append(f"| `{dc}` | — | *(dropped)* | excluded |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Source Contributions",
        "",
        "| # | Source | Images | Format | License | Notes |",
        "|---|--------|-------:|--------|---------|-------|",
    ]
    for i, src in enumerate(sources, 1):
        src_name = src.get("name", "?")
        n_imgs = source_counts.get(src_name, 0)
        fmt = src.get("format", "?")
        license_ = src.get("license", "—")
        notes = src.get("notes", "")
        if n_imgs == 0 and src.get("optional"):
            notes = f"⚠️ empty — {notes}" if notes else "⚠️ empty"
        elif n_imgs == 0:
            notes = f"⚠️ 0 images contributed — {notes}" if notes else "⚠️ 0 images contributed"
        lines.append(f"| {i} | {src_name} | {n_imgs:,} | {fmt} | {license_} | {notes} |")

    lines += [
        "",
        "---",
        "",
        "## Annotation Size Distribution",
        "",
        "> Relative bbox area (w×h). Calibrated for ~640 px images: "
        "tiny ≈ <14² px, small ≈ 14²–32² px, medium ≈ 32²–96² px, large ≈ >96² px.",
        "",
        "| Tier | Annotations | % | Criterion |",
        "|------|------------:|--:|-----------|",
    ]
    for tier, label, criterion in [
        ("tiny",   "Tiny",   "w×h < 0.000479"),
        ("small",  "Small",  "0.000479 ≤ w×h < 0.0025"),
        ("medium", "Medium", "0.0025 ≤ w×h < 0.0225"),
        ("large",  "Large",  "w×h ≥ 0.0225"),
    ]:
        n = size_tiers[tier]
        pct = 100 * n / total_size_anns if total_size_anns > 0 else 0.0
        lines.append(f"| {label} | {n:,} | {pct:.1f}% | {criterion} |")

    # ── Caveats ──────────────────────────────────────────────────────────────
    held_back = config.get("held_back", [])
    caveats = []

    for src in sources:
        n_imgs = source_counts.get(src.get("name", ""), 0)
        if n_imgs == 0:
            name = src.get("name", "?")
            note = src.get("notes", "")
            suffix = f" ({note})" if note else ""
            optional_tag = "optional; " if src.get("optional") else ""
            caveats.append(f"⚠️ `{name}` → 0 images ({optional_tag}populate to reduce domain gap{suffix})")

    if imbalance_ratio > 3.0:
        min_cls = min(classes, key=lambda c: stats.get(c, 0))
        max_cls = max(classes, key=lambda c: stats.get(c, 0))
        caveats.append(
            f"⚠️ Class imbalance {imbalance_ratio:.1f}× "
            f"(`{max_cls}` vs `{min_cls}`) — consider rebalancing or weighted loss"
        )
    else:
        caveats.append(f"✅ Class balance healthy ({imbalance_ratio:.2f}× ratio, threshold 3×)")

    for h in held_back:
        name = h.get("name", "?")
        reason = h.get("reason", "")
        when = h.get("when", "")
        entry = f"ℹ️ `{name}` held back"
        if reason:
            entry += f" — {reason}"
        if when:
            entry += f" (add when: {when})"
        caveats.append(entry)

    lines += [
        "",
        "---",
        "",
        "## Caveats",
        "",
    ]
    for c in caveats:
        lines.append(f"- {c}")
    lines.append("")

    report_path.write_text("\n".join(lines))


def main():
    """Main entry point."""
    args = parse_args()
    config = load_prep_config(args.config)

    try:
        run_data_prep(config, args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
