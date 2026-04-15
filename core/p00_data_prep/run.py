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
    config = load_config(str(path), base_dir=str(base_dir))

    if "output_dir" in config:
        config["output_dir"] = resolve_path(config["output_dir"], base_dir)

    for source in config.get("sources", []):
        if "path" in source:
            source["resolved_path"] = resolve_path(source["path"], base_dir)

    # Store config dir so run_data_prep can pass the right base_dir to parsers
    config["_config_dir"] = str(base_dir.resolve())

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
    if output_dir.exists() and not args.force:
        print(f"\n⚠️  Output directory exists: {output_dir}")
        print("   Use --force to overwrite")
        return

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

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 Processing samples...")
    file_ops = FileOps(handle_duplicates="rename")

    for sample in tqdm(samples, desc="Processing"):
        output_path = file_ops.copy_file(sample["image_path"], images_dir, source_name=sample["source"])

        if output_path is None:
            continue

        stem = output_path.stem  # Use actual output filename (may be renamed)
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            for obj in sample.get("objects", []):
                f.write(f"{obj['class_id']} {obj['cx']:.6f} {obj['cy']:.6f} {obj['w']:.6f} {obj['h']:.6f}\n")

        # Update filename in sample for splits.json
        sample["filename"] = output_path.name

    print(f"\n✂️  Generating splits ({ratios[0]:.0%}/{ratios[1]:.0%}/{ratios[2]:.0%})...")
    splitter = SplitGenerator(ratios=ratios, seed=seed, stratified=True)
    splits_file = output_dir / "splits.json"
    split_samples = [{"filename": s["filename"], "labels": s.get("labels", [])} for s in samples]
    splitter.generate_splits(split_samples, splits_file, task_type=task_type)

    # Generate dataset report
    report_path = output_dir / "DATASET_REPORT.md"
    _write_dataset_report(report_path, config, stats, splits_file, ratios)

    print(f"\n✅ Done! Dataset created at: {output_dir}")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print(f"   Splits: {splits_file}")
    print(f"   Report: {report_path}")


def resplit_only(output_dir: Path, ratios: tuple, seed: int, task_type: str) -> None:
    """Regenerate splits.json for an existing dataset without re-copying files."""
    output_dir = Path(output_dir)
    splits_file = output_dir / "splits.json"
    labels_dir = output_dir / "labels"

    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return

    samples = []
    for label_file in labels_dir.glob("*.txt"):
        labels = []
        try:
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        labels.append(parts[0])
        except (OSError, IOError):
            pass

        # Find the corresponding image filename with extension
        found_name = label_file.stem  # fallback: stem only if no image found
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            test_path = output_dir / "images" / f"{label_file.stem}{ext}"
            if test_path.exists():
                found_name = test_path.name
                break

        samples.append({"filename": found_name, "labels": labels})

    if not samples:
        print(f"❌ No samples found in {output_dir}")
        return

    print(f"   Found {len(samples)} samples")
    splitter = SplitGenerator(ratios=ratios, seed=seed, stratified=True)
    splitter.generate_splits(samples, splits_file, task_type=task_type)
    print(f"✅ Splits regenerated: {splits_file}")


def _write_dataset_report(
    report_path: Path,
    config: dict,
    stats: dict,
    splits_file: Path,
    ratios: tuple,
) -> None:
    """Generate DATASET_REPORT.md summarizing the prepared dataset."""
    dataset_name = config["dataset_name"]
    classes = config.get("classes", [])
    sources = config.get("sources", [])

    # Read split counts from splits.json
    split_counts = {"train": 0, "val": 0, "test": 0}
    if splits_file.exists():
        with open(splits_file) as f:
            splits_data = json.load(f)
        for split_name in split_counts:
            split_counts[split_name] = len(splits_data.get("splits", {}).get(split_name, []))

    total_images = sum(split_counts.values())
    total_objects = sum(stats.values())

    lines = [
        f"# Dataset Report — {dataset_name.replace('_', ' ').title()}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Config:** `configs/{dataset_name}/00_data_preparation.yaml`",
        "",
        "## Classes",
        "",
        "| ID | Name | Count | % |",
        "|----|------|------:|--:|",
    ]

    for i, name in enumerate(classes):
        count = stats.get(name, 0)
        pct = 100 * count / total_objects if total_objects > 0 else 0
        lines.append(f"| {i} | {name} | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Dataset Splits",
        "",
        f"Ratios: {ratios[0]:.0%} / {ratios[1]:.0%} / {ratios[2]:.0%}",
        "",
        "| Split | Images |",
        "|-------|-------:|",
        f"| Train | {split_counts['train']:,} |",
        f"| Val | {split_counts['val']:,} |",
        f"| Test | {split_counts['test']:,} |",
        f"| **Total** | **{total_images:,}** |",
        "",
        "## Raw Sources",
        "",
        "| # | Source | Format |",
        "|---|--------|--------|",
    ]

    for i, src in enumerate(sources, 1):
        lines.append(f"| {i} | {src.get('name', '?')} | {src.get('format', '?')} |")

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
