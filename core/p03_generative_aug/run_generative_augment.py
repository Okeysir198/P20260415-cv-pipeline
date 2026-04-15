#!/usr/bin/env python3
"""Generative Augmentation CLI -- Replace object classes via inpainting.

Uses LangGraph + SAM3 + inpainting to augment datasets by replacing objects
of one class with another (e.g., head_with_helmet -> head_without_helmet) via segmentation and
generative inpainting.

Usage:
    # With config file
    python run_generative_augment.py --data-config features/ppe-helmet_detection/configs/05_data.yaml \\
        --config features/ppe-helmet_detection/configs/03_generative_augment.yaml --dry-run

    # With CLI overrides
    python run_generative_augment.py --data-config features/ppe-helmet_detection/configs/05_data.yaml \\
        --source-class head_with_helmet --target-class head_without_helmet \\
        --prompts "a person with bare head,a person without helmet,uncovered head"

    # Specific splits only
    python run_generative_augment.py --data-config features/ppe-shoes_detection/configs/05_data.yaml \\
        --source-class foot_with_safety_shoes --target-class foot_without_safety_shoes \\
        --prompts "regular shoes,sneakers,bare feet" --splits train val
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import generate_run_dir, load_config, merge_configs, parse_overrides
from utils.service_health import require_services


def _resolve_class_id(class_name: str, class_names: Dict[int, str]) -> int:
    """Resolve a class name to its integer ID.

    Args:
        class_name: Class name string (or integer string).
        class_names: Mapping of class_id to class name.

    Returns:
        Integer class ID.

    Raises:
        ValueError: If class name not found.
    """
    # Try as integer first
    try:
        return int(class_name)
    except ValueError:
        pass

    # Search by name
    for cls_id, name in class_names.items():
        if name == class_name:
            return cls_id

    raise ValueError(
        f"Class '{class_name}' not found in class names: {list(class_names.values())}"
    )


def main() -> None:
    """Run the generative augmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generative Augmentation: Replace object classes via SAM3 + inpainting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_generative_augment.py --data-config features/ppe-helmet_detection/configs/05_data.yaml \\
      --config features/ppe-helmet_detection/configs/03_generative_augment.yaml --dry-run

  python run_generative_augment.py --data-config features/ppe-helmet_detection/configs/05_data.yaml \\
      --source-class head_with_helmet --target-class head_without_helmet \\
      --prompts "a person with bare head,a person without helmet"

  python run_generative_augment.py --data-config features/ppe-shoes_detection/configs/05_data.yaml \\
      --source-class foot_with_safety_shoes --target-class foot_without_safety_shoes \\
      --prompts "regular shoes,sneakers" --splits train val
""",
    )

    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to data YAML config (e.g., features/ppe-helmet_detection/configs/05_data.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to generative augment config (default: configs/_shared/03_generative_augment.yaml)",
    )
    parser.add_argument(
        "--source-class",
        type=str,
        default=None,
        help="Source class name or ID to replace (overrides config)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        help="Target class name or ID for replacement (overrides config)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help='Replacement prompts, comma-separated (e.g., "bare head,no helmet")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing augmented images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Which splits to process (e.g., train val test)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides (e.g., processing.batch_size=4)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip service health checks before starting",
    )

    args = parser.parse_args()

    # Load data config
    data_config_path = Path(args.data_config).resolve()
    data_config = load_config(data_config_path)
    config_dir = str(data_config_path.parent)
    dataset_name = data_config.get("dataset_name", "unknown")
    class_names: Dict[int, str] = {int(k): v for k, v in data_config["names"].items()}

    # Load augment config
    augment_config_path = args.config
    if augment_config_path is None:
        augment_config_path = (
            Path(__file__).resolve().parent.parent.parent / "configs" / "_shared" / "03_generative_augment.yaml"
        )
    else:
        augment_config_path = Path(augment_config_path).resolve()

    if not Path(augment_config_path).exists():
        print(f"Error: Augment config not found at {augment_config_path}")
        sys.exit(1)

    augment_config = load_config(augment_config_path)

    # Service health checks
    sam3_url = augment_config.get("sam3", {}).get("service_url", "http://localhost:18100")
    inpainting = augment_config.get("inpainting", {})
    services_to_check: Dict[str, str] = {"SAM3 :18100": f"{sam3_url}/health"}
    if inpainting.get("mode") == "service":
        flux_url = inpainting.get("service_url", "http://localhost:8002")
        services_to_check["Flux inpainting"] = f"{flux_url}/health"
    require_services(services_to_check, skip=args.skip_health_check)

    # Apply overrides
    if args.override:
        overrides = parse_overrides(args.override)
        augment_config = merge_configs(augment_config, overrides)

    # Resolve source and target classes
    ga_config = augment_config.get("generative_augment", {})

    source_class_str = args.source_class or ga_config.get("source_class")
    target_class_str = args.target_class or ga_config.get("target_class")

    if source_class_str is None:
        parser.error("--source-class is required (or set generative_augment.source_class in config)")
    if target_class_str is None:
        parser.error("--target-class is required (or set generative_augment.target_class in config)")

    source_class_id = _resolve_class_id(source_class_str, class_names)
    target_class_id = _resolve_class_id(target_class_str, class_names)

    # Resolve replacement prompts
    if args.prompts:
        replacement_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        replacement_prompts = ga_config.get("replacement_prompts", [])

    if not replacement_prompts:
        parser.error(
            "--prompts is required (or set generative_augment.replacement_prompts in config)"
        )

    # Determine processing options
    processing = augment_config.get("processing", {})
    dry_run = args.dry_run or processing.get("dry_run", False)

    # Update splits in config if provided via CLI
    if args.splits:
        augment_config.setdefault("generative_augment", {})["splits"] = args.splits

    # Determine output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(generate_run_dir(data_config["dataset_name"], "03_generative_aug"))

    # Build initial state
    source_name = class_names.get(source_class_id, str(source_class_id))
    target_name = class_names.get(target_class_id, str(target_class_id))

    initial_state = {
        "data_config": data_config,
        "augment_config": augment_config,
        "dataset_name": dataset_name,
        "class_names": class_names,
        "config_dir": config_dir,
        "source_class_id": source_class_id,
        "target_class_id": target_class_id,
        "replacement_prompts": replacement_prompts,
        "image_paths": {},
        "total_images": 0,
        "current_batch_idx": 0,
        "total_batches": 0,
        "batch_size": processing.get("batch_size", 16),
        "image_results": [],
        "dry_run": dry_run,
        "summary": {},
        "report_path": "",
        "output_dir": output_dir,
    }

    # Print header
    print("=" * 70)
    print(f"  Generative Augmentation: {dataset_name}")
    print("=" * 70)
    print(f"  Data config   : {data_config_path}")
    print(f"  Classes       : {list(class_names.values())}")
    print(f"  Source class   : {source_class_id} ({source_name})")
    print(f"  Target class   : {target_class_id} ({target_name})")
    print(f"  Prompts       : {replacement_prompts}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Dry run       : {dry_run}")
    print()

    # Build and run graph
    from core.p03_generative_aug.graph import build_graph

    start_time = time.time()
    app = build_graph()

    print("Running generative augmentation pipeline...")
    result = app.invoke(initial_state)

    elapsed = time.time() - start_time

    # Print summary
    summary = result.get("summary", {})
    print()
    print("=" * 70)
    print(f"  Generative Augmentation Complete -- {dataset_name}")
    print("=" * 70)
    print(f"  Source class      : {summary.get('source_class', 'N/A')}")
    print(f"  Target class      : {summary.get('target_class', 'N/A')}")
    print(f"  Images processed  : {summary.get('total_images_processed', 0)}")
    print(f"  Objects segmented : {summary.get('total_objects_segmented', 0)}")
    print(f"  Augmented written : {summary.get('total_augmented_written', 0)}")

    per_prompt = summary.get("per_prompt_counts", {})
    if per_prompt:
        print("  Per-prompt:")
        for prompt, count in sorted(per_prompt.items(), key=lambda x: -x[1]):
            print(f"    {prompt:<40}: {count}")

    per_split = summary.get("per_split_counts", {})
    if per_split:
        print("  Per-split:")
        for split_name, count in sorted(per_split.items()):
            print(f"    {split_name:<20}: {count}")

    timing = summary.get("timing", {})
    if timing:
        print(f"  Avg segment/sample: {timing.get('avg_segment_s', 0):.4f}s")
        print(f"  Avg inpaint/sample: {timing.get('avg_inpaint_s', 0):.4f}s")
        print(f"  Avg total/sample  : {timing.get('avg_total_per_sample_s', 0):.4f}s")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print(f"  Report saved to   : {result.get('report_path', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
