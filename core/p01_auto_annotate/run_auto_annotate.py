#!/usr/bin/env python3
"""Auto-Annotation CLI -- Generate YOLO labels using SAM3.

Uses LangGraph + SAM3 to automatically annotate images with bounding boxes
and/or segmentation polygons.

Usage:
    # YOLO layout mode
    python run_auto_annotate.py --data-config ../features/safety-fire_detection/configs/05_data.yaml --mode text
    python run_auto_annotate.py --data-config ../features/ppe-shoes_detection/configs/05_data.yaml --dry-run --mode text

    # Flat directory mode
    python run_auto_annotate.py --image-dir /path/to/images --classes "0:person,1:car" --mode text

    # With text prompt overrides
    python run_auto_annotate.py --data-config ../features/safety-fire_detection/configs/05_data.yaml \\
        --text-prompts "fire=flames and burning,smoke=gray or white smoke" --mode text
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.yolo_io import parse_classes
from utils.config import load_config, merge_configs, parse_overrides
from utils.service_health import require_services


def parse_text_prompts(prompts_str: str) -> Dict[str, str]:
    """Parse CLI text prompts string like 'person=a standing person,car=a vehicle'.

    Args:
        prompts_str: Comma-separated "name=prompt" pairs.

    Returns:
        Mapping of class name to text prompt.
    """
    prompts: Dict[str, str] = {}
    for pair in prompts_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        name, prompt = pair.split("=", 1)
        prompts[name.strip()] = prompt.strip()
    return prompts


def main() -> None:
    """Run the auto-annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Auto-Annotation: Generate YOLO labels using SAM3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_auto_annotate.py --data-config features/safety-fire_detection/configs/05_data.yaml --mode text
  python run_auto_annotate.py --data-config features/ppe-shoes_detection/configs/05_data.yaml --dry-run
  python run_auto_annotate.py --image-dir /path/to/images --classes "0:person,1:car" --mode text
  python run_auto_annotate.py --data-config features/safety-fire_detection/configs/05_data.yaml \\
      --text-prompts "fire=flames,smoke=gray smoke" --mode hybrid
""",
    )

    # Input mode (mutually exclusive-ish)
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Path to data YAML config (e.g., features/safety-fire_detection/configs/05_data.yaml)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to flat image directory (alternative to --data-config)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help='Class mapping for flat dir mode (e.g., "0:person,1:car")',
    )

    # Annotation config
    parser.add_argument(
        "--annotate-config",
        type=str,
        default=None,
        help="Path to annotation YAML config (default: configs/_shared/01_auto_annotate.yaml)",
    )

    # Processing options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "auto", "hybrid"],
        default=None,
        help="Annotation mode (default: from config)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["bbox", "polygon", "both"],
        default=None,
        help="Output format (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing label files",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["missing", "all"],
        default=None,
        help="Filter mode: 'missing' (unannotated only) or 'all'",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to process (e.g., train val test)",
    )
    parser.add_argument(
        "--text-prompts",
        type=str,
        default=None,
        help='Text prompt overrides (e.g., "person=a standing person,car=a vehicle")',
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

    # Validate input mode
    if args.data_config is None and args.image_dir is None:
        parser.error("Either --data-config or --image-dir must be provided")
    if args.image_dir is not None and args.classes is None:
        parser.error("--classes is required when using --image-dir")

    # Load data config (if YOLO layout mode)
    data_config = None
    data_config_path = None
    config_dir = "."
    dataset_name = "unknown"
    class_names: Dict[int, str] = {}

    if args.data_config:
        data_config_path = Path(args.data_config).resolve()
        data_config = load_config(data_config_path)
        config_dir = str(data_config_path.parent)
        dataset_name = data_config.get("dataset_name", "unknown")
        class_names = {int(k): v for k, v in data_config["names"].items()}
    else:
        dataset_name = Path(args.image_dir).name
        class_names = parse_classes(args.classes)

    # Load annotation config
    annotate_config_path = args.annotate_config
    if annotate_config_path is None:
        annotate_config_path = (
            Path(__file__).resolve().parent.parent.parent / "configs" / "_shared" / "01_auto_annotate.yaml"
        )
    else:
        annotate_config_path = Path(annotate_config_path).resolve()

    if not Path(annotate_config_path).exists():
        print(f"Error: Annotation config not found at {annotate_config_path}")
        sys.exit(1)

    annotate_config = load_config(annotate_config_path)

    # Service health checks
    auto_label_url = annotate_config.get("auto_label_service", {}).get("url", "http://localhost:18104")
    require_services(
        {"Auto-Label :18104": f"{auto_label_url}/health"},
        skip=args.skip_health_check,
    )

    # Apply overrides
    if args.override:
        overrides = parse_overrides(args.override)
        annotate_config = merge_configs(annotate_config, overrides)

    # Build text prompts: config defaults < data config < CLI overrides
    text_prompts: Dict[str, str] = dict(annotate_config.get("text_prompts", {}))
    if data_config and "text_prompts" in data_config:
        text_prompts.update(data_config["text_prompts"])
    if args.text_prompts:
        text_prompts.update(parse_text_prompts(args.text_prompts))

    # Determine processing options (CLI overrides config)
    processing = annotate_config.get("processing", {})
    mode = args.mode or processing.get("mode", "text")
    output_format = args.output_format or processing.get("output_format", "bbox")
    filter_mode = args.filter or processing.get("filter_mode", "missing")
    dry_run = args.dry_run or processing.get("dry_run", False)

    # Update splits in config if provided via CLI
    if args.splits:
        annotate_config.setdefault("auto_annotate", {})["splits"] = args.splits

    # Check for rule-based auto_label config
    auto_label_config = data_config.get("auto_label") if data_config else None
    detection_class_map = None
    if auto_label_config and "detection_classes" in auto_label_config:
        # Assign temp class IDs (100+) to intermediate detection classes
        detection_class_map = {
            name: 100 + i
            for i, name in enumerate(auto_label_config["detection_classes"].keys())
        }

    # Build initial state
    initial_state: Dict[str, Any] = {
        "data_config": data_config,
        "annotate_config": annotate_config,
        "dataset_name": dataset_name,
        "class_names": class_names,
        "text_prompts": text_prompts,
        "config_dir": config_dir,
        "image_paths": {},
        "total_images": 0,
        "current_batch_idx": 0,
        "total_batches": 0,
        "batch_size": processing.get("batch_size", 32),
        "image_results": [],
        "mode": mode,
        "output_format": output_format,
        "dry_run": dry_run,
        "filter_mode": filter_mode,
        "summary": {},
        "report_path": "",
        "auto_label_config": auto_label_config,
        "detection_class_map": detection_class_map,
    }

    # Handle flat dir mode: store image_dir in data_config-like structure
    if args.image_dir:
        initial_state["data_config"] = None
        # Scanner will use image_dir from annotate_config
        annotate_config["_image_dir"] = args.image_dir

    # Print header
    print("=" * 70)
    print(f"  Auto-Annotation: {dataset_name}")
    print("=" * 70)
    if args.data_config:
        print(f"  Data config   : {data_config_path}")
    else:
        print(f"  Image dir     : {args.image_dir}")
    print(f"  Classes       : {list(class_names.values())}")
    print(f"  Mode          : {mode}")
    print(f"  Output format : {output_format}")
    print(f"  Filter        : {filter_mode}")
    print(f"  Dry run       : {dry_run}")
    print()

    # Build and run pipeline
    start_time = time.time()

    if auto_label_config and detection_class_map:
        # Rule-based pipeline (functional API)
        from core.p01_auto_annotate.pipeline import auto_annotate_pipeline
        print("Running auto-annotation pipeline (rule-based)...")
        result = auto_annotate_pipeline.invoke(initial_state)
    else:
        # Standard pipeline (StateGraph)
        from core.p01_auto_annotate.graph import build_graph
        print("Running auto-annotation pipeline...")
        app = build_graph()
        result = app.invoke(initial_state)

    elapsed = time.time() - start_time

    # Print summary
    summary = result.get("summary", {})
    print()
    print("=" * 70)
    print(f"  Auto-Annotation Complete -- {dataset_name}")
    print("=" * 70)
    print(f"  Total images     : {summary.get('total_images', 0)}")
    print(f"  Total annotated  : {summary.get('total_annotated', 0)}")
    print(f"  Total detections : {summary.get('total_detections', 0)}")
    print(f"  Avg dets/image   : {summary.get('avg_detections_per_image', 0):.2f}")

    per_class = summary.get("per_class_counts", {})
    if per_class:
        print("  Per-class:")
        for cls_name, count in sorted(per_class.items(), key=lambda x: -x[1]):
            print(f"    {cls_name:<20}: {count}")

    timing = summary.get("timing", {})
    if timing:
        print(f"  Avg per sample   : {timing.get('avg_total_per_sample_s', 0):.4f}s")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Report saved to  : {result.get('report_path', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
