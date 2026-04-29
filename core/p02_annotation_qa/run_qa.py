#!/usr/bin/env python3
"""Annotation QA CLI -- Check and improve dataset annotation quality.

Uses LangGraph + SAM3 to validate, verify, and score YOLO annotations.

Usage:
    python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml
    python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml --no-sam3
    python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml --resume runs/qa/fire/checkpoint.json
    python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml --override sampling.sample_size=500
"""

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from loguru import logger

from utils.config import (
    feature_name_from_config_path,
    generate_run_dir,
    load_config,
    merge_configs,
    parse_overrides,
)
from utils.service_health import require_services


def main() -> None:
    """Run the annotation QA pipeline."""
    parser = argparse.ArgumentParser(
        description="Annotation QA: Check and improve dataset annotation quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml
  python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml --no-sam3
  python run_qa.py --data-config features/ppe-shoes_detection/configs/05_data.yaml --override sampling.sample_size=500
  python run_qa.py --data-config features/safety-fire_detection/configs/05_data.yaml --resume runs/qa/fire/checkpoint.json
""",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to data YAML config (e.g., features/safety-fire_detection/configs/05_data.yaml)",
    )
    parser.add_argument(
        "--qa-config",
        type=str,
        default=None,
        help="Path to QA YAML config (default: configs/_shared/02_annotation_quality.yaml)",
    )
    parser.add_argument(
        "--no-sam3",
        action="store_true",
        help="Skip SAM3 verification (structural validation only)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides (e.g., sampling.sample_size=500)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip service health checks before starting",
    )
    parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="Apply suggested fixes to dataset labels (backs up originals first).",
    )

    args = parser.parse_args()

    # Load configs
    data_config_path = Path(args.data_config).resolve()
    data_config = load_config(data_config_path)
    config_dir = str(data_config_path.parent)

    # Load QA config
    qa_config_path = args.qa_config
    if qa_config_path is None:
        qa_config_path = (
            Path(__file__).resolve().parent.parent.parent / "configs" / "_shared" / "02_annotation_quality.yaml"
        )
    else:
        qa_config_path = Path(qa_config_path).resolve()

    if not qa_config_path.exists():
        print(f"Error: QA config not found at {qa_config_path}")
        sys.exit(1)

    qa_config = load_config(qa_config_path)

    # Service health checks
    if not args.no_sam3:
        sam3_url = qa_config.get("sam3", {}).get("service_url", "http://localhost:18100")
        qa_svc_url = qa_config.get("qa_service", {}).get("url", "http://localhost:18105")
        require_services(
            {
                "SAM3 :18100": f"{sam3_url}/health",
                "QA :18105": f"{qa_svc_url}/health",
            },
            skip=args.skip_health_check,
        )

    # Apply overrides
    if args.override:
        overrides = parse_overrides(args.override)
        qa_config = merge_configs(qa_config, overrides)

    # Build initial state
    class_names = {int(k): v for k, v in data_config["names"].items()}
    auto_label_config = data_config.get("auto_label") if data_config else None

    initial_state = {
        "data_config": data_config,
        "qa_config": qa_config,
        "dataset_name": data_config["dataset_name"],
        "class_names": class_names,
        "splits": qa_config.get("qa", {}).get("splits", ["train", "val"]),
        "config_dir": config_dir,
        "sampled_paths": {},
        "total_sampled": 0,
        "current_batch_idx": 0,
        "total_batches": 0,
        "batch_size": qa_config.get("processing", {}).get("batch_size", 32),
        "image_results": [],
        "summary": {},
        "report_path": "",
        "use_sam3": not args.no_sam3,
        "auto_label_config": auto_label_config,
    }

    # Handle resume
    if args.resume:
        from core.p02_annotation_qa.reporter import QAReporter

        reporter = QAReporter(
            str(generate_run_dir(feature_name_from_config_path(data_config_path), "02_annotation_quality")),
            data_config["dataset_name"],
            qa_config.get("reporting", {}),
        )
        checkpoint = reporter.load_checkpoint(args.resume)
        initial_state.update(checkpoint)
        print(
            f"Resumed from checkpoint: batch "
            f"{initial_state['current_batch_idx']}/{initial_state['total_batches']}"
        )

    # Print header
    print("=" * 70)
    print(f"  Annotation QA: {data_config['dataset_name']}")
    print("=" * 70)
    print(f"  Data config : {data_config_path}")
    print(f"  QA config   : {qa_config_path}")
    print(f"  Classes     : {list(class_names.values())}")
    print(f"  SAM3        : {'enabled' if not args.no_sam3 else 'disabled'}")
    print(f"  Splits      : {initial_state['splits']}")
    print()

    # Build and run graph
    start_time = time.time()

    from core.p02_annotation_qa.pipeline import qa_pipeline

    print("Running annotation QA pipeline...")
    result = qa_pipeline.invoke(initial_state)

    elapsed = time.time() - start_time

    # Print summary
    summary = result.get("summary", {})
    print()
    print("=" * 70)
    print(f"  QA Complete -- {data_config['dataset_name']}")
    print("=" * 70)
    print(f"  Total checked    : {summary.get('total_checked', 0)}")
    print(f"  Avg quality score: {summary.get('avg_quality_score', 0):.3f}")
    grades = summary.get("grades", {})
    total = sum(grades.values()) or 1
    for grade in ["good", "review", "bad", "unverified"]:
        n = grades.get(grade, 0)
        pct = n / total * 100
        print(f"  {grade:<10s}: {n:>6d} ({pct:5.1f}%)")
    timing = summary.get("timing", {})
    if timing:
        print(f"  Avg per sample   : {timing.get('avg_total_per_sample_s', 0):.4f}s "
              f"(validate={timing.get('avg_validate_s', 0):.4f}s, "
              f"sam3={timing.get('avg_sam3_verify_s', 0):.4f}s)")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Report saved to  : {result.get('report_path', 'N/A')}")
    print("=" * 70)

    # Append QA results to DATASET_REPORT.md in training_ready/
    _append_qa_to_dataset_report(
        data_config=data_config,
        data_config_path=data_config_path,
        summary=summary,
        report_dir=result.get("report_path", ""),
    )

    # Apply fixes if requested
    if args.apply_fixes:
        report_dir = result.get("report_path", "")
        if report_dir:
            apply_fixes(report_dir)
        else:
            print("Warning: No report_path in result state, cannot apply fixes.")


def _append_qa_to_dataset_report(
    data_config: dict,
    data_config_path: Path,
    summary: dict,
    report_dir: str,
) -> None:
    """Append a Label Quality section to the DATASET_REPORT.md in training_ready/.

    Finds DATASET_REPORT.md via data_config["path"] resolved relative to the
    config file's directory. Skips silently if the report doesn't exist yet.
    """
    dataset_path = (data_config_path.parent / data_config.get("path", "")).resolve()
    report_md = dataset_path / "DATASET_REPORT.md"

    if not report_md.is_file():
        return

    total = summary.get("total_checked", 0)
    avg_score = summary.get("avg_quality_score", 0.0)
    grades = summary.get("grades", {})
    issues = summary.get("issue_breakdown", {})
    good_n = grades.get("good", 0)
    review_n = grades.get("review", 0)
    bad_n = grades.get("bad", 0)
    unverified_n = grades.get("unverified", 0)
    good_pct = (good_n / total * 100) if total else 0.0
    review_pct = (review_n / total * 100) if total else 0.0
    bad_pct = (bad_n / total * 100) if total else 0.0
    unverified_pct = (unverified_n / total * 100) if total else 0.0

    # Verdict logic (mirrors skill decision table)
    if good_pct >= 80 and bad_pct <= 5:
        verdict = "✅ ACCEPT — good ≥ 80%, bad ≤ 5% → proceed to training"
    elif unverified_pct > 30:
        verdict = "🔄 RERUN_QA — SAM3 was unreliable (>30% unverified) → re-run QA when SAM3 is stable"
    elif bad_pct <= 20:
        verdict = "🔄 RE-LABEL — bad 5–20% → run p01 auto-relabel then re-QA"
    else:
        verdict = "🛑 STOP — bad > 20% → almost certainly a class_map bug, not bad labels"

    run_ref = str(Path(report_dir).relative_to(Path(report_dir).parents[2])) if report_dir else "N/A"

    lines = [
        "",
        "---",
        "",
        "## Label Quality (Annotation QA)",
        "",
        f"**Sampled:** {total} images &nbsp;|&nbsp; "
        f"**Avg quality score:** {avg_score:.3f} &nbsp;|&nbsp; "
        f"**Run:** `{run_ref}`",
        "",
        "| Grade | Count | % | Action |",
        "|---|---|---|---|",
        f"| ✅ good       | {good_n} | {good_pct:.1f}% | trusted — goes straight to training |",
        f"| 👁 review     | {review_n} | {review_pct:.1f}% | import to Label Studio for human check |",
        f"| ❌ bad        | {bad_n} | {bad_pct:.1f}% | re-label with p01 or discard |",
        f"| ❓ unverified | {unverified_n} | {unverified_pct:.1f}% | SAM3 unavailable — re-run QA when SAM3 is stable |",
        "",
        f"**Verdict:** {verdict}",
    ]

    if issues:
        top_issues = sorted(issues.items(), key=lambda x: -x[1])[:5]
        lines += ["", "**Top issues:**"]
        for issue_type, count in top_issues:
            lines.append(f"- `{issue_type}`: {count}")

    lines.append("")

    existing = report_md.read_text(encoding="utf-8")
    # Remove any previous QA section before appending fresh results
    qa_marker = "\n---\n\n## Label Quality (Annotation QA)"
    if qa_marker in existing:
        existing = existing[: existing.index(qa_marker)]

    report_md.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    print("  ↳ Updated DATASET_REPORT.md with label quality section")


def apply_fixes(report_dir: str) -> None:
    """Read fixes.json and apply corrections to dataset label files.

    For each label file with fixes:
        - ``clip_bbox``: replace the original line with the suggested line.
        - ``remove_duplicate`` / ``remove_degenerate``: delete the line.

    Before writing any changes the entire labels directory is backed up once
    to a sibling ``labels_backup_{timestamp}/`` directory.

    Args:
        report_dir: Path to the QA report directory containing ``fixes.json``.
    """
    fixes_path = Path(report_dir) / "fixes.json"
    if not fixes_path.is_file():
        print(f"No fixes.json found at {fixes_path}")
        return

    data = json.loads(fixes_path.read_text(encoding="utf-8"))
    fixes = data.get("fixes", [])

    if not fixes:
        print("No fixes to apply.")
        return

    # Group fixes by label_path
    fixes_by_label: dict[str, list[dict]] = defaultdict(list)
    for fix in fixes:
        label_path = fix.get("label_path", "")
        if label_path:
            fixes_by_label[label_path].append(fix)

    if not fixes_by_label:
        print("No fixes to apply (no valid label paths).")
        return

    # Back up labels directories ONCE before writing any changes.
    # Collect the unique labels directories that will be modified.
    backed_up_dirs: set[str] = set()
    backup_dir_map: dict[str, str] = {}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    for label_path in fixes_by_label:
        labels_dir = str(Path(label_path).parent)
        if labels_dir not in backed_up_dirs:
            labels_dir_path = Path(labels_dir)
            if labels_dir_path.is_dir():
                backup_name = f"{labels_dir_path.name}_backup_{timestamp}"
                backup_path = labels_dir_path.parent / backup_name
                print(f"  Backing up {labels_dir} -> {backup_path}")
                shutil.copytree(labels_dir_path, backup_path)
                backup_dir_map[labels_dir] = str(backup_path)
            backed_up_dirs.add(labels_dir)

    # Apply fixes per label file
    total_applied = 0
    files_modified = 0

    for label_path, file_fixes in fixes_by_label.items():
        lp = Path(label_path)
        if not lp.is_file():
            logger.warning("Label file not found, skipping: %s", label_path)
            continue

        lines = lp.read_text(encoding="utf-8").splitlines()

        # Sort fixes by annotation_idx in REVERSE order so that removals
        # do not shift the indices of subsequent fixes.
        file_fixes_sorted = sorted(
            file_fixes,
            key=lambda f: f.get("annotation_idx", 0),
            reverse=True,
        )

        for fix in file_fixes_sorted:
            idx = fix.get("annotation_idx")
            fix_type = fix.get("fix_type", "")

            if idx is None or idx < 0 or idx >= len(lines):
                logger.warning(
                    "Invalid annotation_idx %s for %s (file has %d lines), skipping",
                    idx, label_path, len(lines),
                )
                continue

            if fix_type == "clip_bbox":
                suggested = fix.get("suggested", "")
                if suggested:
                    # suggested may be a dict {class_id, bbox_norm} or a plain YOLO string
                    if isinstance(suggested, dict):
                        cid = suggested["class_id"]
                        bx = suggested["bbox_norm"]
                        lines[idx] = f"{cid} {bx[0]:.6f} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f}"
                    else:
                        lines[idx] = str(suggested).strip()
                    total_applied += 1
            elif fix_type in ("remove_duplicate", "remove_degenerate"):
                lines.pop(idx)
                total_applied += 1
            else:
                logger.warning("Unknown fix_type '%s', skipping", fix_type)

        lp.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        files_modified += 1

    # Summary
    backup_dirs_str = ", ".join(backup_dir_map.values()) if backup_dir_map else "N/A"
    print(
        f"Applied {total_applied} fixes to {files_modified} label files. "
        f"Originals backed up to {backup_dirs_str}"
    )


if __name__ == "__main__":
    main()
