"""Promote a training run to a versioned release.

Usage:
    uv run utils/release.py --run-dir runs/fire_detection/2026-03-26_1430_06_training
    uv run utils/release.py --run-dir runs/fire_detection/2026-03-26_1430_06_training --onnx models/fire_yolox_m_640.onnx

The use case name is derived from the run directory's config (dataset_name field).
Version auto-increments (v1, v2, v3...). Add --onnx <path> for ONNX file, --notes "text" for release notes.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ai/ root

import yaml

from utils.config import load_config

logger = logging.getLogger(__name__)


def _detect_use_case(run_dir: Path) -> str:
    """Detect use case name from the run directory's saved configs.

    Checks (in order):
    1. 05_data.yaml in run dir → dataset_name field
    2. 06_training.yaml in run dir → data.dataset_config path
    3. Run directory parent name (e.g. runs/fire_detection/... → fire_detection)
    """
    # From saved data config
    data_config_path = run_dir / "05_data.yaml"
    if data_config_path.exists():
        data_config = load_config(str(data_config_path))
        name = data_config.get("dataset_name")
        if name:
            return name

    # From saved training config
    training_config_path = run_dir / "06_training.yaml"
    if training_config_path.exists():
        training_config = load_config(str(training_config_path))
        dataset_ref = training_config.get("data", {}).get("dataset_config", "")
        if dataset_ref:
            # e.g. "05_data.yaml" from features/safety-fire_detection/configs/ → parent dir name
            return Path(dataset_ref).parent.name

    # Fallback: parent directory name (e.g. runs/fire_detection/timestamp → fire_detection)
    return run_dir.parent.name


def _dvc_tag(use_case: str) -> str:
    """Find latest DVC data tag for this use case."""
    result = subprocess.run(
        ["git", "tag", "--list", f"{use_case}-data-*", "--sort=-v:refname"],
        capture_output=True,
        text=True,
    )
    tags = result.stdout.strip().split("\n")
    return tags[0] if tags and tags[0] else "untagged"


def _next_version(releases_dir: Path, use_case: str) -> int:
    """Auto-detect next version number from existing releases."""
    use_case_dir = releases_dir / use_case
    if not use_case_dir.exists():
        return 1
    existing = sorted(use_case_dir.iterdir())
    versions = []
    for d in existing:
        if d.is_dir() and d.name.startswith("v"):
            try:
                v = int(d.name.split("_")[0][1:])
                versions.append(v)
            except (ValueError, IndexError):
                pass
    return max(versions) + 1 if versions else 1


def _load_metrics(run_dir: Path) -> dict:
    """Load metrics from a training run.

    Looks in the run_dir itself first, then in the feature's canonical eval dir
    (``features/<feature>/eval/metrics.json``) — where
    ``core/p08_evaluation/evaluate.py`` writes them.
    """
    import json

    feat_eval = run_dir.parent.parent / "eval"
    candidates = [
        run_dir / "metrics.json",
        run_dir / "best_metrics.json",
        run_dir / "eval_results.json",
        feat_eval / "metrics.json",
        feat_eval / "eval_results.json",
    ]
    for p in candidates:
        try:
            return json.loads(p.read_text())
        except FileNotFoundError:
            continue
    return {}


def release(
    run_dir: Path,
    use_case: str | None = None,
    onnx_path: Path | None = None,
    version: int | None = None,
    releases_dir: Path = Path("releases"),
    notes: str = "",
) -> Path:
    """Promote a training run to a versioned release.

    Args:
        run_dir: Path to the training run directory.
        use_case: Use case name. Auto-detected from run dir configs if None.
        onnx_path: Path to exported ONNX file (optional).
        version: Explicit version number (auto-detected if None).
        releases_dir: Root releases directory.
        notes: Optional release notes.

    Returns:
        Path to the created release directory.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run not found: {run_dir}")

    # Detect use case from run directory configs
    if use_case is None:
        use_case = _detect_use_case(run_dir)
    logger.info("Use case: %s", use_case)

    # Determine version
    if version is None:
        version = _next_version(releases_dir, use_case)

    date_str = datetime.now().strftime("%Y-%m-%d")
    release_name = f"v{version}_{date_str}"
    release_dir = releases_dir / use_case / release_name
    release_dir.mkdir(parents=True, exist_ok=True)

    # Copy model weights
    for weight_name in ["best.pth", "best.pt", "last.pth"]:
        src = run_dir / weight_name
        if src.exists():
            shutil.copy2(src, release_dir / weight_name)
            logger.info("Copied %s", weight_name)

    # Copy ONNX if provided
    if onnx_path and onnx_path.exists():
        shutil.copy2(onnx_path, release_dir / onnx_path.name)
        logger.info("Copied ONNX: %s", onnx_path.name)

    # Freeze configs from run directory (saved by CheckpointSaver.on_train_start)
    for config_name in ["05_data.yaml", "06_training.yaml", "config_resolved.yaml"]:
        src = run_dir / config_name
        if src.exists():
            shutil.copy2(src, release_dir / config_name)
            logger.info("Frozen config: %s", config_name)

    # Load metrics
    metrics = _load_metrics(run_dir)

    # Build model card
    dataset_tag = _dvc_tag(use_case)

    model_card = {
        "model": {
            "use_case": use_case,
            "version": version,
            "released": date_str,
            "notes": notes or "",
        },
        "dataset": {
            "dvc_tag": dataset_tag,
        },
        "source": {
            "training_run": str(run_dir),
            "onnx_source": str(onnx_path) if onnx_path else "",
        },
        "metrics": metrics,
    }

    # Load architecture info from saved training config
    training_config_path = run_dir / "06_training.yaml"
    if training_config_path.exists():
        training_config = load_config(str(training_config_path))
        model_cfg = training_config.get("model", {})
        model_card["architecture"] = {
            "arch": model_cfg.get("arch", ""),
            "num_classes": model_cfg.get("num_classes", 0),
            "input_size": model_cfg.get("input_size", []),
        }

    # Write model card
    model_card_path = release_dir / "model_card.yaml"
    with open(model_card_path, "w") as f:
        yaml.dump(model_card, f, default_flow_style=False, sort_keys=False)

    logger.info("Release created: %s", release_dir)
    return release_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Promote a training run to a versioned release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the training run directory (e.g. runs/fire_detection/2026-03-26_1430).",
    )
    parser.add_argument(
        "--use-case",
        default=None,
        help="Use case name (auto-detected from run dir configs if not set).",
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="Path to exported ONNX file (optional).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Explicit version number. Auto-increments if not set.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Release notes.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    release_dir = release(
        run_dir=Path(args.run_dir),
        use_case=args.use_case,
        onnx_path=Path(args.onnx) if args.onnx else None,
        version=args.version,
        notes=args.notes,
    )

    print(f"\nRelease created: {release_dir}")
    print(f"Model card:      {release_dir / 'model_card.yaml'}")


if __name__ == "__main__":
    main()
