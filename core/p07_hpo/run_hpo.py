"""CLI entrypoint for Optuna hyperparameter optimization.

Usage:
    python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml
    python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml --n-trials 100 --trial-epochs 50
    python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

from loguru import logger  # noqa: E402
from utils.config import parse_overrides  # noqa: E402


def main() -> None:
    """Parse CLI arguments and launch HPO."""
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for object detection training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default HPO (50 trials, 30 epochs each)\n"
            "  python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml\n\n"
            "  # Quick smoke test\n"
            "  python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml --n-trials 2 --trial-epochs 1\n\n"
            "  # Dry run (print search space only)\n"
            "  python run_hpo.py --config features/safety-fire_detection/configs/06_training.yaml --dry-run\n\n"
            "  # Use best config for full training\n"
            "  python core/p06_training/train.py --config runs/hpo/fire/best_config.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base training YAML config (e.g. features/safety-fire_detection/configs/06_training.yaml).",
    )
    parser.add_argument(
        "--hpo-config",
        type=str,
        default="configs/_shared/08_hyperparameter_tuning.yaml",
        help="Path to the HPO config (default: configs/_shared/08_hyperparameter_tuning.yaml).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials. Overrides HPO config value.",
    )
    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=None,
        help="Epochs per trial. Overrides HPO config value.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Total timeout in seconds. Overrides HPO config value.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///hpo.db) for persistent studies.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Training config overrides as key=value pairs (applied to all trials).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a specific device (e.g. 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print search space summary and exit without running trials.",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Validate config files
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Training config not found: %s", config_path)
        sys.exit(1)

    hpo_config_path = Path(args.hpo_config)
    if not hpo_config_path.exists():
        logger.error("HPO config not found: %s", hpo_config_path)
        sys.exit(1)

    # Parse overrides
    overrides = {}
    if args.override:
        try:
            overrides = parse_overrides(args.override)
            logger.info("Training overrides: %s", overrides)
        except ValueError as e:
            logger.error("Invalid override: %s", e)
            sys.exit(1)

    if args.device:
        overrides["device"] = args.device

    # Import HPO modules (after optuna check)
    from core.p07_hpo.optimizer import HPOOptimizer
    from core.p07_hpo.results import ResultsManager

    # Handle trial-epochs override
    if args.trial_epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.trial_epochs

    # Create optimizer
    optimizer = HPOOptimizer(
        training_config_path=str(config_path),
        hpo_config_path=str(hpo_config_path),
        training_overrides=overrides or None,
    )

    # Dry run mode
    if args.dry_run:
        print(optimizer.search_space.summary())
        print(f"\nBase config: {config_path}")
        print(f"HPO config: {hpo_config_path}")
        print(f"Trials: {args.n_trials or optimizer._study_cfg.get('n_trials', 50)}")
        print(f"Epochs/trial: {args.trial_epochs or optimizer._trial_cfg.get('epochs', 30)}")
        sys.exit(0)

    # Run optimization
    try:
        study = optimizer.optimize(
            n_trials=args.n_trials,
            timeout=args.timeout,
            storage=args.storage,
        )
    except KeyboardInterrupt:
        logger.info("HPO interrupted by user.")
        study = optimizer.study
        if study is None or len(study.trials) == 0:
            logger.info("No completed trials. Exiting.")
            sys.exit(0)

    # Save results
    # Determine save dir — auto-generate timestamped path
    from utils.config import feature_name_from_config_path, generate_run_dir
    save_dir = str(generate_run_dir(
        feature_name_from_config_path(args.config), "08_hyperparameter_tuning"
    ))

    results = ResultsManager(
        study=study,
        best_config=optimizer.best_config,
        save_dir=save_dir,
    )

    results.save_all()
    print(results.print_summary())


if __name__ == "__main__":
    main()
