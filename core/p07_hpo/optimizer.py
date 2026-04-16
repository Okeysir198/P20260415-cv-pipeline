"""Optuna HPO optimizer wrapping the detection model trainer."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import load_config, merge_configs
from core.p06_training.trainer import DetectionTrainer
from core.p07_hpo.search_space import SearchSpace
from core.p07_hpo.pruning_callback import OptunaPruningCallback

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner

logger = logging.getLogger(__name__)


class HPOOptimizer:
    """Optuna-based hyperparameter optimizer for detection model training.

    Each trial creates a trainer with sampled hyperparameters,
    runs a short training loop, and reports the best validation metric.

    Args:
        training_config_path: Path to the base training YAML config.
        hpo_config_path: Path to the HPO YAML config.
        training_overrides: Optional additional overrides for all trials.
    """

    def __init__(
        self,
        training_config_path: str,
        hpo_config_path: str = "configs/_shared/08_hyperparameter_tuning.yaml",
        training_overrides: Optional[dict] = None,
    ) -> None:
        self.training_config_path = Path(training_config_path)
        self.base_config = load_config(training_config_path)
        self.hpo_config = load_config(hpo_config_path)
        self.training_overrides = training_overrides or {}

        # Parse search space
        self.search_space = SearchSpace(self.hpo_config["search_space"])

        # Study settings
        self._study_cfg = self.hpo_config.get("study", {})
        self._trial_cfg = self.hpo_config.get("trial", {})
        self._pruning_cfg = self.hpo_config.get("pruning", {})

        # Track best
        self._best_trial = None
        self._study = None

    def _create_study(self, storage: Optional[str] = None) -> "optuna.Study":
        """Create an Optuna study with configured sampler and pruner.

        Args:
            storage: Optional Optuna storage URL (e.g. sqlite:///hpo.db).

        Returns:
            Configured Optuna Study.
        """
        # Sampler
        sampler_type = self._study_cfg.get("sampler", "tpe")
        seed = self._study_cfg.get("seed", 42)

        if sampler_type == "tpe":
            sampler = TPESampler(seed=seed)
        elif sampler_type == "random":
            sampler = RandomSampler(seed=seed)
        elif sampler_type == "cmaes":
            sampler = CmaEsSampler(seed=seed)
        else:
            logger.warning("Unknown sampler '%s', falling back to TPE.", sampler_type)
            sampler = TPESampler(seed=seed)

        # Pruner
        pruner_type = self._pruning_cfg.get("type", "median")
        n_startup = self._pruning_cfg.get("n_startup_trials", 5)
        n_warmup = self._pruning_cfg.get("n_warmup_epochs", 10)

        if pruner_type == "median":
            pruner = MedianPruner(
                n_startup_trials=n_startup,
                n_warmup_steps=n_warmup,
                interval_steps=self._pruning_cfg.get("interval_epochs", 1),
            )
        elif pruner_type == "percentile":
            pruner = PercentilePruner(
                percentile=self._pruning_cfg.get("percentile", 25.0),
                n_startup_trials=n_startup,
                n_warmup_steps=n_warmup,
            )
        elif pruner_type == "hyperband":
            pruner = HyperbandPruner(
                min_resource=self._pruning_cfg.get("min_resource", 3),
                max_resource=self._trial_cfg.get("epochs", 30),
                reduction_factor=self._pruning_cfg.get("reduction_factor", 3),
            )
        else:
            logger.warning("Unknown pruner '%s', falling back to MedianPruner.", pruner_type)
            pruner = MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup)

        # Dataset name for study name
        dataset_name = self.base_config.get("data", {}).get("dataset_config", "unknown")
        dataset_name = Path(dataset_name).stem  # e.g. "fire" from "features/safety-fire_detection/configs/05_data.yaml"
        study_name_template = self._study_cfg.get("name", "hpo_{dataset_name}")
        study_name = study_name_template.replace("{dataset_name}", dataset_name)

        direction = self._study_cfg.get("direction", "maximize")

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        logger.info(
            "Created study '%s' (direction=%s, sampler=%s, pruner=%s)",
            study_name, direction, sampler_type, pruner_type,
        )
        return study

    def _build_trial_config(self, trial: "optuna.Trial") -> dict:
        """Build the full training config for a single trial.

        Merges: base_config + training_overrides + trial_overrides + sampled_params

        Args:
            trial: Optuna trial for sampling.

        Returns:
            Complete training config dict.
        """
        # Sample hyperparameters
        sampled = self.search_space.sample(trial)

        # Trial-specific overrides (reduced epochs, no wandb, etc.)
        trial_overrides = {
            "training": {
                "epochs": self._trial_cfg.get("epochs", 30),
                "patience": self._trial_cfg.get("patience", 15),
            },
        }

        # Disable wandb for trials
        if self._trial_cfg.get("disable_wandb", True):
            trial_overrides["logging"] = {
                "wandb_project": None,
            }

        # Disable checkpoint saving for trials (save disk space)
        if not self._trial_cfg.get("save_checkpoints", False):
            trial_overrides["checkpoint"] = {
                "save_best": False,
                "save_interval": 0,
            }

        # Trial-specific save dir
        from utils.config import generate_run_dir

        from utils.config import feature_name_from_config_path
        trial_overrides["logging"]["save_dir"] = str(
            generate_run_dir(
                feature_name_from_config_path(self.training_config_path),
                f"08_hyperparameter_tuning_trial_{trial.number}",
            )
        )

        # Merge: base → trial overrides → sampled params → user overrides (CLI wins)
        config = merge_configs(self.base_config, trial_overrides)
        config = merge_configs(config, sampled)
        config = merge_configs(config, self.training_overrides)

        return config

    def objective(self, trial: "optuna.Trial") -> float:
        """Objective function for a single Optuna trial.

        Creates a DetectionTrainer with sampled hyperparameters, injects the
        pruning callback, and runs training. Returns the best metric value.

        Args:
            trial: Optuna trial.

        Returns:
            Best metric value from training.
        """
        trial_config = self._build_trial_config(trial)

        logger.info(
            "Trial %d: lr=%.6f, optimizer=%s, batch_size=%s",
            trial.number,
            trial_config["training"].get("lr", "?"),
            trial_config["training"].get("optimizer", "?"),
            trial_config["data"].get("batch_size", "?"),
        )

        try:
            # Create trainer with the trial config as overrides
            trainer = DetectionTrainer(
                config_path=str(self.training_config_path),
                overrides=trial_config,
            )

            # Monkey-patch _build_callbacks to inject pruning callback
            metric = self._pruning_cfg.get("metric",
                      trial_config.get("checkpoint", {}).get("metric", "val/mAP50"))
            warmup = self._pruning_cfg.get("n_warmup_epochs", 10)

            original_build_callbacks = trainer._build_callbacks

            def patched_build_callbacks():
                runner = original_build_callbacks()
                runner.add(OptunaPruningCallback(
                    trial=trial,
                    metric=metric,
                    warmup_epochs=warmup,
                ))
                return runner

            trainer._build_callbacks = patched_build_callbacks

            # Run training
            start_time = time.time()
            summary = trainer.train()
            elapsed = time.time() - start_time

            best_metric = summary.get("best_metric")
            if best_metric is None:
                best_metric = 0.0

            logger.info(
                "Trial %d finished: best_metric=%.4f, epochs=%d, time=%.1fs",
                trial.number, best_metric, summary["total_epochs"], elapsed,
            )

            return best_metric

        except optuna.TrialPruned:
            raise  # Re-raise for Optuna to handle

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "Trial %d: OOM error. Reporting worst value.",
                    trial.number,
                )
                import torch
                torch.cuda.empty_cache()
                return 0.0  # Worst value for maximize direction
            raise

        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e)
            return 0.0

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        storage: Optional[str] = None,
    ) -> "optuna.Study":
        """Run the full HPO optimization.

        Args:
            n_trials: Number of trials. Overrides config if provided.
            timeout: Timeout in seconds. Overrides config if provided.
            storage: Optuna storage URL.

        Returns:
            Completed Optuna Study.
        """
        self._study = self._create_study(storage)

        n_trials = n_trials or self._study_cfg.get("n_trials", 50)
        timeout = timeout or self._study_cfg.get("timeout")

        logger.info("Starting HPO: n_trials=%s, timeout=%s", n_trials, timeout)

        self._study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        self._best_trial = self._study.best_trial
        logger.info(
            "HPO complete. Best trial %d: value=%.4f, params=%s",
            self._best_trial.number,
            self._best_trial.value,
            self._best_trial.params,
        )

        return self._study

    @property
    def best_config(self) -> Optional[dict]:
        """Reconstruct the full training config with best trial parameters.

        Returns:
            Complete config dict with best hyperparameters, or None if
            no optimization has been run.
        """
        if self._study is None or len(self._study.trials) == 0:
            return None

        best_trial = self._study.best_trial

        # Reconstruct override dict from the best trial's params
        overrides = {}
        for name, value in best_trial.params.items():
            SearchSpace.set_nested(overrides, name, value)

        # Merge with base config (full epochs, wandb enabled, etc.)
        config = merge_configs(self.base_config, self.training_overrides)
        config = merge_configs(config, overrides)

        return config

    @property
    def study(self) -> Optional["optuna.Study"]:
        """The Optuna study, or None if optimize() hasn't been called."""
        return self._study
