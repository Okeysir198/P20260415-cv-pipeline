"""HPO results saving: JSON, best config YAML, and visualization plots."""

import json
import sys
from pathlib import Path
from typing import Any

import kaleido  # noqa: F401
import optuna.visualization as vis
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from loguru import logger


class ResultsManager:
    """Save and visualize Optuna HPO results.

    Args:
        study: Completed Optuna study.
        best_config: Full training config with best hyperparameters.
        save_dir: Directory to save results.
    """

    def __init__(
        self,
        study: Any,  # optuna.Study
        best_config: dict | None,
        save_dir: str,
    ) -> None:
        self.study = study
        self.best_config = best_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_all(self) -> dict[str, Path]:
        """Save all results: JSON, best config, and plots.

        Returns:
            Dictionary mapping result type to file path.
        """
        saved = {}

        saved["study_json"] = self.save_study_json()

        if self.best_config is not None:
            saved["best_config"] = self.save_best_config()

        plot_paths = self.generate_plots()
        saved.update(plot_paths)

        logger.info("Results saved to %s", self.save_dir)
        for name, path in saved.items():
            logger.info("  %s: %s", name, path)

        return saved

    def save_study_json(self) -> Path:
        """Save study results as JSON.

        Includes: best trial info, all trials summary, study metadata.

        Returns:
            Path to the saved JSON file.
        """
        best_trial = self.study.best_trial

        # Build trials list
        trials_data = []
        for trial in self.study.trials:
            trial_info = {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "params": trial.params,
                "duration_seconds": (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
            }
            trials_data.append(trial_info)

        result = {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "n_trials": len(self.study.trials),
            "n_completed": len([t for t in self.study.trials if t.state.name == "COMPLETE"]),
            "n_pruned": len([t for t in self.study.trials if t.state.name == "PRUNED"]),
            "n_failed": len([t for t in self.study.trials if t.state.name == "FAIL"]),
            "best_trial": {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            },
            "trials": trials_data,
        }

        path = self.save_dir / "study_results.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info("Study results saved to %s", path)
        return path

    def save_best_config(self) -> Path:
        """Save the best training config as a YAML file.

        This config can be used directly with train.py --config.

        Returns:
            Path to the saved YAML file.
        """
        path = self.save_dir / "best_config.yaml"

        # Add a header comment
        header = (
            f"# Best training config from HPO study: {self.study.study_name}\n"
            f"# Best trial: {self.study.best_trial.number} "
            f"(value={self.study.best_trial.value:.4f})\n"
            f"# Use with: uv run core/p06_training/train.py --config {path}\n\n"
        )

        with open(path, "w") as f:
            f.write(header)
            yaml.dump(
                self.best_config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info("Best config saved to %s", path)
        return path

    def generate_plots(self) -> dict[str, Path]:
        """Generate visualization plots for the HPO study.

        Uses optuna.visualization (plotly) if available, otherwise skips.

        Returns:
            Dictionary mapping plot name to file path.
        """
        saved = {}

        # Only generate plots if we have enough trials
        completed = [t for t in self.study.trials if t.state.name == "COMPLETE"]
        if len(completed) < 2:
            logger.info("Not enough completed trials for plots (need >= 2).")
            return saved

        plots = {
            "optimization_history": vis.plot_optimization_history,
            "param_importances": vis.plot_param_importances,
            "parallel_coordinate": vis.plot_parallel_coordinate,
            "slice_plot": vis.plot_slice,
        }

        for name, plot_fn in plots.items():
            try:
                fig = plot_fn(self.study)

                html_path = self.save_dir / f"{name}.html"
                fig.write_html(str(html_path))
                saved[name] = html_path
                logger.info("Plot saved: %s", html_path)

                png_path = self.save_dir / f"{name}.png"
                fig.write_image(str(png_path), width=1200, height=600)
                saved[f"{name}_png"] = png_path
            except Exception as e:
                logger.warning("Failed to generate plot '%s': %s", name, e)

        return saved

    def print_summary(self) -> str:
        """Print a human-readable summary of HPO results.

        Returns:
            Summary string.
        """
        best = self.study.best_trial
        completed = [t for t in self.study.trials if t.state.name == "COMPLETE"]
        pruned = [t for t in self.study.trials if t.state.name == "PRUNED"]

        lines = [
            "",
            "=" * 60,
            "HPO Results Summary",
            "=" * 60,
            f"  Study: {self.study.study_name}",
            f"  Total trials: {len(self.study.trials)}",
            f"  Completed: {len(completed)}",
            f"  Pruned: {len(pruned)}",
            f"  Failed: {len(self.study.trials) - len(completed) - len(pruned)}",
            "",
            f"  Best trial: #{best.number}",
            f"  Best value: {best.value:.4f}",
            "",
            "  Best hyperparameters:",
        ]

        for name, value in sorted(best.params.items()):
            if isinstance(value, float):
                lines.append(f"    {name}: {value:.6f}")
            else:
                lines.append(f"    {name}: {value}")

        lines.append("")
        lines.append(f"  Results saved to: {self.save_dir}")

        if self.best_config is not None:
            best_config_path = self.save_dir / "best_config.yaml"
            lines.append(f"  Best config: {best_config_path}")
            lines.append(f"  Run full training: uv run core/p06_training/train.py --config {best_config_path}")

        lines.append("=" * 60)

        summary = "\n".join(lines)
        return summary
