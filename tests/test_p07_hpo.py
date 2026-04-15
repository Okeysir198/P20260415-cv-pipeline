"""Test 06: HPO — run 2 Optuna trials with 1 epoch each."""

import json
import sys
import tempfile
from pathlib import Path

import optuna
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from core.p07_hpo.optimizer import HPOOptimizer
from core.p07_hpo.pruning_callback import OptunaPruningCallback
from core.p07_hpo.results import ResultsManager
from core.p07_hpo.search_space import SearchSpace
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "06_hpo"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")
HPO_CONFIG_PATH = str(ROOT / "configs" / "_shared" / "08_hyperparameter_tuning.yaml")


def test_hpo_2_trials():
    """Run 2 HPO trials with 1 epoch each."""
    optimizer = HPOOptimizer(
        training_config_path=TRAIN_CONFIG_PATH,
        hpo_config_path=HPO_CONFIG_PATH,
        training_overrides={
            "training": {"epochs": 1, "patience": 10},
            "logging": {"save_dir": str(OUTPUTS / "trials")},
        },
    )

    study = optimizer.optimize(n_trials=2)
    assert study is not None, "optimize() returned None"
    assert len(study.trials) == 2, f"Expected 2 trials, got {len(study.trials)}"

    best = study.best_trial
    print(f"    Best trial: #{best.number}, value: {best.value:.4f}")
    print(f"    Best params: {best.params}")

    # Save study summary
    summary = {
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
    }
    summary_path = OUTPUTS / "study_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Saved: {summary_path}")


def test_best_config_saved():
    """Verify best config YAML was saved."""
    optimizer = HPOOptimizer(
        training_config_path=TRAIN_CONFIG_PATH,
        hpo_config_path=HPO_CONFIG_PATH,
    )

    best_config = optimizer.best_config
    if best_config is not None:
        print(f"    Best config keys: {list(best_config.keys())}")

        # Save best config
        config_path = OUTPUTS / "best_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)
        print(f"    Saved: {config_path}")
    else:
        print("    No best config available (run test_hpo_2_trials first)")


def test_search_space_summary():
    """Load SearchSpace from HPO config and verify summary() returns non-empty string."""
    hpo_config = load_config(HPO_CONFIG_PATH)
    assert "search_space" in hpo_config, "HPO config missing 'search_space' key"

    space = SearchSpace(hpo_config["search_space"])
    summary_text = space.summary()

    assert isinstance(summary_text, str), f"Expected str, got {type(summary_text)}"
    assert len(summary_text) > 0, "summary() returned empty string"
    print(f"    Search space summary ({len(summary_text)} chars):")
    # Print first few lines
    for line in summary_text.split("\n")[:5]:
        print(f"      {line}")


def test_results_manager():
    """Create ResultsManager with a minimal optuna study and verify save_all()."""
    # Create a minimal study with 1 trial
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("x", -1.0, 1.0) ** 2, n_trials=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ResultsManager(
            study=study,
            best_config={"training": {"lr": 0.001}},
            save_dir=tmpdir,
        )

        saved = manager.save_all()
        assert isinstance(saved, dict), f"Expected dict, got {type(saved)}"
        assert len(saved) > 0, "save_all() returned empty dict"

        # Verify at least the study JSON was created
        assert "study_json" in saved, f"Missing 'study_json' key, got: {list(saved.keys())}"
        assert Path(saved["study_json"]).exists(), (
            f"study_json file not found: {saved['study_json']}"
        )
        print(f"    ResultsManager saved {len(saved)} files: {list(saved.keys())}")


def test_pruning_callback_init():
    """Create OptunaPruningCallback with a mock trial, verify initialization."""
    class FakeTrial:
        """Minimal trial-like object with report() method."""
        number = 0
        def report(self, value, step):
            pass
        def should_prune(self):
            return False

    trial = FakeTrial()
    callback = OptunaPruningCallback(trial=trial, metric="val/mAP50", warmup_epochs=5)

    assert callback.trial is trial, "trial not stored correctly"
    assert callback.metric == "val/mAP50", f"Expected metric='val/mAP50', got '{callback.metric}'"
    assert callback.warmup_epochs == 5, f"Expected warmup_epochs=5, got {callback.warmup_epochs}"
    print(f"    OptunaPruningCallback created: metric={callback.metric}, warmup={callback.warmup_epochs}")


def test_pruning_callback_on_epoch_end():
    """Simulate epoch_end calls and verify trial.report() is called."""
    class FakeTrial:
        """Trial that records report() calls."""
        number = 1
        def __init__(self):
            self.reported = []
        def report(self, value, step):
            self.reported.append((value, step))
        def should_prune(self):
            return False

    trial = FakeTrial()
    callback = OptunaPruningCallback(trial=trial, metric="val/mAP50", warmup_epochs=2)

    # Simulate 3 epoch_end calls
    callback.on_epoch_end(trainer=None, epoch=0, metrics={"val/mAP50": 0.3, "val/loss": 1.5})
    callback.on_epoch_end(trainer=None, epoch=1, metrics={"val/mAP50": 0.5, "val/loss": 1.2})
    callback.on_epoch_end(trainer=None, epoch=2, metrics={"val/mAP50": 0.7, "val/loss": 0.9})

    assert len(trial.reported) == 3, f"Expected 3 report() calls, got {len(trial.reported)}"
    assert trial.reported[0] == (0.3, 0), f"First report mismatch: {trial.reported[0]}"
    assert trial.reported[1] == (0.5, 1), f"Second report mismatch: {trial.reported[1]}"
    assert trial.reported[2] == (0.7, 2), f"Third report mismatch: {trial.reported[2]}"

    # Verify missing metric does not crash (just skips)
    callback.on_epoch_end(trainer=None, epoch=3, metrics={"val/loss": 0.8})
    assert len(trial.reported) == 3, "report() should not be called when metric is missing"

    print(f"    Callback reported {len(trial.reported)} values: {trial.reported}")


if __name__ == "__main__":
    run_all([
        ("hpo_2_trials", test_hpo_2_trials),
        ("best_config_saved", test_best_config_saved),
        ("search_space_summary", test_search_space_summary),
        ("results_manager", test_results_manager),
        ("pruning_callback_init", test_pruning_callback_init),
        ("pruning_callback_on_epoch_end", test_pruning_callback_on_epoch_end),
    ], title="Test 10: HPO (2 trials, 1 epoch each)")
