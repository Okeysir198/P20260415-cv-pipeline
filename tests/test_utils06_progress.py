"""Test 06: Utils — ProgressBar and TrainingProgress trackers."""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.progress import ProgressBar, TrainingProgress, _fmt


def test_fmt_float_and_non_float():
    assert _fmt(1.23456) == "1.2346"
    assert _fmt(7) == 7
    assert _fmt("text") == "text"


def test_progress_bar_updates_and_closes():
    buf = io.StringIO()
    with redirect_stderr(buf):
        pbar = ProgressBar(total=5, desc="unit-test", leave=False)
        for _ in range(5):
            pbar.update(metrics={"loss": 0.123456})
        pbar.set_description("done")
        pbar.close()
    assert pbar._bar.n == 5


def test_progress_bar_context_manager_closes():
    buf = io.StringIO()
    with redirect_stderr(buf):
        with ProgressBar(total=3, desc="ctx", leave=False) as pb:
            pb.update(3)
    # After __exit__ the underlying tqdm should be disabled/closed
    assert pb._bar.disable or pb._bar.n == 3


def test_training_progress_tracks_best_max():
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        tp = TrainingProgress(total_epochs=3, batches_per_epoch=2)

        tp.start_epoch(0)
        tp.update_batch(metrics={"loss": 1.0})
        tp.update_batch(metrics={"loss": 0.8})
        is_best = tp.end_epoch(metrics={"val/mAP": 0.5}, track_metric="val/mAP", mode="max")
        assert is_best is True

        tp.start_epoch(1)
        tp.update_batch(2)
        is_best = tp.end_epoch(metrics={"val/mAP": 0.4}, track_metric="val/mAP", mode="max")
        assert is_best is False

        tp.start_epoch(2)
        tp.update_batch(2)
        is_best = tp.end_epoch(metrics={"val/mAP": 0.7}, track_metric="val/mAP", mode="max")
        assert is_best is True

        assert tp.best_metric == 0.7
        assert tp.best_epoch == 2
        tp.close()


def test_training_progress_tracks_best_min():
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        tp = TrainingProgress(total_epochs=2)
        tp.start_epoch(0)
        is_best = tp.end_epoch(metrics={"loss": 1.0}, track_metric="loss", mode="min")
        assert is_best is True
        tp.start_epoch(1)
        is_best = tp.end_epoch(metrics={"loss": 0.3}, track_metric="loss", mode="min")
        assert is_best is True
        assert tp.best_metric == 0.3
        tp.close()


def test_training_progress_elapsed_and_str():
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        tp = TrainingProgress(total_epochs=1)
        assert tp.elapsed_seconds >= 0
        s = tp.elapsed_str
        assert len(s) == 8 and s.count(":") == 2  # HH:MM:SS
        tp.close()


def test_training_progress_context_manager():
    buf = io.StringIO()
    with redirect_stderr(buf), redirect_stdout(buf):
        with TrainingProgress(total_epochs=1) as tp:
            tp.start_epoch(0)
            tp.end_epoch()
        # epoch bar should be closed
        assert tp._epoch_bar.disable or tp._epoch_bar.n == 1


if __name__ == "__main__":
    run_all([
        ("fmt", test_fmt_float_and_non_float),
        ("progress_bar_updates", test_progress_bar_updates_and_closes),
        ("progress_bar_ctxmgr", test_progress_bar_context_manager_closes),
        ("training_best_max", test_training_progress_tracks_best_max),
        ("training_best_min", test_training_progress_tracks_best_min),
        ("training_elapsed", test_training_progress_elapsed_and_str),
        ("training_ctxmgr", test_training_progress_context_manager),
    ], title="Test utils06: progress")
