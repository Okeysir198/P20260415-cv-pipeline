"""PaddlePaddle backend runner — runs **inside** ``.venv-paddle/``.

Invoked by :mod:`core.p06_training.paddle_trainer` via subprocess. Reads a
JSON config payload from stdin + small flags from argv, drives the upstream
PaddleDetection / PaddleClas / PaddleSeg ``Trainer`` to convergence, and
emits a sentinel-delimited JSON summary on stdout for the wrapper to parse.

The wrapper has already:
- merged ``--override`` flags into the YAML config
- resolved ``save_dir`` via :func:`utils.config.generate_run_dir`
- verified ``.venv-paddle/`` exists

This file therefore only worries about *building* the upstream paddle
trainer from our YAML keys and shipping the artifact tree.

NEVER import this module from main-venv code — paddle is not installed
there. The wrapper subprocess'es into ``.venv-paddle/bin/python -m
core.p06_training._paddle_runner``.

Sentinels: stdout MUST end with::

    ___PADDLE_SUMMARY_JSON___
    {"metrics": ..., "best_metric": ..., "total_epochs": ..., ...}
    ___END___

so the wrapper can recover the summary even when paddle's logger prints
arbitrary text on the same stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

_SUMMARY_BEGIN = "___PADDLE_SUMMARY_JSON___"
_SUMMARY_END = "___END___"


def _emit_summary(summary: dict[str, Any]) -> None:
    """Print the sentinel-delimited summary block on stdout (and flush)."""
    sys.stdout.write("\n")
    sys.stdout.write(_SUMMARY_BEGIN)
    sys.stdout.write("\n")
    sys.stdout.write(json.dumps(summary, default=str))
    sys.stdout.write("\n")
    sys.stdout.write(_SUMMARY_END)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paddle training subprocess runner")
    parser.add_argument("--save-dir", required=True, type=Path,
                        help="Pre-resolved run directory (created by wrapper)")
    parser.add_argument("--config-path", required=True, type=Path,
                        help="Original training YAML path (for lineage / config copy)")
    parser.add_argument("--feature-name", default="unknown",
                        help="Resolved feature name (e.g. safety-fire_detection)")
    parser.add_argument("--resume-from", default=None,
                        help="Optional path to a paddle checkpoint to resume from")
    return parser.parse_args()


def _read_payload() -> dict[str, Any]:
    """Read the JSON payload (config + extras) the wrapper piped on stdin."""
    raw = sys.stdin.read()
    if not raw:
        raise RuntimeError("Empty stdin payload — wrapper did not send config")
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Upstream paddle dispatcher (placeholder until Units 4-7 land the registry)
# ---------------------------------------------------------------------------

def _build_and_train(config: dict, save_dir: Path, resume_from: str | None) -> dict[str, Any]:
    """Build the upstream paddle trainer and run it to convergence.

    The actual model construction goes through ``core.p06_models``' paddle
    registry (added by Units 4-7) — until those land, this function imports
    the registry lazily and lets ``ImportError`` propagate up so the
    wrapper sees a clear failure.

    Returns the summary dict shape that ``train_with_paddle`` exposes:
    ``{metrics, best_metric, best_epoch, total_epochs, output_dir}``.
    """
    # Lazy import — only resolvable inside .venv-paddle/.
    import paddle  # noqa: F401  (imported for side-effects + readiness check)

    from core.p06_models import build_model  # registry dispatcher

    # `build_model` returns a paddle-aware adapter for the paddle archs
    # registered by Units 4-7 (e.g. picodet-s, ppyoloe-s, ppliteseg, …).
    # The adapter exposes the upstream trainer + a `train()` method that
    # mirrors `train_with_hf`'s summary shape.
    paddle_adapter = build_model(config)
    if not hasattr(paddle_adapter, "train_paddle"):
        raise RuntimeError(
            "build_model() returned an adapter without a `train_paddle()` "
            "method — the paddle model registry (Units 4-7) is not yet wired "
            "for this arch. config['model']['arch'] = "
            f"{config.get('model', {}).get('arch')!r}"
        )

    # Hand off to the adapter; it owns the upstream Trainer.train() loop and
    # is responsible for writing best.pdparams + test_results.json + the
    # observability tree under save_dir.
    return paddle_adapter.train_paddle(
        config=config,
        save_dir=save_dir,
        resume_from=resume_from,
    )


def _ensure_observability_tree(save_dir: Path) -> None:
    """Make sure the unified observability subdirectories exist even if the
    adapter hasn't populated them. Downstream consumers (releases/, p08
    rerun, etc.) walk these by name — empty is fine, missing is not.

    Mirrors the layout documented in core/p06_training/CLAUDE.md.
    """
    for sub in ("data_preview", "val_predictions", "test_predictions"):
        (save_dir / sub).mkdir(parents=True, exist_ok=True)
    # error_analysis lives one level under val_predictions/test_predictions
    (save_dir / "val_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)
    (save_dir / "test_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = _parse_args()
    save_dir: Path = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = _read_payload()
        config = payload["config"]

        summary = _build_and_train(
            config=config,
            save_dir=save_dir,
            resume_from=args.resume_from,
        )

        # Normalise summary shape to match train_with_hf.
        summary = dict(summary or {})
        summary.setdefault("output_dir", str(save_dir))
        summary.setdefault("total_epochs",
                           int((summary.get("metrics") or {}).get("epoch", 0)))

        _ensure_observability_tree(save_dir)
        _emit_summary(summary)
        return 0
    except Exception as e:  # pragma: no cover — surfaces in subprocess stderr
        sys.stderr.write(
            f"_paddle_runner failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        # Still emit a summary so the wrapper can distinguish "ran but
        # crashed" from "never started" — the wrapper additionally checks
        # the non-zero return code.
        _emit_summary({"error": str(e), "output_dir": str(save_dir)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
