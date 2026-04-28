"""Train models using a native PaddlePaddle backend (PaddleDetection /
PaddleClas / PaddleSeg).

This wrapper is the **main-venv-side** entry point. It performs zero
``paddle`` imports — the heavy lifting lives in
``core.p06_training._paddle_runner`` which is executed inside
``.venv-paddle/`` via subprocess. The split mirrors the
``_OfficialYOLOXAdapter`` pattern (`core/p06_models/yolox.py`) which shells
into ``.venv-yolox-official/`` for a similar reason — paddle's pinned
``paddlepaddle-gpu`` wheels conflict with the main venv's CUDA 13 torch +
git transformers stack.

Usage (drop-in replacement for ``train_with_hf``):

    from core.p06_training.paddle_trainer import train_with_paddle
    summary = train_with_paddle("features/safety-fire_detection/configs/06_training_paddle.yaml")

The returned dict has the same shape as ``train_with_hf``:
    {"metrics": {...}, "best_metric": float, "best_epoch": int,
     "total_epochs": int, "output_dir": str, ...}

The runner emits a sentinel-delimited JSON block on stdout
(``___PADDLE_SUMMARY_JSON___\\n<json>\\n___END___``) so we can parse it
reliably even when paddle's verbose logger interleaves arbitrary text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from utils.config import (
    feature_name_from_config_path,
    generate_run_dir,
    load_config,
    merge_configs,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PADDLE_VENV_DIR = _PROJECT_ROOT / ".venv-paddle"
_PADDLE_PYTHON = _PADDLE_VENV_DIR / "bin" / "python"

# Sentinels — must match `_paddle_runner.py`.
_SUMMARY_BEGIN = "___PADDLE_SUMMARY_JSON___"
_SUMMARY_END = "___END___"


def _ensure_paddle_venv() -> Path:
    """Resolve the paddle venv's Python interpreter; fail with a clear error
    pointing at ``scripts/setup-paddle-venv.sh`` when it's missing."""
    if not _PADDLE_PYTHON.is_file() or not os.access(_PADDLE_PYTHON, os.X_OK):
        raise RuntimeError(
            f"PaddlePaddle backend requires {_PADDLE_VENV_DIR} but it was not "
            f"found (expected interpreter at {_PADDLE_PYTHON}).\n"
            f"Create it with:\n  bash scripts/setup-paddle-venv.sh"
        )
    return _PADDLE_PYTHON


def _resolve_save_dir(config_path: Path, config: dict) -> Path:
    """Derive the run directory from the config — main-venv side, so
    ``feature_name_from_config_path`` and ``generate_run_dir`` stay consistent
    with every other pipeline phase. Honours an explicit ``logging.save_dir``
    just like the HF/pytorch backends."""
    log_cfg = (config or {}).get("logging", {}) or {}
    save_dir = log_cfg.get("save_dir")
    if save_dir:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = (config_path.parent / save_dir).resolve()
        return save_dir

    feature_name = feature_name_from_config_path(str(config_path))
    explicit_run_name = log_cfg.get("run_name") or log_cfg.get("project")
    run_name = explicit_run_name or feature_name
    if explicit_run_name and feature_name != "unknown" and explicit_run_name != feature_name:
        logger.warning(
            "logging.run_name=%r differs from feature folder %r — "
            "paddle run will land under features/%s/runs/ (likely a stale config).",
            explicit_run_name, feature_name, run_name,
        )
    return Path(generate_run_dir(run_name, "06_training"))


def _parse_summary(stdout: str) -> dict[str, Any]:
    """Pull the sentinel-delimited JSON block out of the runner's stdout.

    Paddle is *very* chatty (every step logs metrics + GPU memory), so we
    can't rely on JSON being the entire stdout — instead the runner brackets
    it with sentinels we control. Returns ``{}`` if the block is missing
    or unparseable; caller decides how strict to be.
    """
    begin = stdout.rfind(_SUMMARY_BEGIN)
    if begin < 0:
        return {}
    end = stdout.find(_SUMMARY_END, begin)
    if end < 0:
        return {}
    payload = stdout[begin + len(_SUMMARY_BEGIN):end].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        logger.warning("Could not parse paddle runner summary JSON: %s", e)
        return {}


def train_with_paddle(
    config_path: str,
    overrides: dict | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Train a model via PaddlePaddle (PaddleDetection / PaddleClas / PaddleSeg).

    Same function shape as ``core.p06_training.hf_trainer.train_with_hf``:

    Args:
        config_path: Path to our 06_training YAML config.
        overrides: Nested dict of overrides (same as ``--override`` CLI).
        resume_from: Optional path to a paddle checkpoint to resume from.

    Returns:
        Summary dict with keys ``metrics``, ``best_metric``, ``best_epoch``,
        ``total_epochs``, ``output_dir``.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    paddle_python = _ensure_paddle_venv()

    # Read + merge overrides on the main-venv side so the runner sees the
    # final config (and so dataset_config resolution + run-dir derivation
    # share one source of truth with hf_trainer / pytorch trainer).
    config = load_config(str(config_path))
    if overrides:
        config = merge_configs(config, overrides)

    save_dir = _resolve_save_dir(config_path, config)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Paddle backend: run dir = %s", save_dir)

    feature_name = feature_name_from_config_path(str(config_path))

    # Pass the resolved config + save_dir + resume target to the runner via
    # stdin (avoids argv length limits on large configs and shell escaping
    # foot-guns). argv carries the small things only.
    # Invoke via file path, not `-m core.p06_training._paddle_runner`, to
    # bypass `core/p06_training/__init__.py` — that package init eagerly
    # imports torch via metrics_registry, which the paddle venv may not have.
    runner_path = Path(__file__).with_name("_paddle_runner.py")
    runner_args = [
        str(paddle_python),
        str(runner_path),
        "--save-dir", str(save_dir),
        "--config-path", str(config_path),
        "--feature-name", feature_name,
    ]
    if resume_from:
        runner_args += ["--resume-from", str(resume_from)]

    payload = json.dumps({"config": config}, default=str)

    env = os.environ.copy()
    # Make the project root importable in the paddle venv even when modules
    # are run via `-m core.p06_training._paddle_runner` from a different cwd.
    env["PYTHONPATH"] = (
        f"{_PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    )

    logger.info("Spawning paddle runner: %s", " ".join(runner_args))
    # Use Popen + line streaming so paddle's per-step logs reach the parent
    # logger live (training runs hours; capture_output would silence the
    # whole subprocess until exit). stdout is captured into a buffer for
    # sentinel-based summary parsing; stderr is line-streamed to the parent
    # logger as it arrives.
    proc = subprocess.Popen(
        runner_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(_PROJECT_ROOT),
        env=env,
    )
    proc.stdin.write(payload)
    proc.stdin.close()

    import threading
    stderr_lines: list[str] = []

    def _pump_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)
            logger.opt(raw=True).info(f"[paddle] {line}")

    t = threading.Thread(target=_pump_stderr, daemon=True)
    t.start()

    stdout_text = proc.stdout.read()
    rc = proc.wait()
    t.join()

    if rc != 0:
        raise RuntimeError(
            f"PaddlePaddle runner exited with code {rc}. "
            f"stderr tail:\n{''.join(stderr_lines)[-2000:]}"
        )

    summary = _parse_summary(stdout_text or "")
    if not summary:
        raise RuntimeError(
            "PaddlePaddle runner finished but did not emit a summary block. "
            "Last 2000 chars of stdout:\n"
            f"{(stdout_text or '')[-2000:]}"
        )

    summary.setdefault("output_dir", str(save_dir))
    logger.info("Paddle training complete: %s", {
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "total_epochs": summary.get("total_epochs"),
        "output_dir": summary.get("output_dir"),
    })
    return summary
