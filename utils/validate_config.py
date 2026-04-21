"""Config validation CLI — fail fast before training begins.

Usage::

    # Validate a single config directory
    uv run utils/validate_config.py features/safety-fire_detection/configs/

    # Validate a specific file
    uv run utils/validate_config.py features/safety-fire_detection/configs/06_training.yaml

    # Validate all configs
    uv run utils/validate_config.py configs/

    # CI-friendly (exit code 0 = pass, 1 = fail)
    uv run utils/validate_config.py configs/ && echo "All configs valid"

Exit codes: 0 = all valid, 1 = one or more errors.
"""

import argparse
import functools
import sys
from pathlib import Path

import yaml

# One sys.path.insert at the project root (this file lives in utils/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config, resolve_path

# --- Terminal color helpers ---

_USE_COLOR = sys.stdout.isatty()


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _USE_COLOR else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _USE_COLOR else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _USE_COLOR else text


# --- Registry loader (lazy, cached) ---


@functools.cache
def _load_registry() -> frozenset[str]:
    """Return all registered model arch names. Cached after first call."""
    try:
        from core.p06_models.registry import _VARIANT_MAP, MODEL_REGISTRY

        return frozenset(MODEL_REGISTRY.keys()) | frozenset(_VARIANT_MAP.keys())
    except Exception:
        # torch not installed (e.g. lightweight CI) — skip arch check
        return frozenset()


# --- Validation helpers ---


def _missing(config: dict, keys: list[str]) -> list[str]:
    """Return keys absent from *config* (supports dot-notation for nested keys)."""
    missing = []
    for key in keys:
        parts = key.split(".")
        current = config
        found = True
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if not found:
            missing.append(key)
    return missing


# --- Per-file validators ---
# Each returns (errors: list[str], warnings: list[str])


def _validate_data(config: dict, config_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required = ["dataset_name", "path", "train", "val", "names", "num_classes", "input_size"]
    for key in _missing(config, required):
        errors.append(f"missing required key: '{key}'")

    if errors:
        return errors, warnings

    names = config["names"]
    num_classes = config["num_classes"]
    if not isinstance(names, dict):
        errors.append("'names' must be a dict mapping int -> str")
    elif len(names) != num_classes:
        errors.append(
            f"'num_classes' ({num_classes}) does not match "
            f"number of entries in 'names' ({len(names)})"
        )

    input_size = config["input_size"]
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or not all(isinstance(v, int) and v > 0 for v in input_size)
    ):
        errors.append("'input_size' must be a list of 2 positive integers e.g. [640, 640]")

    # Dataset path existence (warn only — datasets may not be downloaded yet)
    dataset_path = resolve_path(config["path"], config_path.parent)
    if not dataset_path.exists():
        warnings.append(f"dataset path '{dataset_path}' not found (download before training)")
    else:
        for split_key in ("train", "val"):
            split_dir = dataset_path / config[split_key]
            if not split_dir.exists():
                warnings.append(f"{split_key} split dir '{split_dir}' not found")

    return errors, warnings


def _validate_training(config: dict, config_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        "model.arch",
        "model.num_classes",
        "training.epochs",
        "training.lr",
    ]
    absent = _missing(config, required)

    # batch_size lives under data.batch_size in the real schema; accept either location
    if _missing(config, ["data.batch_size"]) and _missing(config, ["training.batch_size"]):
        absent.append("data.batch_size")

    for key in absent:
        errors.append(f"missing required key: '{key}'")

    if errors:
        return errors, warnings

    arch = config["model"]["arch"]
    known = _load_registry()
    if known and arch not in known:
        errors.append(f"unknown model arch '{arch}'; registered archs: {sorted(known)}")

    num_classes = config["model"]["num_classes"]
    if not isinstance(num_classes, int) or num_classes <= 0:
        errors.append("'model.num_classes' must be a positive integer")

    epochs = config["training"]["epochs"]
    if not isinstance(epochs, int) or epochs <= 0:
        errors.append("'training.epochs' must be a positive integer")

    lr = config["training"]["lr"]
    if not isinstance(lr, (int, float)) or lr <= 0:
        errors.append("'training.lr' must be a positive number")

    backend = config.get("training", {}).get("backend")
    if backend is not None and backend not in ("pytorch", "hf", "custom"):
        errors.append(
            f"'training.backend' must be one of ['pytorch', 'hf', 'custom'], got '{backend}'"
        )

    # Warn if a local pretrained path doesn't exist (HF model ids contain '/' without leading './')
    pretrained = config["model"].get("pretrained")
    if pretrained and isinstance(pretrained, str):
        is_local = "/" not in pretrained or pretrained.startswith((".", "/"))
        if is_local:
            pretrained_path = resolve_path(pretrained, config_path.parent)
            if not pretrained_path.exists():
                warnings.append(
                    f"pretrained path '{pretrained}' not found (download before training)"
                )

    dataset_config = config.get("data", {}).get("dataset_config")
    if dataset_config:
        dc_path = resolve_path(dataset_config, config_path.parent)
        if not dc_path.exists():
            errors.append(f"data.dataset_config '{dc_path}' not found")

    return errors, warnings


# --- Dispatch by filename ---

_VALIDATORS = {
    "05_data.yaml": _validate_data,
    "06_training.yaml": _validate_training,
}

# Directories whose configs follow a different schema — skip validation
_SKIP_DIRS = {"_shared", "_test"}


def validate_file(path: Path) -> tuple[list[str], list[str], bool]:
    """Validate a single YAML file.

    Returns:
        (errors, warnings, skipped) — skipped=True when no validator applies.
    """
    if path.name not in _VALIDATORS:
        return [], [], True

    try:
        config = load_config(path)
    except (yaml.YAMLError, OSError) as exc:
        return [str(exc)], [], False

    errors, warnings = _VALIDATORS[path.name](config, path)
    return errors, warnings, False


# --- Discovery ---


def collect_config_files(target: Path) -> list[Path]:
    """Return all validatable config files under *target*, sorted by path.

    If *target* is a file, returns ``[target]`` when validatable, else ``[]``.
    Directories named ``_shared`` or ``_test`` are excluded.
    """
    if target.is_file():
        return [target] if target.name in _VALIDATORS else []

    files: list[Path] = []
    for found in sorted(target.rglob("*.yaml")):
        if found.name not in _VALIDATORS:
            continue
        if any(part in _SKIP_DIRS for part in found.parts):
            continue
        files.append(found)
    return files


# --- Main ---


def run(target_str: str) -> int:
    """Run validation; return exit code 0 (all valid) or 1 (errors found)."""
    target = Path(target_str).resolve()
    if not target.exists():
        print(_red(f"ERROR: path not found: {target}"))
        return 1

    files = collect_config_files(target)
    if not files:
        print(_yellow(f"WARNING: no validatable configs found under '{target}'"))
        return 0

    cwd = Path.cwd()
    total_ok = 0
    total_errors = 0
    total_warnings = 0

    for path in files:
        rel = path.relative_to(cwd) if path.is_relative_to(cwd) else path
        prefix = f"Validating {rel} ..."

        errors, warnings, skipped = validate_file(path)
        if skipped:
            continue

        if errors:
            print(f"{prefix} {_red('ERROR')}")
            for msg in errors:
                print(f"  {_red('ERROR')}: {msg}")
            for msg in warnings:
                print(f"  {_yellow('WARNING')}: {msg}")
            total_errors += 1
        else:
            print(f"{prefix} {_green('OK')}")
            for msg in warnings:
                print(f"  {_yellow('WARNING')}: {msg}")
            total_ok += 1

        total_warnings += len(warnings)

    parts = [
        _green(f"{total_ok} valid"),
        _red(f"{total_errors} errors") if total_errors else f"{total_errors} errors",
        _yellow(f"{total_warnings} warnings") if total_warnings else f"{total_warnings} warnings",
    ]
    print(f"\nSummary: {', '.join(parts)}")

    return 1 if total_errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate camera_edge YAML configs before training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        help="Config file or directory to validate (e.g. features/safety-fire_detection/configs/ or configs/)",
    )
    args = parser.parse_args()
    sys.exit(run(args.target))


if __name__ == "__main__":
    main()
