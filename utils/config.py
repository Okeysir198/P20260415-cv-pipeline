"""Configuration loading, merging, and validation utilities."""

import copy
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from loguru import logger

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and resolve relative paths.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dictionary with resolved paths.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    config = _resolve_variables(config, config)
    _migrate_legacy_tensor_prep(config)
    _sync_legacy_input_size_from_tensor_prep(config)

    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two config dictionaries. Override values take precedence.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary. Values here
            replace or extend those in base.

    Returns:
        New merged dictionary (inputs are not mutated).

    Examples:
        >>> base = {"model": {"arch": "yolox-m", "num_classes": 2}}
        >>> override = {"model": {"num_classes": 3}}
        >>> merge_configs(base, override)
        {'model': {'arch': 'yolox-m', 'num_classes': 3}}
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: dict, schema: str) -> bool:
    """Validate a config dictionary against a known schema type.

    Performs basic structural validation (required keys, value types).
    Does not perform full JSON Schema validation.

    Args:
        config: Configuration dictionary to validate.
        schema: Schema type — one of "data", "training", or "export".

    Returns:
        True if the config passes validation.

    Raises:
        ValueError: If required keys are missing or values have wrong types.
    """
    validators = {
        "data": _validate_data_config,
        "training": _validate_training_config,
        "export": _validate_export_config,
    }

    if schema not in validators:
        raise ValueError(f"Unknown schema type: {schema}. Must be one of {list(validators.keys())}")

    return validators[schema](config)


def resolve_path(path_str: str, base_dir: str | Path) -> Path:
    """Resolve a potentially relative path against a base directory.

    Args:
        path_str: Path string (absolute or relative).
        base_dir: Base directory for resolving relative paths.

    Returns:
        Resolved absolute Path object.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def feature_name_from_config_path(config_path: str | Path) -> str:
    """Derive the feature folder name from a config file or config-dir path.

    Assumes the repo convention ``features/<feature>/configs/<step>.yaml``:
    - Any path containing a ``features/<name>/`` segment → ``<name>``.
    - A bare ``configs/`` directory under a feature → parent name.
    - Anything else (e.g. ``configs/_test/…``) → ``"unknown"``.

    The explicit ``"unknown"`` fallback prevents the silent-ghost-folder
    bug where a config outside ``features/`` (like ``configs/_test/…``)
    used to resolve via ``parent.parent.name`` to the project root's dir
    name and then materialise as ``features/<project-root>/runs/…``.
    Callers that land here should pass an explicit ``output_dir_override``
    or set the ``CV_RUNS_BASE`` env var.
    """
    p = Path(config_path)
    if str(p) in (".", ""):
        return "unknown"
    resolved = p.resolve()
    # Preferred path: look for a `features/<name>/` segment anywhere.
    parts = resolved.parts
    for i in range(len(parts) - 1):
        if parts[i] == "features":
            return parts[i + 1]
    # Legacy compat: bare `configs/` dir whose parent IS a feature folder.
    if resolved.name == "configs" and resolved.is_dir():
        parent = resolved.parent
        if parent.parent.name == "features":
            return parent.name
    # No `features/<name>/` in the path — fall back to a safe sentinel.
    return "unknown"


def generate_run_dir(
    use_case: str,
    step: str,
    base_dir: str | Path | None = None,
) -> Path:
    """Generate a timestamped run directory under ``features/<use_case>/runs/``.

    Override order: ``base_dir`` arg → ``$CV_RUNS_BASE`` env → feature folder.

    Args:
        use_case: Feature name (must match a directory under ``features/``).
        step: Pipeline step name (e.g. "06_training").
        base_dir: Explicit base directory override.

    Returns:
        Path to the new run directory (not yet created).
    """
    leaf = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{step}"

    if base_dir is not None:
        return Path(base_dir) / use_case / leaf
    env_base = os.environ.get("CV_RUNS_BASE")
    if env_base:
        return Path(env_base) / use_case / leaf

    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "features" / use_case / "runs" / leaf


def parse_overrides(override_list: list) -> dict:
    """Parse key=value override strings into a nested dictionary.

    Supports dot-notation keys for nested config values
    (e.g. "training.lr=0.005" -> {"training": {"lr": 0.005}}).

    Args:
        override_list: List of "key=value" strings.

    Returns:
        Nested dictionary of parsed overrides.
    """
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value format, got: '{item}'")

        key, value_str = item.split("=", 1)
        keys = key.strip().split(".")
        value = _parse_value(value_str.strip())

        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return result


def _parse_value(value_str: str):
    """Parse a string value to its appropriate Python type."""
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "none":
        return None
    # JSON-ish list/dict literals, e.g. "[1,1]" or "[0.5,1.5]"
    stripped = value_str.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or \
       (stripped.startswith("{") and stripped.endswith("}")):
        import json
        try:
            return json.loads(stripped)
        except (ValueError, TypeError):
            pass
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


# --- Tensor-prep contract (applied_by = hf_processor | v2_pipeline) -------


_VALID_APPLIED_BY = ("hf_processor", "v2_pipeline")


def _migrate_legacy_tensor_prep(config: dict) -> None:
    """In-place: if `tensor_prep` is absent but legacy keys exist, synthesise it.

    Legacy keys read: ``model.input_size``, ``data.input_size``, ``data.mean``,
    ``data.std``, ``augmentation.normalize``. ``applied_by`` defaults to
    ``hf_processor`` on the HF backend, ``v2_pipeline`` otherwise.

    Emits one WARNING per migration so users know to add the explicit block.
    """
    if not isinstance(config, dict):
        return
    if "tensor_prep" in config and config["tensor_prep"]:
        # Already present. If it was auto-migrated on an earlier load pass
        # (e.g. before overrides flipped training.backend), refresh the
        # applied_by default to match the current backend. Explicit user
        # blocks (no _auto_migrated marker) are left untouched.
        tp = config["tensor_prep"]
        if isinstance(tp, dict) and tp.pop("_auto_migrated", False):
            backend = (config.get("training") or {}).get("backend", "pytorch")
            tp["applied_by"] = "hf_processor" if backend == "hf" else "v2_pipeline"
            tp["_auto_migrated"] = True
        return
    model_cfg = config.get("model") or {}
    data_cfg = config.get("data") or {}
    aug_cfg = config.get("augmentation") or {}
    # Only migrate when we have *any* legacy tensor-prep signal (a training
    # config). Unrelated configs (e.g. 05_data.yaml) must pass through
    # untouched.
    has_legacy = (
        "input_size" in model_cfg or "input_size" in data_cfg
        or "mean" in data_cfg or "std" in data_cfg
        or "normalize" in aug_cfg
    )
    if not has_legacy:
        return
    input_size = model_cfg.get("input_size") or data_cfg.get("input_size")
    if input_size is None:
        return  # nothing to anchor on
    mean = data_cfg.get("mean", _IMAGENET_MEAN)
    std = data_cfg.get("std", _IMAGENET_STD)
    normalize = bool(aug_cfg.get("normalize", True))
    backend = (config.get("training") or {}).get("backend", "pytorch")
    applied_by = "hf_processor" if backend == "hf" else "v2_pipeline"
    config["tensor_prep"] = {
        "input_size": list(input_size),
        "rescale": True,
        "normalize": normalize,
        "mean": list(mean) if mean is not None else list(_IMAGENET_MEAN),
        "std": list(std) if std is not None else list(_IMAGENET_STD),
        "applied_by": applied_by,
        "_auto_migrated": True,
    }
    logger.warning(
        "Legacy config: auto-migrated tensor_prep. Add an explicit tensor_prep "
        "block to 06_training.yaml to suppress this warning."
    )


def _sync_legacy_input_size_from_tensor_prep(config: dict) -> None:
    """Back-compat: if tensor_prep.input_size is set but model.input_size is
    missing, mirror it into model.input_size (and data.input_size when the
    data section is present). Many code paths still read model.input_size.

    This is intentionally one-way (tensor_prep → legacy); the reverse
    direction is handled by ``_migrate_legacy_tensor_prep`` on load.
    """
    tp = config.get("tensor_prep") or {}
    in_size = tp.get("input_size")
    if not in_size:
        return
    _m = config.get("model")
    model_cfg = config.setdefault("model", {}) if isinstance(_m, dict) else None
    if isinstance(model_cfg, dict) and "input_size" not in model_cfg:
        model_cfg["input_size"] = list(in_size)


def resolve_tensor_prep(config: dict, backend: str | None = None) -> dict:
    """Return the authoritative ``tensor_prep`` dict for this config.

    If ``tensor_prep`` is already in ``config``, it's returned as-is (a shallow
    copy). Otherwise the legacy migration runs on-demand. ``backend`` only
    influences the migration default for ``applied_by``.
    """
    if backend is not None and (config.get("training") or {}).get("backend") is None:
        # Prime backend hint for the migration helper.
        config.setdefault("training", {})["backend"] = backend
    # Always call migration — the refresh-on-backend-change path updates a
    # previously auto-migrated block when overrides flipped training.backend.
    _migrate_legacy_tensor_prep(config)
    tp = config.get("tensor_prep") or {}
    # Strip the internal marker from the returned copy (it's implementation
    # detail, not part of the contract).
    tp = {k: v for k, v in tp.items() if k != "_auto_migrated"}
    return tp


def _validate_tensor_prep(config: dict, backend: str, processor: Any = None) -> None:
    """Hard-error on illegal tensor_prep states.

    See contract in the user-facing docstring / CLAUDE.md: enforces
    applied_by vs backend, mandatory mean/std when normalize=true, and the
    'no double-normalize / no missing-normalize' invariant.
    """
    tp = resolve_tensor_prep(config, backend=backend)
    if not tp:
        raise ValueError(
            "tensor_prep is required in 06_training.yaml. Example:\n"
            "  tensor_prep:\n"
            "    input_size: [480, 480]\n"
            "    rescale: true\n"
            "    normalize: true\n"
            "    mean: [0.485, 0.456, 0.406]\n"
            "    std:  [0.229, 0.224, 0.225]\n"
            "    applied_by: hf_processor   # or v2_pipeline"
        )

    applied_by = tp.get("applied_by")
    if applied_by not in _VALID_APPLIED_BY:
        raise ValueError(
            f"tensor_prep.applied_by must be one of {_VALID_APPLIED_BY}, "
            f"got {applied_by!r}"
        )

    if "input_size" not in tp or tp["input_size"] is None:
        raise ValueError("tensor_prep.input_size is required")

    normalize = bool(tp.get("normalize", True))
    if normalize and (tp.get("mean") is None or tp.get("std") is None):
        raise ValueError(
            "tensor_prep.normalize=true requires non-null mean and std"
        )

    if applied_by == "hf_processor" and backend != "hf":
        raise ValueError(
            "tensor_prep.applied_by='hf_processor' is only valid on the HF "
            f"backend, but training.backend={backend!r}. Use "
            "applied_by='v2_pipeline' for the pytorch backend."
        )

    # Site-by-site normalize enforcement (only meaningful when processor exists).
    if processor is not None:
        proc_normalize = bool(getattr(processor, "do_normalize", True))
        if applied_by == "hf_processor":
            # Processor must be forced to normalize if the user asked for it.
            if normalize and not proc_normalize:
                raise ValueError(
                    "tensor_prep.normalize=true with applied_by='hf_processor' "
                    "but processor.do_normalize is False. Did the override "
                    "pass run? build_hf_model should force this."
                )
        else:  # v2_pipeline
            if proc_normalize and normalize:
                raise ValueError(
                    "Double-normalize: applied_by='v2_pipeline' requires "
                    "processor.do_normalize=False, but processor still has "
                    "do_normalize=True. Disable the processor's normalize "
                    "step in build_hf_model."
                )


# --- Private helpers ---


def _resolve_variables(obj: Any, root: dict) -> Any:
    """Recursively resolve ${...} variable references in config values.

    Supports simple dot-notation references like ${model.arch}.

    Args:
        obj: Current object being processed (str, dict, list, or scalar).
        root: Root config dictionary for variable lookups.

    Returns:
        Object with all variable references resolved.
    """
    if isinstance(obj, str):
        match = _VAR_PATTERN.search(obj)
        while match:
            var_path = match.group(1)
            value = _lookup_dotpath(root, var_path)
            if value is not None:
                if match.start() == 0 and match.end() == len(obj):
                    # Entire string is a variable reference — return native type
                    return value
                # Partial replacement — convert to string
                obj = obj[: match.start()] + str(value) + obj[match.end() :]
            match = _VAR_PATTERN.search(obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_variables(v, root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_variables(item, root) for item in obj]
    return obj


def _lookup_dotpath(config: dict, dotpath: str) -> Any:
    """Look up a value in a nested dict using dot-separated path.

    Args:
        config: Nested dictionary.
        dotpath: Dot-separated key path (e.g. "model.arch").

    Returns:
        Value at the path, or None if not found.
    """
    keys = dotpath.split(".")
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _validate_data_config(config: dict) -> bool:
    """Validate a data config has required fields."""
    required = ["dataset_name", "path", "train", "val", "names", "num_classes", "input_size"]
    _check_required_keys(config, required, "data")

    if not isinstance(config["names"], dict):
        raise ValueError("data config: 'names' must be a dict mapping int -> str")

    if len(config["names"]) != config["num_classes"]:
        raise ValueError(
            f"data config: 'num_classes' ({config['num_classes']}) does not match "
            f"number of names ({len(config['names'])})"
        )

    if not isinstance(config["input_size"], list) or len(config["input_size"]) != 2:
        raise ValueError("data config: 'input_size' must be a list of [height, width]")

    return True


def _validate_training_config(config: dict) -> bool:
    """Validate a training config has required sections."""
    required_sections = ["model", "data", "training", "logging"]
    _check_required_keys(config, required_sections, "training")

    # depth/width only required for YOLOX-family models, not for timm/HF classifiers
    arch = config["model"].get("arch", "")
    # tensor_prep.input_size supersedes model.input_size when present.
    has_tp_input = bool((config.get("tensor_prep") or {}).get("input_size"))
    if arch.startswith("yolox"):
        model_keys = ["arch", "num_classes", "depth", "width"]
        if not has_tp_input:
            model_keys.append("input_size")
    else:
        model_keys = ["arch", "num_classes"]
        if not has_tp_input:
            model_keys.append("input_size")
    _check_required_keys(config["model"], model_keys, "training.model")

    training_keys = ["epochs", "optimizer", "lr"]
    _check_required_keys(config["training"], training_keys, "training.training")

    if config["training"]["epochs"] <= 0:
        raise ValueError("training config: 'epochs' must be positive")

    if config["training"]["lr"] <= 0:
        raise ValueError("training config: 'lr' must be positive")

    return True


def _validate_export_config(config: dict) -> bool:
    """Validate an export config has required fields."""
    required = ["format", "input_size", "output_dir", "naming"]
    _check_required_keys(config, required, "export")

    valid_formats = ["onnx"]
    if config["format"] not in valid_formats:
        raise ValueError(f"export config: 'format' must be one of {valid_formats}")

    return True


def _check_required_keys(config: dict, keys: list, context: str) -> None:
    """Check that all required keys exist in a config dict.

    Args:
        config: Dictionary to check.
        keys: List of required key names.
        context: Description for error messages (e.g. "data", "training.model").

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"{context} config: missing required keys: {missing}")
