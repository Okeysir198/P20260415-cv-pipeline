"""Configuration loading, merging, and validation utilities."""

import copy
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


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
    if arch.startswith("yolox"):
        model_keys = ["arch", "num_classes", "input_size", "depth", "width"]
    else:
        model_keys = ["arch", "num_classes", "input_size"]
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
