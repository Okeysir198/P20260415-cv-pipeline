"""Search space definition and sampling for Optuna HPO."""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from loguru import logger


class SearchSpace:
    """Parse YAML search space config and sample hyperparameters via Optuna trial.

    Args:
        search_space_config: Dictionary from configs/_shared/08_hyperparameter_tuning.yaml["search_space"].
    """

    def __init__(self, search_space_config: dict) -> None:
        self._config = search_space_config
        self._params = self._parse_params()

    def _parse_params(self) -> list[dict]:
        """Flatten nested search space config into a list of parameter definitions.

        Each entry has: name (dot-separated), type, and range info.
        """
        params = []
        for section_key, section_params in self._config.items():
            if isinstance(section_params, dict):
                for param_key, param_def in section_params.items():
                    if isinstance(param_def, dict) and "type" in param_def:
                        params.append({
                            "name": f"{section_key}.{param_key}",
                            "section": section_key,
                            "key": param_key,
                            **param_def,
                        })
        return params

    def sample(self, trial) -> dict:
        """Sample hyperparameters for one Optuna trial.

        Args:
            trial: optuna.Trial instance.

        Returns:
            Nested override dict compatible with merge_configs().
        """
        sampled = {}  # flat name -> value
        overrides = {}  # nested dict

        for param in self._params:
            # Check conditional
            condition = param.get("condition")
            if condition:
                dep_param = condition["param"]
                dep_value = condition["value"]
                if sampled.get(dep_param) != dep_value:
                    continue

            name = param["name"]
            value = self._suggest(trial, name, param)
            sampled[name] = value

            # Build nested dict
            self.set_nested(overrides, name, value)

        return overrides

    def _suggest(self, trial, name: str, param: dict) -> Any:
        """Call the appropriate trial.suggest_* method."""
        suggest_map = {
            "float": lambda: trial.suggest_float(name, param["low"], param["high"]),
            "log_float": lambda: trial.suggest_float(name, param["low"], param["high"], log=True),
            "int": lambda: trial.suggest_int(name, param["low"], param["high"]),
            "categorical": lambda: trial.suggest_categorical(name, param["choices"]),
        }
        ptype = param["type"]
        if ptype not in suggest_map:
            raise ValueError(f"Unknown parameter type: {ptype} for {name}")
        return suggest_map[ptype]()

    @staticmethod
    def set_nested(d: dict, name: str, value: Any) -> None:
        """Set a value in a nested dict using dot-separated key.

        Special handling for scale_min/scale_max which map to augmentation.scale list.

        Args:
            d: Target dictionary to set value in.
            name: Dot-separated key path (e.g. "training.lr").
            value: Value to set.
        """
        keys = name.split(".")

        # Special case: scale_min/scale_max -> scale list
        if keys[-1] in ("scale_min", "scale_max"):
            section = keys[0]
            d.setdefault(section, {})
            if "scale" not in d[section]:
                d[section]["scale"] = [None, None]
            idx = 0 if keys[-1] == "scale_min" else 1
            d[section]["scale"][idx] = value
            return

        current = d
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    def summary(self) -> str:
        """Return a human-readable summary of the search space."""
        lines = ["Search Space Summary:", "=" * 60]

        current_section = None
        for param in self._params:
            if param["section"] != current_section:
                current_section = param["section"]
                lines.append(f"\n  [{current_section}]")

            ptype = param["type"]
            name = param["name"]

            if ptype in ("float", "log_float"):
                scale = " (log)" if ptype == "log_float" else ""
                lines.append(f"    {name}: [{param['low']}, {param['high']}]{scale}")
            elif ptype == "int":
                lines.append(f"    {name}: [{param['low']}, {param['high']}] (int)")
            elif ptype == "categorical":
                lines.append(f"    {name}: {param['choices']}")

            if param.get("condition"):
                cond = param["condition"]
                lines.append(f"      └─ only when {cond['param']} = {cond['value']}")

        lines.append(f"\n  Total parameters: {len(self._params)}")
        lines.append("=" * 60)
        return "\n".join(lines)
