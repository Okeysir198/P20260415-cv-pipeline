"""Shared LangGraph helpers used across tools."""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


def replace_reducer(_old: Any, new: Any) -> Any:
    """Default reducer: new value replaces old."""
    return new


def list_append_reducer(_old: List, new: List) -> List:
    """Reducer for image_results: replace (nodes manage appending themselves)."""
    return new


def get_batch_range(state: dict) -> Tuple[int, int]:
    """Return ``(start, end)`` indices for the current batch.

    Computes from state dict with keys ``current_batch_idx``, ``batch_size``,
    and the total count derived from ``sampled_paths`` or ``image_paths``.
    """
    paths_key = "sampled_paths" if "sampled_paths" in state else "image_paths"
    total = sum(len(paths) for paths in state[paths_key].values())
    start = state["current_batch_idx"] * state["batch_size"]
    end = min(start + state["batch_size"], total)
    return start, end


def get_batch_paths(state: dict) -> List[Tuple[str, Path]]:
    """Return ``(split, image_path)`` tuples for the current batch.

    Args:
        state: Current LangGraph state dict.

    Returns:
        List of ``(split, Path)`` pairs for the current batch window.
    """
    paths_key = "sampled_paths" if "sampled_paths" in state else "image_paths"
    all_paths: List[Tuple[str, Path]] = []
    for split, paths in state[paths_key].items():
        for p in paths:
            all_paths.append((split, Path(p)))

    start, end = get_batch_range(state)
    return all_paths[start:end]


def should_continue(state: dict) -> str:
    """Decide whether to process more batches or aggregate.

    Returns:
        "continue" if ``current_batch_idx < total_batches``, else "aggregate".
    """
    if state["current_batch_idx"] < state["total_batches"]:
        return "continue"
    return "aggregate"


def make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
