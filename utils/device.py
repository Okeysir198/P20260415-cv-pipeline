"""Device detection and reproducibility utilities."""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA > MPS > CPU. Optionally force a specific device.

    Args:
        device: Force a specific device string (e.g. "cuda:0", "cpu").
            If None, auto-detects the best available device.

    Returns:
        torch.device for the selected compute backend.

    Examples:
        >>> dev = get_device()       # auto-detect
        >>> dev = get_device("cpu")  # force CPU
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    Also configures PyTorch for deterministic operations where possible.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for the current system.

    Returns:
        Dictionary with GPU details:
            - available: bool — whether a GPU is available
            - count: int — number of GPUs
            - devices: list[dict] — per-device info (name, memory, capability)
            - cuda_version: str — CUDA toolkit version

    Examples:
        >>> info = get_gpu_info()
        >>> if info["available"]:
        ...     print(f"GPU: {info['devices'][0]['name']}")
    """
    info: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "count": 0,
        "devices": [],
        "cuda_version": None,
    }

    if not torch.cuda.is_available():
        return info

    info["count"] = torch.cuda.device_count()
    info["cuda_version"] = torch.version.cuda

    for i in range(info["count"]):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_mb": round(props.total_memory / (1024 ** 2)),
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        }

        # Current memory usage (only if device is initialized)
        try:
            device_info["allocated_memory_mb"] = round(
                torch.cuda.memory_allocated(i) / (1024 ** 2)
            )
            device_info["cached_memory_mb"] = round(
                torch.cuda.memory_reserved(i) / (1024 ** 2)
            )
        except RuntimeError:
            device_info["allocated_memory_mb"] = 0
            device_info["cached_memory_mb"] = 0

        info["devices"].append(device_info)

    return info
