"""Device detection and reproducibility utilities."""

import os
import random
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def auto_select_gpu(verbose: bool = True) -> Optional[str]:
    """Pick the NVIDIA GPU with the most free memory and set CUDA_VISIBLE_DEVICES.

    Respects an existing user-supplied ``CUDA_VISIBLE_DEVICES`` (never
    overrides). No-ops and returns None if ``nvidia-smi`` is missing or
    reports no GPUs.

    MUST be called before any CUDA-using library performs its first CUDA
    op. ``import torch`` itself is fine — torch doesn't touch CUDA until
    the first ``torch.cuda.*`` call or tensor-to-cuda move — but anything
    that queries/initialises CUDA (e.g. ``torch.cuda.is_available()``,
    ``ort.InferenceSession(providers=["CUDAExecutionProvider"])``) will
    freeze the visible-device set to whatever it sees at that moment.

    Returns:
        The chosen device index as a string (e.g. ``"1"``), or the
        pre-existing ``CUDA_VISIBLE_DEVICES`` value if already set, or
        ``None`` if no GPU could be chosen.

    Examples:
        >>> # At the top of a test or training entrypoint, before
        >>> # torch does anything cuda-ish:
        >>> from utils.device import auto_select_gpu
        >>> auto_select_gpu()
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing:
        return existing

    free_mb = _query_free_memory_mib()
    if not free_mb:
        return None

    best = max(range(len(free_mb)), key=lambda i: free_mb[i])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    if verbose:
        summary = ", ".join(
            f"gpu{i}={mb}MiB" + (" [picked]" if i == best else "")
            for i, mb in enumerate(free_mb)
        )
        print(f"[utils.device.auto_select_gpu] {summary}")
    return str(best)


def _query_free_memory_mib() -> List[int]:
    """Return a list of free VRAM (MiB) per GPU, or [] if nvidia-smi missing."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    if out.returncode != 0:
        return []

    free: List[int] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            free.append(int(line))
        except ValueError:
            continue
    return free


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
