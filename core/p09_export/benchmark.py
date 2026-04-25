"""Benchmark model inference performance across formats.

Compares PyTorch (.pt) vs ONNX (.onnx) vs quantized models.
Measures: latency (ms), throughput (FPS), model size (MB), memory usage.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root


class ModelBenchmark:
    """Benchmark model inference performance.

    Compares: PyTorch (.pt) vs ONNX (.onnx) vs quantized.
    Measures: latency (ms), throughput (FPS), model size (MB), memory usage.

    Args:
        input_size: Input image size as (H, W).
        warmup_runs: Number of warmup iterations before timing.
        num_runs: Number of timed iterations.
        device: Device for PyTorch benchmarking ("cpu", "cuda", etc.).
    """

    def __init__(
        self,
        input_size: tuple = (640, 640),
        warmup_runs: int = 10,
        num_runs: int = 100,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        self.device = device

    def benchmark_pytorch(
        self,
        model,
        batch_size: int = 1,
        label: str = "PyTorch",
    ) -> dict:
        """Benchmark a PyTorch model.

        Args:
            model: PyTorch nn.Module in eval mode.
            batch_size: Batch size for inference.
            label: Label for this result in comparison tables.

        Returns:
            Dictionary with benchmark results.
        """
        model.eval()
        device = torch.device(self.device)
        model = model.to(device)

        dummy = torch.randn(
            batch_size, 3, *self.input_size, device=device, dtype=torch.float32
        )

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy)

        # Sync CUDA before timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        # Memory usage
        memory_mb = None
        if device.type == "cuda":
            memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            torch.cuda.reset_peak_memory_stats(device)

        # Model size (parameter-based estimate)
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        model_size_mb = round(param_bytes / (1024 * 1024), 2)

        return self._build_result(
            label=label,
            format="pytorch",
            latencies=latencies,
            batch_size=batch_size,
            model_size_mb=model_size_mb,
            memory_mb=memory_mb,
        )

    def benchmark_onnx(
        self,
        onnx_path: str,
        batch_size: int = 1,
        label: str = "ONNX",
        providers: list | None = None,
    ) -> dict:
        """Benchmark an ONNX model via onnxruntime.

        Args:
            onnx_path: Path to the .onnx model.
            batch_size: Batch size for inference.
            label: Label for this result in comparison tables.
            providers: ORT execution providers. Default: CUDAExecutionProvider.

        Returns:
            Dictionary with benchmark results.
        """
        if providers is None:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("GPU required: onnxruntime-gpu not available for ONNX benchmarking.")
            providers = ["CUDAExecutionProvider"]

        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name

        dummy = np.random.randn(
            batch_size, 3, *self.input_size
        ).astype(np.float32)

        # Warmup
        for _ in range(self.warmup_runs):
            session.run(None, {input_name: dummy})

        # Timed runs
        latencies = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        # Model size on disk
        model_size_mb = round(os.path.getsize(onnx_path) / (1024 * 1024), 2)

        return self._build_result(
            label=label,
            format="onnx",
            latencies=latencies,
            batch_size=batch_size,
            model_size_mb=model_size_mb,
            memory_mb=None,
            path=onnx_path,
        )

    def compare(self, results: list[dict]) -> str:
        """Pretty-print a comparison table of benchmark results.

        Args:
            results: List of result dicts from benchmark_* methods.

        Returns:
            Formatted comparison table as a string.
        """
        if not results:
            return "No results to compare."

        # Header
        sep = "-" * 90
        header = (
            f"{'Label':<20} {'Format':<10} {'Size (MB)':>10} "
            f"{'Latency (ms)':>14} {'Std (ms)':>10} {'FPS':>10} "
            f"{'Mem (MB)':>10}"
        )

        lines = [sep, "  Model Benchmark Comparison", sep, header, sep]

        for r in results:
            mem_str = f"{r['memory_mb']:.1f}" if r["memory_mb"] is not None else "N/A"
            line = (
                f"{r['label']:<20} {r['format']:<10} {r['model_size_mb']:>10.2f} "
                f"{r['latency_mean_ms']:>14.2f} {r['latency_std_ms']:>10.2f} "
                f"{r['throughput_fps']:>10.1f} {mem_str:>10}"
            )
            lines.append(line)

        lines.append(sep)

        # Speedup vs first result
        if len(results) > 1:
            base = results[0]["latency_mean_ms"]
            lines.append("\n  Speedup vs " + results[0]["label"] + ":")
            for r in results[1:]:
                speedup = base / r["latency_mean_ms"] if r["latency_mean_ms"] > 0 else 0
                lines.append(f"    {r['label']}: {speedup:.2f}x")
            lines.append(sep)

        table = "\n".join(lines)
        return table

    def save_results(self, results: list[dict], save_path: str) -> None:
        """Save benchmark results as JSON.

        Args:
            results: List of result dicts from benchmark_* methods.
            save_path: Path to save the JSON file.
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "config": {
                "input_size": list(self.input_size),
                "warmup_runs": self.warmup_runs,
                "num_runs": self.num_runs,
                "device": self.device,
            },
            "results": results,
        }

        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Benchmark results saved to: %s", save_path)

    @staticmethod
    def _build_result(
        label: str,
        format: str,
        latencies: list,
        batch_size: int,
        model_size_mb: float,
        memory_mb: float | None = None,
        path: str | None = None,
    ) -> dict:
        """Build a standardized result dictionary from raw latencies.

        Args:
            label: Human-readable label.
            format: Model format ("pytorch", "onnx").
            latencies: List of per-inference latencies in ms.
            batch_size: Batch size used.
            model_size_mb: Model size in MB.
            memory_mb: Peak GPU memory if applicable.
            path: Path to model file if applicable.

        Returns:
            Standardized benchmark result dict.
        """
        arr = np.array(latencies)
        mean_ms = float(np.mean(arr))
        fps = (1000.0 / mean_ms) * batch_size if mean_ms > 0 else 0.0

        result = {
            "label": label,
            "format": format,
            "batch_size": batch_size,
            "model_size_mb": model_size_mb,
            "latency_mean_ms": round(mean_ms, 3),
            "latency_std_ms": round(float(np.std(arr)), 3),
            "latency_min_ms": round(float(np.min(arr)), 3),
            "latency_max_ms": round(float(np.max(arr)), 3),
            "latency_p50_ms": round(float(np.percentile(arr, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
            "latency_p99_ms": round(float(np.percentile(arr, 99)), 3),
            "throughput_fps": round(fps, 1),
            "memory_mb": memory_mb,
            "num_runs": len(latencies),
        }

        if path is not None:
            result["path"] = path

        logger.info(
            "Benchmark [%s]: %.2f ms (std %.2f), %.1f FPS, %.2f MB",
            label,
            mean_ms,
            result["latency_std_ms"],
            fps,
            model_size_mb,
        )

        return result
