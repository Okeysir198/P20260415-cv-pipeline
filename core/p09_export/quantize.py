"""Post-training quantization and graph optimization for ONNX models.

Supports dynamic quantization (no calibration data needed), static INT8
quantization (requires a calibration dataset), and graph optimization
via HuggingFace Optimum ORTOptimizer.

Quantization uses onnxruntime.quantization directly (file-path based).
Quantization presets (avx512_vnni, avx2, arm64) are mapped to appropriate
onnxruntime.quantization settings with sensible built-in defaults.

Graph optimization uses optimum.onnxruntime.ORTOptimizer (optional).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

# DETR-family ops where INT8 emulation degrades accuracy without a real speedup
# on ORT CUDA EP (no native INT8 kernels for transformer attention/normalisation).
# Skipped from static quantization when an arch hint signals DETR-family. List is
# conservative — these are the ops where MinMax/percentile calibration most often
# collapses pred_boxes to NaN or all-zero scores on D-FINE / RT-DETRv2.
_DETR_OPS_EXCLUDE_TYPES: list[str] = [
    "LayerNormalization",
    "Softmax",
    "Gather",
]


# Paddle CNN-ish architectures known safe for INT8 ORT CUDA (no transformer
# attention emulation, real INT8 kernels available). Listed here so the
# DETR-family substring matcher below doesn't accidentally flag them.
_PADDLE_INT8_SAFE_PREFIXES: tuple[str, ...] = (
    "picodet-",
    "ppyoloe-",
    "ppclas-",
    "ppseg-",
    "pp-tinypose-",
)


def _is_paddle_arch(arch_hint: str | None) -> bool:
    """Return True for known PaddlePaddle CNN architectures.

    Mirrors `core.p09_export.export._is_paddle_arch` — duplicated to avoid a
    circular import between export.py (which imports ModelQuantizer from here)
    and quantize.py.
    """
    if not arch_hint:
        return False
    return arch_hint.lower().startswith(_PADDLE_INT8_SAFE_PREFIXES)


def _is_detr_family(arch_hint: str | None, onnx_path: str) -> bool:
    """Detect DETR-family from an explicit arch string or the ONNX filename.

    Paddle archs (picodet/ppyoloe/ppclas/ppseg/pp-tinypose) are explicitly
    excluded — they are CNN-ish and INT8 emulation is fine on ORT CUDA.
    """
    if _is_paddle_arch(arch_hint):
        return False
    candidates = []
    if arch_hint:
        candidates.append(arch_hint.lower())
    candidates.append(Path(onnx_path).stem.lower())
    return any("dfine" in c or "rtdetr" in c or "detr" in c for c in candidates)


_CALIBRATION_METHOD_MAP = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

# Quantization preset configurations: maps preset name to ORT quantization params.
_PRESET_CONFIGS = {
    "avx512_vnni": {
        "weight_type": QuantType.QInt8,
        "activation_type": QuantType.QUInt8,
        "per_channel": True,
        "quant_format": QuantFormat.QDQ,
    },
    "avx2": {
        "weight_type": QuantType.QInt8,
        "activation_type": QuantType.QUInt8,
        "per_channel": False,
        "quant_format": QuantFormat.QOperator,
    },
    "arm64": {
        "weight_type": QuantType.QUInt8,
        "activation_type": QuantType.QUInt8,
        "per_channel": False,
        "quant_format": QuantFormat.QDQ,
    },
}


def _require_optimum():
    """Import and return optimum modules, raising ImportError if missing."""
    try:
        from optimum.onnxruntime import ORTOptimizer
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        return ORTOptimizer, AutoOptimizationConfig
    except ImportError as err:
        raise ImportError(
            "Install optimum[onnxruntime]: pip install optimum[onnxruntime]"
        ) from err


def _resolve_preset(preset_name: str | None) -> dict:
    """Resolve a preset name to ORT quantization parameters.

    Args:
        preset_name: One of "avx512_vnni", "avx2", "arm64", or None.

    Returns:
        Dict with weight_type, activation_type, per_channel, quant_format.

    Raises:
        ValueError: If preset_name is not recognized.
    """
    if preset_name is None:
        return {
            "weight_type": QuantType.QInt8,
            "activation_type": QuantType.QInt8,
            "per_channel": False,
            "quant_format": QuantFormat.QDQ,
        }
    preset = _PRESET_CONFIGS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(_PRESET_CONFIGS.keys())}"
        )
    return preset


def _log_size_change(label: str, orig_path: str, new_path: str) -> None:
    """Log file size comparison between two ONNX models."""
    orig_mb = os.path.getsize(orig_path) / (1024 * 1024)
    new_mb = os.path.getsize(new_path) / (1024 * 1024)
    reduction = (1 - new_mb / orig_mb) * 100
    logger.info(
        "%s: %.2f MB -> %.2f MB (%.1f%% reduction)",
        label,
        orig_mb,
        new_mb,
        reduction,
    )


class _CalibrationDataReaderWrapper(CalibrationDataReader):
    """Wraps a PyTorch DataLoader as a CalibrationDataReader for ORT."""

    def __init__(self, data_loader, input_name: str, input_size: tuple, max_samples: int = 100):
        """Initialize the calibration reader.

        Args:
            data_loader: Iterable yielding (images, ...) batches.
                Images should be torch Tensors or numpy arrays of shape (B, C, H, W).
            input_name: ONNX model input name.
            input_size: Expected input size (H, W).
            max_samples: Maximum number of calibration samples to use.
        """
        self.data_loader = data_loader
        self.input_name = input_name
        self.input_size = input_size
        self.max_samples = max_samples
        self._iter = None
        self._count = 0

    def get_next(self):
        """Get the next calibration sample."""
        if self._iter is None:
            self._iter = iter(self.data_loader)

        if self._count >= self.max_samples:
            return None

        try:
            batch = next(self._iter)
        except StopIteration:
            return None

        # Handle both (images,) and (images, targets) batches
        images = batch[0] if isinstance(batch, (tuple, list)) else batch

        # Convert to numpy if needed
        if hasattr(images, "numpy"):
            images = images.numpy()

        # Use only the first image in batch for calibration
        if images.ndim == 4:
            images = images[:1]

        images = images.astype(np.float32)
        self._count += 1

        return {self.input_name: images}


class ModelQuantizer:
    """Post-training quantization and graph optimization for ONNX models.

    Supports:
    - Dynamic quantization (no calibration data needed)
    - Static INT8 quantization (requires calibration dataset)
    - Graph optimization via HuggingFace Optimum ORTOptimizer
    - Quantization presets: "avx512_vnni" (server), "avx2" (older x86), "arm64" (edge ARM)

    Args:
        onnx_path: Path to the ONNX model file.
        config: Optional config dict with quantization settings.
    """

    def __init__(self, onnx_path: str, config: dict | None = None):
        self.onnx_path = str(Path(onnx_path).resolve())
        self.config = config or {}

        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

    def quantize_dynamic(
        self,
        save_path: str | None = None,
        quantization_preset: str | None = None,
    ) -> str:
        """Apply dynamic quantization (weight-only, no calibration data).

        Quantizes weights to INT8, activations remain in float.
        Fast and easy -- good baseline for size reduction.

        Args:
            save_path: Path to save quantized model. If None, appends
                '_dynamic_int8' to the original filename.
            quantization_preset: Preset name ("avx512_vnni", "avx2", "arm64").
                Controls weight quantization type. If None, uses QInt8.

        Returns:
            Path to the quantized .onnx file.
        """
        if save_path is None:
            save_path = self._default_path("dynamic_int8")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        preset = _resolve_preset(quantization_preset)

        logger.info("Applying dynamic INT8 quantization...")
        logger.info("  Input:  %s", self.onnx_path)
        logger.info("  Output: %s", save_path)
        if quantization_preset:
            logger.info("  Preset: %s", quantization_preset)

        quantize_dynamic(
            model_input=self.onnx_path,
            model_output=save_path,
            weight_type=preset["weight_type"],
        )

        _log_size_change("Dynamic quantization complete", self.onnx_path, save_path)

        return str(Path(save_path).resolve())

    def quantize_static(
        self,
        calibration_loader,
        save_path: str | None = None,
        max_calibration_samples: int = 100,
        quantization_preset: str | None = None,
        calibration_method: str = "percentile",
        arch_hint: str | None = None,
        nodes_to_exclude: list[str] | None = None,
    ) -> str:
        """Apply static INT8 quantization with calibration data.

        Quantizes both weights and activations to INT8 using calibration
        data to determine optimal quantization ranges.

        Args:
            calibration_loader: DataLoader yielding (images, ...) batches.
            save_path: Path to save quantized model. If None, appends
                '_static_int8' to the original filename.
            max_calibration_samples: Max number of calibration samples.
            quantization_preset: Preset name ("avx512_vnni", "avx2", "arm64").
                Controls quantization types, per_channel, and format. If None,
                uses QDQ format with QInt8 weights/activations.

        Returns:
            Path to the quantized .onnx file.
        """
        if save_path is None:
            save_path = self._default_path("static_int8")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        preset = _resolve_preset(quantization_preset)

        # Calibration method gating: MinMax on DETR-family collapses mAP to ~0
        # (verified on D-FINE / RT-DETRv2). Refuse MinMax for DETR-family
        # rather than silently producing a broken model.
        calib_key = (calibration_method or "percentile").lower()
        if calib_key not in _CALIBRATION_METHOD_MAP:
            raise ValueError(
                f"Unknown calibration_method='{calibration_method}'. "
                f"Choose one of: {sorted(_CALIBRATION_METHOD_MAP)}"
            )
        is_detr = _is_detr_family(arch_hint, self.onnx_path)
        if calib_key == "minmax" and is_detr:
            raise ValueError(
                "MinMax calibration on DETR-family models collapses mAP to ~0 "
                "(see CLAUDE.md). Use calibration_method='percentile' (default) "
                "or 'entropy', and pass arch_hint to keep the DETR op-exclude "
                "list active."
            )
        # Auto-exclude transformer ops on DETR-family unless caller pinned a list.
        if nodes_to_exclude is None and is_detr:
            ops_excluded = list(_DETR_OPS_EXCLUDE_TYPES)
            logger.warning(
                f"DETR-family detected (arch_hint={arch_hint!r}); excluding op types "
                f"{ops_excluded} from INT8 quantization. ORT CUDA EP has no real "
                f"INT8 kernels for these ops; emulation is often slower than fp32."
            )
        else:
            ops_excluded = []

        logger.info("Applying static INT8 quantization...")
        logger.info("  Input:  %s", self.onnx_path)
        logger.info("  Output: %s", save_path)
        logger.info("  Calibration samples: %d", max_calibration_samples)
        logger.info("  Calibration method: %s", calib_key)
        if quantization_preset:
            logger.info("  Preset: %s", quantization_preset)

        # Get input name from model
        session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        input_size = (input_shape[2], input_shape[3]) if len(input_shape) >= 4 else (640, 640)
        del session

        # Create calibration reader
        calib_reader = _CalibrationDataReaderWrapper(
            data_loader=calibration_loader,
            input_name=input_name,
            input_size=input_size,
            max_samples=max_calibration_samples,
        )

        static_kwargs = dict(
            model_input=self.onnx_path,
            model_output=save_path,
            calibration_data_reader=calib_reader,
            quant_format=preset["quant_format"],
            weight_type=preset["weight_type"],
            activation_type=preset["activation_type"],
            calibrate_method=_CALIBRATION_METHOD_MAP[calib_key],
        )
        # Only pass nodes_to_exclude when something actually needs excluding —
        # ORT treats `op_types_to_quantize` and `nodes_to_exclude` as separate
        # filters; we use op-type-level exclusion via `op_types_to_exclude`
        # (pre-onnxruntime 1.19 named `nodes_to_exclude` accepts node names
        # only, so we route through the op-type-keyed `extra_options` dict
        # which is supported across versions).
        if ops_excluded:
            static_kwargs["extra_options"] = {"OpTypesToExcludeOutputQuantization": ops_excluded}
        if nodes_to_exclude:
            static_kwargs["nodes_to_exclude"] = list(nodes_to_exclude)

        quantize_static(**static_kwargs)

        _log_size_change("Static quantization complete", self.onnx_path, save_path)

        return str(Path(save_path).resolve())

    def optimize(
        self,
        save_path: str | None = None,
        level: str = "O2",
    ) -> str:
        """Apply graph optimization using HuggingFace Optimum ORTOptimizer.

        Performs operator fusion, constant folding, and other graph-level
        optimizations that reduce latency without changing numerical precision.

        Requires: pip install optimum[onnxruntime]

        Args:
            save_path: Path to save optimized model. If None, appends
                '_optimized' to the original filename.
            level: Optimization level ("O1", "O2", "O3", "O4").
                O1 = basic, O2 = extended (recommended), O3 = layout opt,
                O4 = all optimizations.

        Returns:
            Path to the optimized .onnx file.
        """
        ORTOptimizer, AutoOptimizationConfig = _require_optimum()

        if save_path is None:
            save_path = self._default_path("optimized")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Applying graph optimization (level=%s)...", level)
        logger.info("  Input:  %s", self.onnx_path)
        logger.info("  Output: %s", save_path)

        # ORTOptimizer works with directories containing model.onnx.
        # Copy the source model into a temp dir, run optimization, then
        # move the result to save_path.
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            tmp_model_path = Path(tmp_input_dir) / "model.onnx"
            shutil.copy2(self.onnx_path, tmp_model_path)

            optimizer = ORTOptimizer.from_pretrained(tmp_input_dir)
            optimization_config = AutoOptimizationConfig.with_optimization_level(level)

            with tempfile.TemporaryDirectory() as tmp_output_dir:
                optimizer.optimize(
                    save_dir=tmp_output_dir,
                    optimization_config=optimization_config,
                )

                # ORTOptimizer writes model.onnx in the output dir
                optimized_model = Path(tmp_output_dir) / "model.onnx"
                if not optimized_model.exists():
                    candidates = list(Path(tmp_output_dir).glob("*.onnx"))
                    if candidates:
                        optimized_model = candidates[0]
                    else:
                        raise RuntimeError(
                            f"ORTOptimizer produced no ONNX output in {tmp_output_dir}"
                        )

                shutil.copy2(optimized_model, save_path)

        _log_size_change("Graph optimization complete", self.onnx_path, save_path)

        return str(Path(save_path).resolve())

    def compare_accuracy(
        self,
        original_path: str,
        quantized_path: str,
        test_loader,
        max_samples: int = 200,
    ) -> dict:
        """Compare accuracy between original and quantized ONNX models.

        Runs both models on the same test data and reports output
        differences (not task-level accuracy -- use evaluation pipeline for that).

        Args:
            original_path: Path to original .onnx model.
            quantized_path: Path to quantized .onnx model.
            test_loader: DataLoader yielding (images, ...) batches.
            max_samples: Maximum number of samples to compare.

        Returns:
            Dictionary with comparison metrics:
                - mean_abs_diff: Mean absolute output difference.
                - max_abs_diff: Maximum absolute output difference.
                - mean_rel_diff: Mean relative output difference.
                - num_samples: Number of samples compared.
                - original_size_mb: Original model size.
                - quantized_size_mb: Quantized model size.
                - size_reduction_pct: Size reduction percentage.
        """
        orig_session = ort.InferenceSession(
            original_path, providers=["CPUExecutionProvider"]
        )
        quant_session = ort.InferenceSession(
            quantized_path, providers=["CPUExecutionProvider"]
        )

        input_name = orig_session.get_inputs()[0].name

        abs_diffs = []
        rel_diffs = []
        count = 0

        for batch in test_loader:
            if count >= max_samples:
                break

            images = batch[0] if isinstance(batch, (tuple, list)) else batch

            if hasattr(images, "numpy"):
                images = images.numpy()

            if images.ndim == 4:
                images = images[:1]

            images = images.astype(np.float32)

            orig_out = orig_session.run(None, {input_name: images})[0]
            quant_out = quant_session.run(None, {input_name: images})[0]

            diff = np.abs(orig_out - quant_out)
            abs_diffs.append(np.mean(diff))

            # Relative difference (avoid division by zero)
            denom = np.abs(orig_out) + 1e-8
            rel_diffs.append(np.mean(diff / denom))

            count += 1

        orig_size = os.path.getsize(original_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)

        results = {
            "mean_abs_diff": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
            "max_abs_diff": float(np.max(abs_diffs)) if abs_diffs else 0.0,
            "mean_rel_diff": float(np.mean(rel_diffs)) if rel_diffs else 0.0,
            "num_samples": count,
            "original_size_mb": round(orig_size, 2),
            "quantized_size_mb": round(quant_size, 2),
            "size_reduction_pct": round((1 - quant_size / orig_size) * 100, 1),
        }

        logger.info(
            "Accuracy comparison (%d samples): "
            "mean_abs_diff=%.6f, max_abs_diff=%.6f, size_reduction=%.1f%%",
            results["num_samples"],
            results["mean_abs_diff"],
            results["max_abs_diff"],
            results["size_reduction_pct"],
        )

        return results

    def _default_path(self, suffix: str) -> str:
        """Generate default save path with a suffix."""
        p = Path(self.onnx_path)
        return str(p.parent / f"{p.stem}_{suffix}{p.suffix}")
