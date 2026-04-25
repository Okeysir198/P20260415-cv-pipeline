"""Export PyTorch detection models to ONNX format for edge deployment.

Dual-path export strategy:
- HF models (D-FINE, RT-DETRv2): uses HuggingFace Optimum ``main_export()``
- YOLOX/custom models: uses ``torch.onnx.export(dynamo=True)`` with onnxsim

Handles ONNX export, optional simplification (onnxsim), and output
validation against the original PyTorch model.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify as onnxsim_simplify
from thop import profile

from core.p06_models.hf_model import HFDetectionModel

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

_OUTPUT_FORMAT_TO_EXPORT_TASK = {
    "classification": "image-classification",
    "segmentation": "semantic-segmentation",
}


def _generate_filename(
    naming_template: str,
    model: str,
    arch: str,
    imgsz: int,
    version: str,
) -> str:
    """Generate filename from a naming template.

    Args:
        naming_template: Template string with {model}, {arch}, {imgsz}, {version}.
        model: Model name (e.g. "fire").
        arch: Architecture name (e.g. "yolox-m"), sanitized to remove dashes.
        imgsz: Input image size (e.g. 640).
        version: Version string (e.g. "1").

    Returns:
        Formatted filename string (without extension).
    """
    arch_clean = arch.replace("-", "").replace("_", "")
    return naming_template.format(
        model=model,
        arch=arch_clean,
        imgsz=imgsz,
        version=version,
    )


class ModelExporter:
    """Export a PyTorch detection model to ONNX format.

    Uses HuggingFace Optimum for HF models (D-FINE, RT-DETRv2) and
    ``torch.onnx.export(dynamo=True)`` for custom models (YOLOX).

    Args:
        model: Trained PyTorch model (nn.Module).
        config: Export config dict (from default.yaml).
        model_name: Name for the exported model (used in filename).
        version: Model version string.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        model_name: str = "model",
        version: str = "1",
    ):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.version = version

        self.input_size = tuple(config.get("input_size", [640, 640]))
        self.opset = config.get("opset", 11)
        self.simplify = config.get("simplify", True)
        self.dynamic_batch = config.get("dynamic_batch", False)
        self.fp16 = config.get("fp16", False)
        self.naming_template = config.get("naming", "{model}_{arch}_{imgsz}_v{version}")
        self.output_dir = config.get("output_dir", ".")

    def _resolve_save_path(self, save_path: str | None) -> str:
        """Resolve the ONNX save path from explicit path or naming template.

        Args:
            save_path: Explicit path, or None to auto-generate.

        Returns:
            Resolved absolute path string for the .onnx file.
        """
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            return save_path

        arch = self._detect_arch()
        filename = _generate_filename(
            self.naming_template,
            model=self.model_name,
            arch=arch,
            imgsz=self.input_size[0],
            version=self.version,
        )
        save_dir = Path(self.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir / f"{filename}.onnx")

    def export(
        self,
        save_path: str | None = None,
        fmt: str = "onnx",
    ) -> str:
        """Export the model to the specified format.

        Args:
            save_path: Optional explicit save path.
            fmt: Export format — ``"onnx"`` (default), ``"tensorrt"``, or
                ``"openvino"``. TensorRT and OpenVINO require the ONNX
                file to exist first (exports ONNX as an intermediate step).

        Returns:
            Absolute path to the exported model file.

        Raises:
            ImportError: If the required export library is not installed.
            ValueError: If an unsupported format is requested.
        """
        fmt = fmt.lower()
        if fmt == "onnx":
            return self.export_onnx(save_path)
        elif fmt == "tensorrt":
            return self._export_tensorrt(save_path)
        elif fmt == "openvino":
            return self._export_openvino(save_path)
        else:
            raise ValueError(
                f"Unsupported export format '{fmt}'. "
                f"Available: onnx, tensorrt, openvino"
            )

    def export_onnx(self, save_path: str | None = None) -> str:
        """Export the model to ONNX format.

        Automatically selects the export backend:
        - HF models: ``optimum.exporters.onnx.main_export()``
        - Custom models: ``torch.onnx.export(dynamo=True)`` with onnxsim fallback

        Args:
            save_path: Optional explicit save path. If None, generates from
                naming template and output_dir.

        Returns:
            Absolute path to the saved .onnx file.

        Raises:
            ImportError: If optimum is not installed when exporting HF models.
        """
        from core.p06_models.hf_model import HFClassificationModel, HFSegmentationModel
        if isinstance(self.model, (HFDetectionModel, HFClassificationModel, HFSegmentationModel)):
            return self._export_hf(save_path)
        return self._export_custom(save_path)

    def _export_hf(self, save_path: str | None = None) -> str:
        """Export an HF model via Optimum.

        Args:
            save_path: Optional explicit save path for the .onnx file.

        Returns:
            Absolute path to the saved .onnx file.

        Raises:
            ImportError: If ``optimum`` is not installed.
        """
        try:
            from optimum.exporters.onnx import main_export
        except ImportError as err:
            raise ImportError(
                "optimum is required for exporting HF models. "
                "Install it with: pip install optimum[exporters]"
            ) from err

        onnx_final = Path(self._resolve_save_path(save_path))
        save_dir = onnx_final.parent / f"{onnx_final.stem}_optimum"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting HF model via Optimum to: %s", save_dir)

        hf_model = self.model.hf_model
        device = next(hf_model.parameters()).device

        main_export(
            model_name_or_path=hf_model,
            output=save_dir,
            task=self._hf_export_task(),
            opset=self.opset,
            device=str(device),
        )

        # Optimum writes model.onnx inside save_dir -- move to final path
        optimum_onnx = save_dir / "model.onnx"
        if not optimum_onnx.exists():
            # Fallback: find any .onnx file in save_dir
            onnx_files = list(save_dir.glob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(
                    f"Optimum export completed but no .onnx file found in {save_dir}"
                )
            optimum_onnx = onnx_files[0]

        onnx_final.parent.mkdir(parents=True, exist_ok=True)
        optimum_onnx.rename(onnx_final)
        shutil.rmtree(save_dir, ignore_errors=True)

        size_mb = os.path.getsize(onnx_final) / (1024 * 1024)
        logger.info("HF model exported: %s (%.2f MB)", onnx_final, size_mb)

        return str(onnx_final.resolve())

    def _hf_export_task(self) -> str:
        """Map model output_format to Optimum export task string."""
        fmt = getattr(self.model, "output_format", "yolox")
        return _OUTPUT_FORMAT_TO_EXPORT_TASK.get(fmt, "object-detection")

    def _export_custom(self, save_path: str | None = None) -> str:
        """Export a custom (YOLOX) model via torch.onnx.export.

        Tries ``dynamo=True`` first (PyTorch >= 2.5), falls back to legacy export.

        Args:
            save_path: Optional explicit save path for the .onnx file.

        Returns:
            Absolute path to the saved .onnx file.
        """
        save_path = self._resolve_save_path(save_path)
        logger.info("Exporting model to ONNX: %s", save_path)

        # Prepare model
        self.model.eval()
        for module in self.model.modules():
            if hasattr(module, "_grid_cache"):
                module._grid_cache.clear()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, *self.input_size, device=device)

        if self.fp16 and device.type == "cuda":
            self.model = self.model.half()
            dummy_input = dummy_input.half()
            logger.info("Exporting in FP16 mode")

        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        input_names = ["input"]
        output_names = ["output"]
        opset = max(self.opset, 18)  # dynamo needs >= 18

        with torch.no_grad():
            try:
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    save_path,
                    opset_version=opset,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    dynamo=True,
                )
                logger.info("ONNX export (dynamo) complete: %s", save_path)
            except Exception as e:
                logger.info(
                    "dynamo=True export failed (%s), falling back to legacy export", e
                )
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    save_path,
                    opset_version=self.opset,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                )
                logger.info("ONNX export (legacy) complete: %s", save_path)

        # Verify the exported model loads
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model check passed")

        if self.simplify:
            save_path = self._simplify_onnx(save_path)

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info("Exported model size: %.2f MB", size_mb)

        return str(Path(save_path).resolve())

    def validate_onnx(
        self, onnx_path: str, rtol: float = 1e-3, atol: float = 1e-5
    ) -> bool:
        """Validate ONNX model output matches PyTorch output.

        Runs the same input through both models and compares outputs
        element-wise within tolerance.

        Args:
            onnx_path: Path to the .onnx model file.
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.

        Returns:
            True if outputs match within tolerance.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, *self.input_size, device=device)

        if self.fp16 and device.type == "cuda":
            self.model = self.model.half()
            dummy_input = dummy_input.half()

        with torch.no_grad():
            pt_output = self.model(dummy_input)
            if isinstance(pt_output, (tuple, list)):
                pt_output = pt_output[0]
            pt_numpy = pt_output.cpu().float().numpy()

        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        ort_input = dummy_input.cpu().float().numpy()
        ort_output = session.run(None, {input_name: ort_input})[0]

        try:
            np.testing.assert_allclose(pt_numpy, ort_output, rtol=rtol, atol=atol)
            logger.info(
                "Validation PASSED: PyTorch and ONNX outputs match "
                "(rtol=%.1e, atol=%.1e)",
                rtol,
                atol,
            )
            return True
        except AssertionError as e:
            max_diff = np.max(np.abs(pt_numpy - ort_output))
            logger.warning(
                "Validation FAILED: max absolute diff = %.6f. %s", max_diff, e
            )
            return False

    def get_model_info(self) -> dict:
        """Get model information: parameter count, size, FLOPs estimate.

        Returns:
            Dictionary with keys:
                - total_params: Total number of parameters.
                - trainable_params: Number of trainable parameters.
                - model_size_mb: Estimated model size in MB (float32).
                - input_size: (H, W) tuple.
                - flops_estimate: Estimated FLOPs (if thop available), else None.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        size_mb = total_params * 4 / (1024 * 1024)

        info = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(size_mb, 2),
            "input_size": self.input_size,
            "flops_estimate": None,
        }

        try:
            device = next(self.model.parameters()).device
            dummy = torch.randn(1, 3, *self.input_size, device=device)
            flops, _ = profile(self.model, inputs=(dummy,), verbose=False)
            info["flops_estimate"] = flops
        except Exception as e:
            logger.debug("FLOPs estimation failed: %s", e)

        return info

    def _detect_arch(self) -> str:
        """Detect architecture name from model property or class name."""
        if hasattr(self.model, 'output_format'):
            return self.model.output_format
        cls_name = type(self.model).__name__.lower()
        if "yolox" in cls_name:
            return cls_name
        return "model"

    def _export_tensorrt(self, save_path: str | None = None) -> str:
        """Export to TensorRT engine via ONNX intermediate.

        Requires ``tensorrt`` Python package.

        Args:
            save_path: Optional explicit save path for the .engine file.

        Returns:
            Absolute path to the saved .engine file.
        """
        try:
            import tensorrt as trt  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "TensorRT is required for TensorRT export. "
                "Install it with: pip install tensorrt"
            ) from err

        # Export ONNX first as intermediate
        onnx_path = self._resolve_save_path(None)
        if not Path(onnx_path).exists():
            onnx_path = self.export_onnx()

        # Resolve TensorRT save path
        if save_path is None:
            save_path = str(Path(onnx_path).with_suffix(".engine"))

        logger.info("Building TensorRT engine from ONNX: %s", onnx_path)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TRT parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX for TensorRT")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 30  # 1 GB
        )

        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TensorRT FP16 mode enabled")

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(engine_bytes)

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info("TensorRT engine saved: %s (%.2f MB)", save_path, size_mb)
        return str(Path(save_path).resolve())

    def _export_openvino(self, save_path: str | None = None) -> str:
        """Export to OpenVINO IR format via ONNX intermediate.

        Requires ``openvino`` Python package.

        Args:
            save_path: Optional explicit save path for the .xml file.

        Returns:
            Absolute path to the saved .xml file.
        """
        try:
            import openvino as ov
        except ImportError as err:
            raise ImportError(
                "OpenVINO is required for OpenVINO export. "
                "Install it with: pip install openvino"
            ) from err

        # Export ONNX first as intermediate
        onnx_path = self._resolve_save_path(None)
        if not Path(onnx_path).exists():
            onnx_path = self.export_onnx()

        if save_path is None:
            save_path = str(Path(onnx_path).with_suffix(".xml"))

        logger.info("Converting ONNX to OpenVINO IR: %s", onnx_path)

        core = ov.Core()
        ov_model = core.read_model(onnx_path)

        if self.fp16:
            from openvino.tools.mo import convert_model

            ov_model = convert_model(
                onnx_path, compress_to_fp16=True
            )
            logger.info("OpenVINO FP16 compression enabled")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov_model, save_path)

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info("OpenVINO IR saved: %s (%.2f MB)", save_path, size_mb)
        return str(Path(save_path).resolve())

    def _simplify_onnx(self, onnx_path: str) -> str:
        """Simplify the ONNX model with onnx-simplifier.

        Args:
            onnx_path: Path to the original .onnx file.

        Returns:
            Path to the simplified .onnx file (same path, overwritten).
        """
        logger.info("Simplifying ONNX model...")
        model = onnx.load(onnx_path)
        try:
            simplified, ok = onnxsim_simplify(model)
            if ok:
                onnx.save(simplified, onnx_path)
                logger.info("ONNX simplification successful")
            else:
                logger.warning("ONNX simplification returned ok=False, keeping original")
        except Exception as e:
            logger.warning("ONNX simplification failed: %s. Keeping original.", e)

        return onnx_path
