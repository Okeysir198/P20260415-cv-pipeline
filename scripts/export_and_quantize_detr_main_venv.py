"""Export HF detection checkpoints to ONNX + static INT8 — all in the main venv
(transformers 5.5), bypassing optimum-onnx's transformers 4.57 pin.

Does three things per run_dir:
  1. Load the trained model via `_load_hf_model` (strips the `hf_model.` prefix
     added by our HFModelWrapper at save time).
  2. `torch.onnx.export` to `<run_dir>/model.onnx` with dynamic batch/H/W.
  3. Static INT8 quantization via `onnxruntime.quantization.quantize_static`
     using MinMax calibration on 32 CPPE-5 train images → `<run_dir>/int8_trt/
     model_quantized.onnx`.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run scripts/export_and_quantize_detr_main_venv.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForObjectDetection

REPO = Path(__file__).resolve().parent.parent

RUNS = [
    ("rtdetr_v2", REPO / "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42"),
    ("dfine_50ep", REPO / "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep"),
]

CALIB_DIR = REPO / "dataset_store/training_ready/cppe5/train/images"
EXPORT_SIZE = 640       # preprocessor default; picked up from preprocessor_config.json
N_CALIB = 32
OPSET = 17


def _load_hf_model(run_dir: Path) -> torch.nn.Module:
    """Load trained weights correctly — our HFModelWrapper prefixed every key
    with `hf_model.`; `AutoModelForObjectDetection.from_pretrained` silently
    re-inits otherwise."""
    config = AutoConfig.from_pretrained(run_dir)
    model = AutoModelForObjectDetection.from_config(config)
    orig = run_dir / "pytorch_model.bin.orig"
    curr = run_dir / "pytorch_model.bin"
    weight_path = orig if orig.exists() else curr
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    if any(k.startswith("hf_model.") for k in sd):
        sd = {k.removeprefix("hf_model."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{run_dir}: load failed — missing={len(missing)} unexpected={len(unexpected)} "
            f"(expected 0 under transformers 5.5 naming)"
        )
    return model.eval()


class _HFDetectionWrapper(torch.nn.Module):
    """Unwrap the ModelOutput dict — ONNX needs tuple outputs."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


class _CalibReader(CalibrationDataReader):
    def __init__(self, processor, image_paths: list[Path]):
        self._processor = processor
        self._paths = image_paths
        self._iter = iter(image_paths)

    def get_next(self):
        try:
            p = next(self._iter)
        except StopIteration:
            return None
        img = Image.open(p).convert("RGB")
        pv = self._processor(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
        return {"pixel_values": pv}

    def rewind(self):
        self._iter = iter(self._paths)


def export_onnx(model: torch.nn.Module, processor, dst: Path) -> None:
    print(f"  export_onnx → {dst.name} ...", flush=True)
    wrapped = _HFDetectionWrapper(model).eval()
    dummy = torch.randn(1, 3, EXPORT_SIZE, EXPORT_SIZE)
    with torch.inference_mode():
        # dynamo=False: the dynamo exporter doesn't support aten._is_all_true
        # used by transformers 5.5 internally. Legacy TorchScript exporter works.
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(dst),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                "logits": {0: "batch_size"},
                "pred_boxes": {0: "batch_size"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  {dst.stat().st_size / 1e6:.1f} MB")


def quantize(fp32_path: Path, int8_path: Path, processor) -> None:
    print(f"  quantize → {int8_path.name} ...", flush=True)
    # Symbolic shape inference fails on RT-DETRv2/D-FINE dynamic ops. Skip it —
    # MinMax calibration doesn't need shape info beyond what ORT infers at
    # session build time.

    random.seed(42)
    imgs = sorted(CALIB_DIR.glob("*.jpg"))
    calib_imgs = random.sample(imgs, min(N_CALIB, len(imgs)))
    reader = _CalibReader(processor, calib_imgs)

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  {int8_path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    for name, run_dir in RUNS:
        if not run_dir.exists():
            print(f"SKIP {name}: {run_dir} not found")
            continue
        print(f"\n=== {name} ({run_dir.name}) ===", flush=True)

        model = _load_hf_model(run_dir)
        processor = AutoImageProcessor.from_pretrained(run_dir)

        fp32_path = run_dir / "model.onnx"
        export_onnx(model, processor, fp32_path)

        int8_dir = run_dir / "int8_trt"
        int8_dir.mkdir(exist_ok=True)
        int8_path = int8_dir / "model_quantized.onnx"
        quantize(fp32_path, int8_path, processor)


if __name__ == "__main__":
    main()
