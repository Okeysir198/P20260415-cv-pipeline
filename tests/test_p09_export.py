"""Test 12: Export — export trained model to ONNX, validate."""

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.quantization
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p09_export.benchmark import ModelBenchmark
from core.p09_export.exporter import ModelExporter
from core.p06_models import build_model
from core.p09_export.quantize import ModelQuantizer
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "12_export"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")
EXPORT_CONFIG_PATH = str(ROOT / "configs" / "_shared" / "09_export.yaml")


def _get_checkpoint_path():
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            return p
    return None


def _load_model(ckpt_path):
    config = load_config(TRAIN_CONFIG_PATH)
    model = build_model(config)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()
    return model


def test_export_onnx():
    """Export trained model to ONNX."""
    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    model = _load_model(ckpt_path)
    export_config = load_config(EXPORT_CONFIG_PATH)
    export_config["output_dir"] = str(OUTPUTS)

    exporter = ModelExporter(model, export_config, model_name="test_fire")
    onnx_path = exporter.export_onnx(save_path=str(OUTPUTS / "model.onnx"))

    assert Path(onnx_path).exists(), f"ONNX file not found: {onnx_path}"
    print(f"    Exported: {onnx_path}")

    # Validate with onnx checker
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"    ONNX checker passed")


def test_validate_onnx():
    """Validate ONNX output matches PyTorch."""
    onnx_path = OUTPUTS / "model.onnx"
    assert onnx_path.exists(), "model.onnx not found — test_export_onnx must pass first"

    ckpt_path = _get_checkpoint_path()
    model = _load_model(ckpt_path)
    export_config = load_config(EXPORT_CONFIG_PATH)
    export_config["output_dir"] = str(OUTPUTS)

    exporter = ModelExporter(model, export_config, model_name="test_fire")
    valid = exporter.validate_onnx(str(onnx_path))
    assert valid, "ONNX validation failed — outputs don't match PyTorch"
    print(f"    ONNX-PyTorch output match: OK")


def test_onnx_inference_shape():
    """Run ONNX inference on a real image and check output shape."""
    onnx_path = OUTPUTS / "model.onnx"
    assert onnx_path.exists(), "model.onnx not found"

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"    ONNX input: {input_name}, shape: {input_shape}")

    # Use a real image instead of random noise
    from fixtures import real_image_bgr_640
    image = real_image_bgr_640(idx=0, split="val")
    image = image[:, :, ::-1].copy()  # BGR -> RGB
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalize
    image = np.expand_dims(image, axis=0)  # add batch dim

    outputs = session.run(None, {input_name: image})
    print(f"    ONNX output shapes: {[o.shape for o in outputs]}")


def test_quantize_dynamic():
    """Apply dynamic INT8 quantization to exported ONNX model."""
    onnx_path = OUTPUTS / "model.onnx"
    assert onnx_path.exists(), "model.onnx not found — test_export_onnx must pass first"

    quantized_path = OUTPUTS / "model_dynamic_int8.onnx"
    quantizer = ModelQuantizer(str(onnx_path))
    result_path = quantizer.quantize_dynamic(save_path=str(quantized_path))

    assert Path(result_path).exists(), f"Quantized model not found: {result_path}"
    print(f"    Quantized model saved: {result_path}")

    # Verify quantized model is loadable by onnxruntime with a real image
    session = ort.InferenceSession(result_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    from fixtures import real_image_bgr_640
    real_img = real_image_bgr_640(idx=0, split="val")[:, :, ::-1].copy()
    real_input = np.expand_dims(real_img.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
    outputs = session.run(None, {input_name: real_input})
    assert len(outputs) > 0, "Quantized model produced no outputs"
    print(f"    Quantized model inference OK, output shapes: {[o.shape for o in outputs]}")

    # Compare sizes
    orig_size = os.path.getsize(str(onnx_path)) / (1024 * 1024)
    quant_size = os.path.getsize(result_path) / (1024 * 1024)
    print(f"    Size: {orig_size:.2f} MB -> {quant_size:.2f} MB ({(1 - quant_size/orig_size)*100:.1f}% reduction)")


def test_benchmark_pytorch():
    """Benchmark PyTorch model inference latency and throughput."""
    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    model = _load_model(ckpt_path)

    benchmarker = ModelBenchmark(
        input_size=(640, 640),
        warmup_runs=3,
        num_runs=10,
        device="cpu",
    )
    result = benchmarker.benchmark_pytorch(model)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "latency_mean_ms" in result, f"Missing 'latency_mean_ms' in result keys: {list(result.keys())}"
    assert "throughput_fps" in result, f"Missing 'throughput_fps' in result keys: {list(result.keys())}"
    assert result["latency_mean_ms"] > 0, "Latency should be positive"
    assert result["throughput_fps"] > 0, "FPS should be positive"

    print(f"    Latency: {result['latency_mean_ms']:.2f} ms (std {result['latency_std_ms']:.2f})")
    print(f"    Throughput: {result['throughput_fps']:.1f} FPS")
    print(f"    Model size: {result['model_size_mb']:.2f} MB")


def test_export_onnx_dynamo():
    """Export YOLOX model using torch.onnx.export dynamo path (or fallback)."""
    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, (
        f"No checkpoint in {TRAINING_OUTPUTS}. Run test_core13_training.py first."
    )

    model = _load_model(ckpt_path)  # already calls model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    dynamo_path = str(OUTPUTS / "model_dynamo.onnx")

    # Try dynamo export (torch >= 2.1), fall back to standard export
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                dynamo_path,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamo=True,
            )
        print("    Used dynamo=True export path")
    except Exception as e:
        print(f"    Dynamo export not available ({e}), falling back to standard export")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                dynamo_path,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
            )
        print("    Used standard torch.onnx.export fallback")

    assert Path(dynamo_path).exists(), f"ONNX file not found: {dynamo_path}"

    # Validate with onnx checker
    onnx_model = onnx.load(dynamo_path)
    onnx.checker.check_model(onnx_model)
    print(f"    ONNX checker passed: {dynamo_path}")

    size_mb = os.path.getsize(dynamo_path) / (1024 * 1024)
    print(f"    Model size: {size_mb:.2f} MB")


def _export_hf_model_optimum(arch: str, pretrained: str, model_name: str, onnx_filename: str) -> None:
    """Export an HF model via Optimum path. Shared helper for all HF export tests."""
    try:
        import optimum  # noqa: F401
    except ImportError:
        print("    SKIP: optimum not installed")
        return

    config = {
        "model": {
            "arch": arch,
            "num_classes": 2,
            "pretrained": pretrained,
            "ignore_mismatched_sizes": True,
        }
    }
    model = build_model(config)
    export_config = load_config(EXPORT_CONFIG_PATH)
    export_config["output_dir"] = str(OUTPUTS)

    exporter = ModelExporter(model, export_config, model_name=model_name)
    onnx_path = exporter.export_onnx(save_path=str(OUTPUTS / onnx_filename))

    assert Path(onnx_path).exists(), f"ONNX file not found: {onnx_path}"
    print(f"    Exported {model_name}: {onnx_path}")
    print(f"    Model size: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")


def test_export_hf_model_optimum():
    """Export HF model (D-FINE-N) via Optimum path."""
    _export_hf_model_optimum("dfine-n", "ustc-community/dfine_n_coco", "test_dfine", "dfine_model.onnx")


def test_export_dfine_s_optimum():
    """Export D-FINE-S via Optimum path."""
    _export_hf_model_optimum("dfine-s", "ustc-community/dfine_s_coco", "test_dfine_s", "dfine_s_model.onnx")


def test_export_rtdetrv2_r18_optimum():
    """Export RT-DETRv2-R18 via Optimum path."""
    _export_hf_model_optimum("rtdetrv2-r18", "PekingU/rtdetr_v2_r18vd", "test_rtdetrv2_r18", "rtdetrv2_r18_model.onnx")


def _run_hf_onnx_inference(onnx_path: str, label: str) -> None:
    """Load an HF-exported ONNX model, run a forward pass, and assert outputs are valid."""
    # Validate ONNX graph structure (string path avoids loading full proto into memory)
    onnx.checker.check_model(onnx_path)
    print(f"    [{label}] ONNX checker passed")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"    [{label}] Inputs:  {[(i.name, i.shape) for i in inputs]}")
    print(f"    [{label}] Outputs: {[(o.name, o.shape) for o in outputs]}")

    # Build feed dict — use the model's declared shape, fall back to 640×640
    feed = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        # Force batch=1, keep C/H/W; if H/W are dynamic use 640
        if len(shape) == 4:
            shape[0] = 1
            if shape[2] <= 0:
                shape[2] = 640
            if shape[3] <= 0:
                shape[3] = 640
        feed[inp.name] = np.random.rand(*shape).astype(np.float32)

    results = session.run(None, feed)
    assert len(results) > 0, f"[{label}] Model produced no outputs"
    for i, (out_meta, out_val) in enumerate(zip(outputs, results)):
        print(f"    [{label}] Output[{i}] '{out_meta.name}': shape={out_val.shape}, dtype={out_val.dtype}")
    print(f"    [{label}] Inference OK")


def test_onnx_inference_dfine_n():
    """Run ONNX inference on the exported D-FINE-N model."""
    onnx_path = str(OUTPUTS / "dfine_model.onnx")
    if not Path(onnx_path).exists():
        print("    SKIP: dfine_model.onnx not found — run test_export_hf_model_optimum first")
        return
    _run_hf_onnx_inference(onnx_path, "D-FINE-N")


def test_onnx_inference_dfine_s():
    """Run ONNX inference on the exported D-FINE-S model."""
    onnx_path = str(OUTPUTS / "dfine_s_model.onnx")
    if not Path(onnx_path).exists():
        print("    SKIP: dfine_s_model.onnx not found — run test_export_dfine_s_optimum first")
        return
    _run_hf_onnx_inference(onnx_path, "D-FINE-S")


def test_onnx_inference_rtdetrv2_r18():
    """Run ONNX inference on the exported RT-DETRv2-R18 model."""
    onnx_path = str(OUTPUTS / "rtdetrv2_r18_model.onnx")
    if not Path(onnx_path).exists():
        print("    SKIP: rtdetrv2_r18_model.onnx not found — run test_export_rtdetrv2_r18_optimum first")
        return
    _run_hf_onnx_inference(onnx_path, "RT-DETRv2-R18")


def test_optimize_onnx():
    """Apply graph optimization to exported ONNX model via ORT session options."""
    onnx_path = OUTPUTS / "model.onnx"
    if not onnx_path.exists():
        print("    SKIP: model.onnx not found — test_export_onnx must pass first")
        return

    optimized_path = OUTPUTS / "model_optimized.onnx"

    # Use ORT session options to apply O1 graph optimizations and save
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = str(optimized_path)

    # Creating the session triggers optimization and saves the optimized graph
    ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    assert optimized_path.exists(), f"Optimized model not found: {optimized_path}"
    print(f"    Optimized model saved: {optimized_path}")

    orig_size = os.path.getsize(str(onnx_path)) / (1024 * 1024)
    opt_size = os.path.getsize(str(optimized_path)) / (1024 * 1024)
    print(f"    Size: {orig_size:.2f} MB -> {opt_size:.2f} MB")


if __name__ == "__main__":
    run_all([
        # Core tests (require training checkpoint)
        ("export_onnx", test_export_onnx),
        ("validate_onnx", test_validate_onnx),
        ("onnx_inference_shape", test_onnx_inference_shape),
        ("quantize_dynamic", test_quantize_dynamic),
        ("benchmark_pytorch", test_benchmark_pytorch),
        ("export_onnx_dynamo", test_export_onnx_dynamo),
        ("optimize_onnx", test_optimize_onnx),
        # Optional tests (require optimum + HF model download)
        ("export_hf_model_optimum", test_export_hf_model_optimum),
        ("export_dfine_s_optimum", test_export_dfine_s_optimum),
        ("export_rtdetrv2_r18_optimum", test_export_rtdetrv2_r18_optimum),
        # ONNX inference validation for HF-exported models
        ("onnx_inference_dfine_n", test_onnx_inference_dfine_n),
        ("onnx_inference_dfine_s", test_onnx_inference_dfine_s),
        ("onnx_inference_rtdetrv2_r18", test_onnx_inference_rtdetrv2_r18),
    ], title="Test 07: Export")
