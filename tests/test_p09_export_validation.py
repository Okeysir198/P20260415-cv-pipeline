"""Test 13: Export Validation — numerical match between PyTorch and ONNX outputs."""

import sys
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p06_models.registry import build_model
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "13_export_validation"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAINING_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "08_training"
EXPORT_OUTPUTS = Path(__file__).resolve().parent / "outputs" / "12_export"
TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")


def _get_checkpoint_path() -> Path | None:
    for name in ["best.pth", "best.pt", "last.pth", "last.pt"]:
        p = TRAINING_OUTPUTS / name
        if p.exists():
            return p
    return None


def _load_model(ckpt_path: Path) -> torch.nn.Module:
    config = load_config(TRAIN_CONFIG_PATH)
    model = build_model(config)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_key = (
        "model_state_dict" if "model_state_dict" in ckpt
        else "model" if "model" in ckpt
        else "state_dict"
    )
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()
    return model


_has_checkpoint = _get_checkpoint_path() is not None
_skip_reason = "No checkpoint — run test_core13_training.py first"


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_onnx_pytorch_numerical_match():
    """Export model to ONNX, run same real input through both, compare outputs."""
    import onnx
    import onnxruntime as ort
    from fixtures import real_image_bgr_640

    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, f"No checkpoint in {TRAINING_OUTPUTS}"

    model = _load_model(ckpt_path)

    # Prepare real image input
    image_bgr = real_image_bgr_640(idx=0, split="val")
    image_rgb = image_bgr[:, :, ::-1].copy()  # BGR -> RGB
    image_chw = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalize
    input_np = np.expand_dims(image_chw, axis=0)  # (1, 3, 640, 640)
    input_tensor = torch.from_numpy(input_np)

    # PyTorch forward pass
    with torch.no_grad():
        pt_output = model(input_tensor)
        if isinstance(pt_output, (tuple, list)):
            pt_output = pt_output[0]
        pt_numpy = pt_output.cpu().float().numpy()

    # Export to ONNX
    onnx_path = str(OUTPUTS / "validation_model.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tensor,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )

    assert Path(onnx_path).exists(), f"ONNX export failed: {onnx_path}"
    onnx.checker.check_model(onnx_path)
    print(f"    Exported ONNX: {onnx_path}")

    # ONNX Runtime forward pass
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: input_np})
    ort_numpy = ort_outputs[0]

    # Numerical comparison
    print(f"    PyTorch output shape: {pt_numpy.shape}, dtype: {pt_numpy.dtype}")
    print(f"    ONNX output shape:    {ort_numpy.shape}, dtype: {ort_numpy.dtype}")

    assert pt_numpy.shape == ort_numpy.shape, (
        f"Shape mismatch: PyTorch {pt_numpy.shape} vs ONNX {ort_numpy.shape}"
    )

    max_abs_diff = np.max(np.abs(pt_numpy - ort_numpy))
    mean_abs_diff = np.mean(np.abs(pt_numpy - ort_numpy))
    print(f"    Max absolute diff:  {max_abs_diff:.6e}")
    print(f"    Mean absolute diff: {mean_abs_diff:.6e}")

    match = np.allclose(pt_numpy, ort_numpy, rtol=1e-2, atol=1e-3)
    assert match, (
        f"Numerical mismatch: max_abs_diff={max_abs_diff:.6e}, "
        f"mean_abs_diff={mean_abs_diff:.6e} (rtol=1e-2, atol=1e-3)"
    )
    print("    Numerical match: OK (rtol=1e-2, atol=1e-3)")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_onnx_deterministic_output():
    """Verify ONNX model produces identical outputs on repeated runs."""
    import onnxruntime as ort
    from fixtures import real_image_bgr_640

    # Use previously exported model, or export fresh
    onnx_path = OUTPUTS / "validation_model.onnx"
    if not onnx_path.exists():
        ckpt_path = _get_checkpoint_path()
        assert ckpt_path is not None
        model = _load_model(ckpt_path)
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
            )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Prepare real image input
    image_bgr = real_image_bgr_640(idx=0, split="val")
    image_rgb = image_bgr[:, :, ::-1].copy()
    input_np = np.expand_dims(
        image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0
    )

    # Run 3 times and compare
    outputs = []
    for run_idx in range(3):
        result = session.run(None, {input_name: input_np})
        outputs.append(result[0])

    for i in range(1, len(outputs)):
        assert np.array_equal(outputs[0], outputs[i]), (
            f"Run 0 vs run {i}: outputs differ (max diff = "
            f"{np.max(np.abs(outputs[0] - outputs[i])):.6e})"
        )
    print("    ONNX deterministic: 3 runs produce identical outputs")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_onnx_different_images():
    """Verify ONNX produces different outputs for different real images."""
    import onnxruntime as ort
    from fixtures import real_image_bgr_640

    onnx_path = OUTPUTS / "validation_model.onnx"
    if not onnx_path.exists():
        ckpt_path = _get_checkpoint_path()
        assert ckpt_path is not None
        model = _load_model(ckpt_path)
        dummy = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
            )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Two different real images
    img1 = real_image_bgr_640(idx=0, split="train")
    img2 = real_image_bgr_640(idx=1, split="train")

    def preprocess(img_bgr: np.ndarray) -> np.ndarray:
        rgb = img_bgr[:, :, ::-1].copy()
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)

    out1 = session.run(None, {input_name: preprocess(img1)})[0]
    out2 = session.run(None, {input_name: preprocess(img2)})[0]

    assert not np.array_equal(out1, out2), (
        "Different images should produce different outputs"
    )
    max_diff = np.max(np.abs(out1 - out2))
    print(f"    Different images produce different outputs (max diff: {max_diff:.6e})")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_pytorch_deterministic_eval():
    """Verify PyTorch model in eval mode produces identical outputs on repeated runs."""
    from fixtures import real_image_bgr_640

    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None
    model = _load_model(ckpt_path)

    image_bgr = real_image_bgr_640(idx=0, split="val")
    image_rgb = image_bgr[:, :, ::-1].copy()
    input_np = np.expand_dims(
        image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0
    )
    input_tensor = torch.from_numpy(input_np)

    outputs = []
    for _ in range(3):
        with torch.no_grad():
            out = model(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs.append(out.cpu().float().numpy())

    for i in range(1, len(outputs)):
        assert np.array_equal(outputs[0], outputs[i]), (
            f"PyTorch eval mode: run 0 vs run {i} differ "
            f"(max diff = {np.max(np.abs(outputs[0] - outputs[i])):.6e})"
        )
    print("    PyTorch eval deterministic: 3 runs produce identical outputs")


@pytest.mark.skipif(not _has_checkpoint, reason=_skip_reason)
def test_onnx_pytorch_detection_match():
    """Compare PyTorch vs ONNX at detection level (boxes, scores, labels) after postprocessing."""
    import onnxruntime as ort
    from fixtures import real_image_bgr_640
    from core.p06_training.postprocess import postprocess

    ckpt_path = _get_checkpoint_path()
    assert ckpt_path is not None, f"No checkpoint in {TRAINING_OUTPUTS}"

    model = _load_model(ckpt_path)

    # Prepare real image input
    image_bgr = real_image_bgr_640(idx=0, split="val")
    image_rgb = image_bgr[:, :, ::-1].copy()
    image_chw = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_np = np.expand_dims(image_chw, axis=0)  # (1, 3, 640, 640)
    input_tensor = torch.from_numpy(input_np)

    # PyTorch forward pass
    with torch.no_grad():
        pt_output = model(input_tensor)
        if isinstance(pt_output, (tuple, list)):
            pt_raw = pt_output[0]
        else:
            pt_raw = pt_output

    # Export to ONNX (reuse if already exported)
    onnx_path = OUTPUTS / "validation_model.onnx"
    if not onnx_path.exists():
        with torch.no_grad():
            torch.onnx.export(
                model, input_tensor, str(onnx_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
            )
    assert onnx_path.exists(), f"ONNX model not found: {onnx_path}"

    # ONNX Runtime forward pass
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: input_np})
    ort_raw = torch.from_numpy(ort_outputs[0])

    # Postprocess both outputs into detections
    output_format = model.output_format
    conf_threshold = 0.3
    nms_threshold = 0.45

    pt_dets = postprocess(output_format, model, pt_raw, conf_threshold, nms_threshold)
    ort_dets = postprocess(output_format, model, ort_raw, conf_threshold, nms_threshold)

    assert len(pt_dets) == len(ort_dets), (
        f"Batch size mismatch: PyTorch {len(pt_dets)} vs ONNX {len(ort_dets)}"
    )

    for img_idx in range(len(pt_dets)):
        pt_det = pt_dets[img_idx]
        ort_det = ort_dets[img_idx]

        pt_boxes = pt_det["boxes"]
        pt_scores = pt_det["scores"]
        pt_labels = pt_det["labels"]

        ort_boxes = ort_det["boxes"]
        ort_scores = ort_det["scores"]
        ort_labels = ort_det["labels"]

        n_pt = len(pt_scores)
        n_ort = len(ort_scores)
        print(f"    Image {img_idx}: PyTorch detections={n_pt}, ONNX detections={n_ort}")

        assert n_pt == n_ort, (
            f"Image {img_idx}: detection count mismatch: "
            f"PyTorch {n_pt} vs ONNX {n_ort} (conf > {conf_threshold})"
        )

        if n_pt == 0:
            print(f"    Image {img_idx}: no detections above conf {conf_threshold}")
            continue

        # Compare class labels (must match exactly)
        np.testing.assert_array_equal(
            pt_labels, ort_labels,
            err_msg=f"Image {img_idx}: label mismatch between PyTorch and ONNX",
        )

        # Compare box coordinates (within 2 pixels tolerance)
        np.testing.assert_allclose(
            pt_boxes, ort_boxes, atol=2.0,
            err_msg=f"Image {img_idx}: box coordinate mismatch (atol=2 pixels)",
        )

        # Compare scores (loose tolerance for floating point differences)
        np.testing.assert_allclose(
            pt_scores, ort_scores, atol=0.01,
            err_msg=f"Image {img_idx}: score mismatch",
        )

        print(f"    Image {img_idx}: labels match, boxes within 2px, scores within 0.01")

    print("    Detection-level match: OK")


if __name__ == "__main__":
    run_all([
        ("onnx_pytorch_numerical_match", test_onnx_pytorch_numerical_match),
        ("onnx_deterministic_output", test_onnx_deterministic_output),
        ("onnx_different_images", test_onnx_different_images),
        ("pytorch_deterministic_eval", test_pytorch_deterministic_eval),
        ("onnx_pytorch_detection_match", test_onnx_pytorch_detection_match),
    ], title="Test 12: Export Validation (Numerical Match)")
