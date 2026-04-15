"""Test: End-to-End Pipeline — train → evaluate → export → inference in one flow.

Validates hand-offs between pipeline stages using the test_fire_100 dataset.
Depends on the same test configs as other p06+ tests.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import run_all
from utils.config import load_config

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "15_e2e_pipeline"
OUTPUTS.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = str(ROOT / "configs" / "_test" / "06_training.yaml")
DATA_CONFIG_PATH = str(ROOT / "configs" / "_test" / "05_data.yaml")


def test_e2e_train_eval_export_infer():
    """Full pipeline: train 2 epochs → evaluate → export ONNX → ONNX inference."""
    import onnx
    import onnxruntime as ort
    from fixtures import real_image_bgr_640

    from core.p06_training.trainer import DetectionTrainer
    from core.p06_models import build_model
    from core.p08_evaluation.evaluator import ModelEvaluator

    # -----------------------------------------------------------------------
    # Step 1: Train for 2 epochs
    # -----------------------------------------------------------------------
    print("  [1/4] Training 2 epochs...")
    save_dir = str(OUTPUTS / "runs")
    trainer = DetectionTrainer(
        config_path=TRAIN_CONFIG_PATH,
        overrides={
            "training": {"epochs": 2},
            "logging": {"save_dir": save_dir, "wandb_project": None},
        },
    )
    summary = trainer.train()
    assert "final_metrics" in summary, f"Training summary missing final_metrics"

    # Find checkpoint
    runs_dir = Path(save_dir)
    checkpoints = list(runs_dir.rglob("best.pth")) + list(runs_dir.rglob("last.pth"))
    assert len(checkpoints) > 0, f"No checkpoint found in {runs_dir}"
    ckpt_path = checkpoints[0]
    print(f"    Checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------------
    # Step 2: Evaluate on val set
    # -----------------------------------------------------------------------
    print("  [2/4] Evaluating on val set...")
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

    data_config = load_config(DATA_CONFIG_PATH)
    data_config["_config_dir"] = Path(DATA_CONFIG_PATH).parent

    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        output_format=model.output_format,
        batch_size=4,
        num_workers=0,
    )
    eval_results = evaluator.evaluate(split="val")
    assert "mAP" in eval_results or "mAP50" in eval_results, (
        f"Evaluation missing mAP. Keys: {list(eval_results.keys())}"
    )
    print(f"    Eval keys: {list(eval_results.keys())}")

    # -----------------------------------------------------------------------
    # Step 3: Export to ONNX
    # -----------------------------------------------------------------------
    print("  [3/4] Exporting to ONNX...")
    model.cpu()
    onnx_path = str(OUTPUTS / "e2e_model.onnx")
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    assert Path(onnx_path).exists(), f"ONNX export failed"
    onnx.checker.check_model(onnx_path)
    print(f"    Exported: {onnx_path}")

    # -----------------------------------------------------------------------
    # Step 4: ONNX inference on real image
    # -----------------------------------------------------------------------
    print("  [4/4] Running ONNX inference...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image_bgr = real_image_bgr_640(idx=0, split="val")
    image_rgb = image_bgr[:, :, ::-1].copy()
    input_np = np.expand_dims(
        image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0
    )

    ort_outputs = session.run(None, {input_name: input_np})
    assert ort_outputs[0].shape[0] == 1, f"Unexpected batch size: {ort_outputs[0].shape}"

    # Compare with PyTorch output
    input_tensor = torch.from_numpy(input_np)
    with torch.no_grad():
        pt_output = model(input_tensor)
        if isinstance(pt_output, (tuple, list)):
            pt_output = pt_output[0]

    assert pt_output.shape == ort_outputs[0].shape, (
        f"Shape mismatch: PT {pt_output.shape} vs ONNX {ort_outputs[0].shape}"
    )
    print(f"    ONNX output shape: {ort_outputs[0].shape}")
    print("    E2E pipeline: OK")


if __name__ == "__main__":
    run_all([
        ("e2e_train_eval_export_infer", test_e2e_train_eval_export_infer),
    ], title="Test 15: End-to-End Pipeline")
