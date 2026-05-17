"""Smoke tests for multi-task D-FINE detection training.

Verifies:
1. The `dfine-*-multitask` archs are registered in MODEL_REGISTRY.
2. `build_model()` constructs a working DFineMultitaskModel.
3. A single forward pass on dummy inputs succeeds for each task.
4. Shared trunk parameter count matches a single-task baseline (no Nx blow-up).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _make_config(arch: str = "dfine-n-multitask") -> dict:
    return {
        "model": {
            "arch": arch,
            "pretrained": "ustc-community/dfine_n_coco",
            "input_size": [320, 320],  # small for speed
            "num_queries": 30,
            "num_denoising": 0,
        },
        "tensor_prep": {
            "input_size": [320, 320],
            "rescale": True,
            "normalize": False,
            "applied_by": "hf_processor",
        },
        "_tasks": [
            {"name": "fire_smoke",
             "num_classes": 2,
             "names": {0: "fire", 1: "smoke"}},
            {"name": "helmet",
             "num_classes": 4,
             "names": {0: "person", 1: "head_with_helmet",
                       2: "head_without_helmet", 3: "head_with_nitto_hat"}},
        ],
    }


def test_multitask_arch_registered():
    from core.p06_models import MODEL_REGISTRY
    for arch in ("dfine-n-multitask", "dfine-s-multitask",
                 "dfine-m-multitask", "dfine-l-multitask"):
        assert arch in MODEL_REGISTRY, f"{arch} not registered"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for D-FINE forward")
def test_multitask_build_and_forward():
    from core.p06_models import build_model
    config = _make_config()
    model = build_model(config).cuda().eval()
    assert set(model.task_names) == {"fire_smoke", "helmet"}
    assert model.num_classes == {"fire_smoke": 2, "helmet": 4}

    px = torch.randn(1, 3, 320, 320, device="cuda")
    for task in model.task_names:
        with torch.no_grad():
            out = model(pixel_values=px, task_name=task)
        # Per-task logits dim matches that task's num_classes.
        assert out.logits.shape[-1] == model.num_classes[task], (
            f"task={task}: logits cls dim {out.logits.shape[-1]} != "
            f"num_classes {model.num_classes[task]}"
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA for D-FINE forward")
def test_multitask_shared_trunk_param_count():
    """Multitask params should be ~single-task params (shared trunk), not N×."""
    from core.p06_models import build_model

    config = _make_config()
    model = build_model(config)
    mt_params = sum(p.numel() for _, p in model.named_parameters())

    # Single-task baseline for comparison.
    from transformers import DFineForObjectDetection
    single = DFineForObjectDetection.from_pretrained(
        "ustc-community/dfine_n_coco", num_labels=2, ignore_mismatched_sizes=True,
    )
    single_params = sum(p.numel() for p in single.parameters())

    # Per-task heads are small (Linear hidden_dim -> num_classes per decoder
    # layer + enc_score_head + denoising_class_embed). 2 tasks should add
    # well under 50% overhead.
    overhead = (mt_params - single_params) / single_params
    assert 0 <= overhead < 0.5, (
        f"Multitask param overhead {overhead:.2%} too large — trunk not shared "
        f"(mt={mt_params}, single={single_params})"
    )


def test_multitask_collate_enforces_single_task():
    from core.p05_data.multitask_dataset import multitask_collate_fn
    img = torch.zeros(3, 32, 32)
    tgt = torch.zeros(0, 5)
    # All-same-task batch: works.
    batch = [(img, tgt, "fire_smoke", "a.jpg"),
             (img, tgt, "fire_smoke", "b.jpg")]
    out = multitask_collate_fn(batch)
    assert out["task_name"] == "fire_smoke"
    assert out["pixel_values"].shape == (2, 3, 32, 32)
    # Mixed-task batch: rejected.
    bad = [(img, tgt, "fire_smoke", "a.jpg"),
           (img, tgt, "helmet", "b.jpg")]
    with pytest.raises(ValueError, match="Mixed-task batch"):
        multitask_collate_fn(bad)


if __name__ == "__main__":
    test_multitask_arch_registered()
    print("OK: arch registered")
    test_multitask_collate_enforces_single_task()
    print("OK: collate enforces single-task")
    if _has_cuda():
        test_multitask_build_and_forward()
        print("OK: build + forward")
        test_multitask_shared_trunk_param_count()
        print("OK: shared trunk")
    else:
        print("SKIP: cuda-dependent tests (no GPU)")
