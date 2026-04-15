"""Test 06: Model Registry — Framework Extension Smoke Tests.

Verifies that the model registry extension contracts work correctly:
- Custom models can be registered and built
- Unknown architectures raise clear errors
- Variant aliases resolve to canonical builders
- forward_with_loss path is detectable by trainer
- output_format values are correct on built models
"""

import sys
import traceback
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
import core.p06_models.yolox  # triggers YOLOX registration
import core.p06_models.dfine  # triggers D-FINE variant aliases
from core.p06_models.base import DetectionModel
from core.p06_models.registry import MODEL_REGISTRY, _VARIANT_MAP, build_model, register_model


class _MinimalDetector(DetectionModel):
    """Minimal DetectionModel subclass for registration tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 100, 85)

    @property
    def output_format(self) -> str:
        return "yolox"

    @property
    def strides(self):
        return [8, 16, 32]


class _ModelWithBuiltinLoss(DetectionModel):
    """Model that implements forward_with_loss (DETR-style)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_with_loss(self, images, targets):
        return torch.tensor(0.0), {}, images

    @property
    def output_format(self) -> str:
        return "detr"

    @property
    def strides(self):
        return []


def test_register_custom_model():
    """A minimal DetectionModel subclass can be registered and built."""
    arch_name = "test_minimal_detector_core09"
    register_model(arch_name)(lambda cfg: _MinimalDetector())

    try:
        model = build_model({"model": {"arch": arch_name, "num_classes": 80}})
        assert isinstance(model, DetectionModel), (
            f"Expected DetectionModel, got {type(model)}"
        )
        assert model.output_format == "yolox", (
            f"Expected output_format='yolox', got '{model.output_format}'"
        )
        dummy = torch.zeros(2, 3, 64, 64)
        out = model(dummy)
        assert out.shape == (2, 100, 85), f"Unexpected output shape: {out.shape}"
    finally:
        MODEL_REGISTRY.pop(arch_name, None)

    print("    Custom model registered, built, and produces correct output shape.")


def test_unknown_arch_error():
    """build_model raises ValueError for unknown arch with helpful message."""
    with pytest.raises(ValueError, match="Unknown model architecture"):
        build_model({"model": {"arch": "nonexistent_framework_xyz", "num_classes": 5}})
    print("    ValueError raised for unknown arch as expected.")


def test_variant_alias_resolution():
    """Variant aliases resolve to the canonical builder key."""
    assert "dfine-s" in _VARIANT_MAP, "'dfine-s' not found in _VARIANT_MAP"
    canonical = _VARIANT_MAP["dfine-s"]
    assert canonical in MODEL_REGISTRY, (
        f"Canonical key '{canonical}' for 'dfine-s' not in MODEL_REGISTRY"
    )
    print(f"    'dfine-s' -> '{canonical}' (in MODEL_REGISTRY: True)")


def test_forward_with_loss_detection():
    """Models with forward_with_loss() are detected correctly by the trainer."""
    model = _ModelWithBuiltinLoss()
    assert hasattr(model, "forward_with_loss"), (
        "Model should have forward_with_loss attribute"
    )
    # Verify forward_with_loss actually returns the expected tuple
    dummy_images = torch.zeros(1, 3, 64, 64)
    loss, loss_dict, preds = model.forward_with_loss(dummy_images, targets=None)
    assert isinstance(loss, torch.Tensor), f"Loss should be a Tensor, got {type(loss)}"
    assert isinstance(loss_dict, dict), f"loss_dict should be dict, got {type(loss_dict)}"
    print(
        f"    forward_with_loss detected and returns (loss={loss.item():.4f}, "
        f"loss_dict={loss_dict}, preds shape={preds.shape})"
    )


def test_model_without_forward_with_loss():
    """Models without forward_with_loss use the standard training path."""
    model = _MinimalDetector()
    assert not hasattr(model, "forward_with_loss"), (
        "Standard model should not have forward_with_loss"
    )
    print("    Standard-path model correctly lacks forward_with_loss.")


def test_output_format_yolox():
    """YOLOX model has output_format='yolox'."""
    config = {
        "model": {
            "arch": "yolox-m",
            "num_classes": 2,
            "input_size": [640, 640],
            "depth": 0.67,
            "width": 0.75,
        },
        "data": {"num_classes": 2},
    }
    model = build_model(config)
    assert model.output_format == "yolox", (
        f"Expected output_format='yolox', got '{model.output_format}'"
    )
    print(f"    yolox-m output_format='{model.output_format}' as expected.")


def test_registry_isolation():
    """Temporary registrations are cleaned up and do not pollute MODEL_REGISTRY."""
    arch_name = "test_isolation_core09"
    assert arch_name not in MODEL_REGISTRY, "Registry should not contain test arch before test"

    register_model(arch_name)(lambda cfg: _MinimalDetector())
    assert arch_name in MODEL_REGISTRY, "Registry should contain arch after registration"

    MODEL_REGISTRY.pop(arch_name)
    assert arch_name not in MODEL_REGISTRY, "Registry should not contain arch after cleanup"
    print("    Registry isolation verified — no leaked test entries.")


def test_get_param_groups_custom_model():
    """Custom DetectionModel inherits functional get_param_groups from base class."""
    model = _MinimalDetector()
    groups = model.get_param_groups(lr=0.01, weight_decay=5e-4)
    assert isinstance(groups, list), f"Expected list, got {type(groups)}"
    assert len(groups) >= 2, f"Expected >=2 param groups, got {len(groups)}"
    assert all("params" in g for g in groups), "Each group must have 'params' key"
    print(f"    get_param_groups returns {len(groups)} groups for custom model.")


if __name__ == "__main__":
    run_all([
        ("register_custom_model", test_register_custom_model),
        ("unknown_arch_error", test_unknown_arch_error),
        ("variant_alias_resolution", test_variant_alias_resolution),
        ("forward_with_loss_detection", test_forward_with_loss_detection),
        ("model_without_forward_with_loss", test_model_without_forward_with_loss),
        ("output_format_yolox", test_output_format_yolox),
        ("registry_isolation", test_registry_isolation),
        ("get_param_groups_custom_model", test_get_param_groups_custom_model),
    ], title="Test 09: Framework Extension Smoke Tests")
