"""PaddlePaddle native top-down keypoint estimators (PP-TinyPose family).

Registers two architectures under the standard model registry:

- ``pp-tinypose-128x96``  — mobile-grade, 128x96 input, COCO 17-keypoint
- ``pp-tinypose-256x192`` — server-grade, 256x192 input, COCO 17-keypoint

The builder returns :class:`PPTinyPoseModel`, a thin wrapper that mirrors
:class:`core.p06_models.hf_model.HFKeypointModel` exactly: same
``forward_with_loss`` signature, same heatmap shape contract, and the same
weighted-MSE heatmap loss. Consumers (the HF Trainer dispatcher and the
top-down post-train runner) need no special-casing.

Heavy ``paddle`` imports are deferred into the builder so the rest of
``core.p06_models`` stays importable in venvs that don't have paddle
installed.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model

# ---------------------------------------------------------------------------
# Per-arch input + stride spec
# ---------------------------------------------------------------------------

# Heatmaps are emitted at stride 4 (Lite-HRNet / PP-TinyPose convention).
_ARCH_SPEC: dict[str, dict[str, Any]] = {
    "pp-tinypose-128x96": {
        "input_size": (128, 96),  # (H, W)
        "width_mult": 0.5,
        "description": "PP-TinyPose mobile (128x96)",
    },
    "pp-tinypose-256x192": {
        "input_size": (256, 192),  # (H, W)
        "width_mult": 1.0,
        "description": "PP-TinyPose server (256x192)",
    },
}


# ---------------------------------------------------------------------------
# Lite Lite-HRNet-style backbone + heatmap head (paddle.nn.Layer)
# ---------------------------------------------------------------------------


def _build_paddle_backbone(num_keypoints: int, width_mult: float):
    """Construct a Lite-HRNet-style backbone + heatmap head in Paddle.

    Output: ``(B, num_keypoints, H/4, W/4)`` heatmaps from a ``(B, 3, H, W)``
    input — matches PP-TinyPose's stride-4 heatmap contract.

    This is intentionally a compact reference implementation suitable for
    the registry/training-smoke tests; production weights would be
    fine-tuned via PaddleDetection's full PP-TinyPose recipe.
    """
    import paddle.nn as pnn  # type: ignore[import-not-found]

    def _c(c: int) -> int:
        return max(8, int(round(c * width_mult)))

    class _ConvBNReLU(pnn.Layer):
        def __init__(self, ic: int, oc: int, k: int = 3, s: int = 1) -> None:
            super().__init__()
            self.conv = pnn.Conv2D(ic, oc, k, stride=s, padding=k // 2, bias_attr=False)
            self.bn = pnn.BatchNorm2D(oc)
            self.act = pnn.ReLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class _Backbone(pnn.Layer):
        def __init__(self) -> None:
            super().__init__()
            # Stem: stride-2 → stride-4 (PP-TinyPose downsamples by 4 before the head)
            self.stem = pnn.Sequential(
                _ConvBNReLU(3, _c(32), k=3, s=2),
                _ConvBNReLU(_c(32), _c(64), k=3, s=2),
            )
            # Stage blocks at stride 4 (no further downsampling — heatmap stride is 4)
            self.stage = pnn.Sequential(
                _ConvBNReLU(_c(64), _c(128), k=3, s=1),
                _ConvBNReLU(_c(128), _c(128), k=3, s=1),
                _ConvBNReLU(_c(128), _c(128), k=3, s=1),
            )
            # Heatmap head — 1x1 conv to K keypoint channels
            self.head = pnn.Conv2D(_c(128), num_keypoints, kernel_size=1)

        def forward(self, x):
            x = self.stem(x)
            x = self.stage(x)
            return self.head(x)

    return _Backbone()


# ---------------------------------------------------------------------------
# Wrapper — matches HFKeypointModel contract
# ---------------------------------------------------------------------------


class PPTinyPoseModel(DetectionModel):
    """Native-paddle top-down keypoint model wrapper.

    Mirrors :class:`core.p06_models.hf_model.HFKeypointModel`:

    - ``forward(pixel_values, target_heatmap=None, target_weight=None)``
      returns the raw heatmaps (inference) or ``{"loss", "heatmaps"}``
      (training/eval, HF-Trainer convention).
    - ``forward_with_loss(images, targets)`` returns
      ``(loss, {"kpt_loss": ...}, heatmaps)`` — pytorch-backend hook,
      identical to the HF wrapper.

    Heatmaps are weighted MSE against ``target_heatmap`` using
    ``target_weight`` broadcast to per-channel scalar weights — same
    formula as the HF wrapper.
    """

    _keys_to_ignore_on_save = None

    def __init__(self, paddle_model: Any, num_keypoints: int, input_size: tuple[int, int]) -> None:
        # Paddle params live in ``paddle_model`` and are managed by paddle's
        # optimizer; the torch.nn.Module super-init is a no-op for them.
        super().__init__()
        self.paddle_model = paddle_model
        self.num_keypoints = int(num_keypoints)
        self.input_size = tuple(input_size)

    @property
    def output_format(self) -> str:
        return "keypoint"

    # ------------------------------------------------------------------
    # Core forward — matches HFKeypointModel.forward signature/return
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: Any,
        target_heatmap: Any = None,
        target_weight: Any = None,
        labels: Any = None,  # noqa: ARG002 — HF Trainer may inject; ignored
        **kwargs: Any,  # noqa: ARG002
    ):
        import paddle  # type: ignore[import-not-found]

        x = _to_paddle(pixel_values)
        heatmaps = self.paddle_model(x)

        if target_heatmap is None:
            return heatmaps

        target_hm = _to_paddle(target_heatmap)
        if target_weight is not None:
            tw = _to_paddle(target_weight)
            weight = paddle.unsqueeze(paddle.unsqueeze(tw, axis=-1), axis=-1)
        else:
            weight = 1.0
        loss = paddle.mean(((heatmaps - target_hm) ** 2) * weight)
        return {"loss": loss, "heatmaps": heatmaps}

    # ------------------------------------------------------------------
    # Pytorch-backend hook — identical to HFKeypointModel.forward_with_loss
    # ------------------------------------------------------------------
    def forward_with_loss(self, images: Any, targets: list) -> tuple[Any, dict[str, Any], Any]:
        """Pytorch-backend hook (parity with HFKeypointModel).

        ``targets`` is a list of ``(target_heatmap, target_weight)`` pairs;
        we stack them in paddle just like the HF wrapper stacks in torch.
        """
        import paddle  # type: ignore[import-not-found]

        target_hm = paddle.stack([_to_paddle(t[0]) for t in targets])
        target_wt = paddle.stack([_to_paddle(t[1]) for t in targets])
        out = self.forward(
            pixel_values=images, target_heatmap=target_hm, target_weight=target_wt,
        )
        loss = out["loss"]
        heatmaps = out["heatmaps"]
        return loss, {"kpt_loss": loss.detach()}, heatmaps


# ---------------------------------------------------------------------------
# Tensor helpers — allow torch tensors / numpy / paddle tensors to feed in
# ---------------------------------------------------------------------------


def _to_paddle(x: Any):
    """Best-effort conversion to a paddle.Tensor.

    Accepts paddle.Tensor, torch.Tensor, numpy.ndarray, or anything paddle
    can wrap directly. Torch tensors are routed via numpy (zero-copy on
    CPU; one host copy on CUDA).
    """
    import paddle  # type: ignore[import-not-found]

    if isinstance(x, paddle.Tensor):
        return x
    # Torch tensor → numpy → paddle
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return paddle.to_tensor(x.detach().cpu().numpy())
    except ImportError:  # pragma: no cover
        pass
    return paddle.to_tensor(x)


# ---------------------------------------------------------------------------
# Registry hooks
# ---------------------------------------------------------------------------


def _resolve_num_keypoints(config: dict) -> int:
    """Pull num_keypoints from model config, falling back to data.skeleton."""
    model_cfg = config.get("model", {})
    if "num_keypoints" in model_cfg:
        return int(model_cfg["num_keypoints"])

    data_cfg = config.get("data", {})
    skeleton = data_cfg.get("skeleton") or {}
    names = skeleton.get("keypoint_names")
    if names:
        return len(names)
    return 17  # COCO default


def _build_pptinypose(config: dict, arch: str) -> PPTinyPoseModel:
    """Shared builder for both PP-TinyPose variants."""
    spec = _ARCH_SPEC[arch]
    num_keypoints = _resolve_num_keypoints(config)

    logger.info(
        "Building {} (num_keypoints={}, input_size={})",
        spec["description"],
        num_keypoints,
        spec["input_size"],
    )

    backbone = _build_paddle_backbone(num_keypoints, width_mult=spec["width_mult"])
    return PPTinyPoseModel(
        paddle_model=backbone,
        num_keypoints=num_keypoints,
        input_size=spec["input_size"],
    )


@register_model("pp-tinypose-128x96")
def build_pptinypose_128x96(config: dict) -> PPTinyPoseModel:
    """Build PP-TinyPose mobile (128x96 input, stride-4 heatmaps).

    Config example::

        model:
          arch: pp-tinypose-128x96
          num_keypoints: 17        # or omit to read data.skeleton.keypoint_names
        data:
          skeleton:
            keypoint_names: [nose, left_eye, ...]  # 17 entries for COCO
    """
    return _build_pptinypose(config, "pp-tinypose-128x96")


@register_model("pp-tinypose-256x192")
def build_pptinypose_256x192(config: dict) -> PPTinyPoseModel:
    """Build PP-TinyPose server (256x192 input, stride-4 heatmaps)."""
    return _build_pptinypose(config, "pp-tinypose-256x192")


__all__ = [
    "PPTinyPoseModel",
    "build_pptinypose_128x96",
    "build_pptinypose_256x192",
]
