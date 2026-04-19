"""YOLOX model architecture (self-contained fallback).

Contains CSPDarknet backbone, PAFPN neck, and Decoupled Head.  When the
official ``yolox`` package is installed it is used instead; otherwise the
self-contained implementation below serves as a drop-in replacement.

All YOLOX size variants (tiny, s, m, l, x) are registered in the model
registry with their default depth/width multipliers.  Explicit ``depth``
and ``width`` values in the config override the variant defaults.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from core.p06_models.base import DetectionModel
from core.p06_models.registry import MODEL_REGISTRY, _VARIANT_MAP, register_model

logger = logging.getLogger(__name__)

# --- YOLOX variant defaults (depth, width) --------------------------------

_YOLOX_VARIANTS: Dict[str, Tuple[float, float]] = {
    "yolox-nano": (0.33, 0.25),
    "yolox-tiny": (0.33, 0.375),
    "yolox-s": (0.33, 0.50),
    "yolox-m": (0.67, 0.75),
    "yolox-l": (1.0, 1.0),
    "yolox-x": (1.33, 1.25),
}


# ---------------------------------------------------------------------------
# Official YOLOX builder (optional dependency)
# ---------------------------------------------------------------------------


class _OfficialYOLOXAdapter(DetectionModel):
    """Adapter wrapping the official Megvii ``yolox`` package.

    Selected via ``config["model"]["impl"] = "official"``. Requires the
    ``yolox`` package — install by running ``bash scripts/setup-yolox-venv.sh``
    and using the resulting ``.venv-yolox-official/`` for training.

    Exposes the same :class:`DetectionModel` contract as
    :class:`YOLOXModel` so the existing ``DetectionTrainer`` works unchanged
    via its ``forward_with_loss()`` dispatch hook.
    """

    def __init__(
        self, num_classes: int, depth: float, width: float
    ) -> None:
        super().__init__()
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        head = YOLOXHead(num_classes, width, in_channels=in_channels)
        self._model = YOLOX(backbone, head)
        # Decoded outputs (B, N, 5+C) in eval mode — matches YOLOXModel layout
        self._model.head.decode_in_inference = True
        self.num_classes = num_classes
        self._strides = [8, 16, 32]

    @property
    def output_format(self) -> str:
        return "yolox"

    @property
    def strides(self) -> List[int]:
        return list(self._strides)

    @staticmethod
    def _convert_targets(targets: List[torch.Tensor]) -> torch.Tensor:
        """Convert per-image target list to official padded batch tensor.

        Args:
            targets: List of B tensors, each ``(M_i, 5)`` with
                ``[cls, cx, cy, w, h]`` in pixel coordinates.

        Returns:
            Padded tensor of shape ``(B, max_M, 5)``. All-zero rows are
            ignored by ``YOLOXHead`` (class id 0 + zero box = padding).
        """
        B = len(targets)
        max_m = max((t.shape[0] for t in targets), default=0)
        if max_m == 0:
            max_m = 1  # YOLOXHead requires at least 1 slot
        device = targets[0].device if B > 0 else torch.device("cpu")
        dtype = targets[0].dtype if B > 0 else torch.float32
        out = torch.zeros(B, max_m, 5, device=device, dtype=dtype)
        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                out[i, : t.shape[0]] = t[:, :5]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decoded predictions in eval mode; raw loss dict is not exposed here.

        The trainer uses :meth:`forward_with_loss` for both training and
        validation; this method is the inference-only fallback used by
        downstream pipelines that call ``model(imgs)`` directly.
        """
        return self._model(x)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Forward to wrapped upstream ``YOLOX`` module.

        Upstream Megvii checkpoints (``pretrained/yolox_*.pth``) use keys like
        ``backbone.backbone.*`` and ``head.*`` — matching this adapter's
        ``self._model`` exactly. Without this override, those keys would land
        unprefixed on the adapter, produce all-missing / all-unexpected (and
        silently train-from-scratch under ``strict=False``). Also filters
        shape-mismatched entries (e.g. 80-class COCO ``cls_preds`` into a
        custom ``num_classes`` head) when ``strict=False``.
        """
        has_model_prefix = any(k.startswith("_model.") for k in state_dict)
        target = self if has_model_prefix else self._model
        if not strict:
            state_dict = _filter_shape_mismatched(state_dict, target)
        return target.load_state_dict(state_dict, strict=strict)

    def forward_with_loss(
        self, images: torch.Tensor, targets: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward + loss contract matching HF detection models.

        Training mode: returns ``(loss, loss_dict, None)`` — predictions
        unused in the training loop.

        Eval mode: two forwards — one in train mode under ``no_grad`` for
        loss, one in eval mode for decoded predictions.
        """
        official_targets = self._convert_targets(targets)

        if self.training:
            loss_dict = self._model(images, official_targets)
            loss = loss_dict["total_loss"]
            mapped = {
                "cls_loss": loss_dict["cls_loss"],
                "obj_loss": loss_dict["conf_loss"],
                "reg_loss": loss_dict["iou_loss"],
            }
            return loss, mapped, None

        # Eval: loss requires train-mode forward (targets path); predictions
        # require eval-mode forward. Official YOLOX can't return both.
        self._model.train()
        with torch.no_grad():
            loss_dict = self._model(images, official_targets)
        self._model.eval()
        predictions = self._model(images)
        loss = loss_dict["total_loss"]
        mapped = {
            "cls_loss": loss_dict["cls_loss"],
            "obj_loss": loss_dict["conf_loss"],
            "reg_loss": loss_dict["iou_loss"],
        }
        return loss, mapped, predictions


# ---------------------------------------------------------------------------
# Self-contained YOLOX building blocks
# ---------------------------------------------------------------------------


_ACT_MAP = {"silu": nn.SiLU, "relu": nn.ReLU, "hardswish": nn.Hardswish}


def _filter_shape_mismatched(
    state_dict: Dict[str, torch.Tensor], model: nn.Module
) -> Dict[str, torch.Tensor]:
    """Drop state_dict entries whose shape does not match the target model.

    ``nn.Module.load_state_dict(strict=False)`` handles missing/unexpected
    keys but still raises ``RuntimeError`` on shape mismatches. When loading
    an 80-class COCO-pretrained YOLOX checkpoint into a model with a
    different ``num_classes``, the classification head ``cls_preds.*`` shapes
    differ and must be dropped so the rest of the weights (backbone, neck,
    regression head) load.
    """
    target = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k in target and target[k].shape != v.shape:
            skipped.append(f"{k}: ckpt={tuple(v.shape)} model={tuple(target[k].shape)}")
        else:
            filtered[k] = v
    if skipped:
        logger.info(
            "Skipping %d shape-mismatched pretrained weights (e.g. class head "
            "on num_classes change): %s",
            len(skipped), ", ".join(skipped[:3]) + (" ..." if len(skipped) > 3 else ""),
        )
    return filtered


class _BaseConv(nn.Module):
    """Convolution + BatchNorm + activation block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        ksize: Kernel size. Default: 3.
        stride: Convolution stride. Default: 1.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv for ksize > 1. Default: False.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: int = 3,
        stride: int = 1,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        if depthwise and ksize > 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, ksize, stride, pad, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = _ACT_MAP[act_type](inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _Bottleneck(nn.Module):
    """CSP bottleneck block with residual connection.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        shortcut: Whether to use residual connection. Default: True.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        shortcut: bool = True,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        mid_ch = out_ch
        self.conv1 = _BaseConv(in_ch, mid_ch, ksize=1, stride=1, act_type=act_type)
        self.conv2 = _BaseConv(mid_ch, out_ch, ksize=3, stride=1, act_type=act_type, depthwise=depthwise)
        self.use_shortcut = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.use_shortcut:
            out = out + x
        return out


class _Focus(nn.Module):
    """Focus module — slices input into 4 sub-images then applies conv.

    Converts a ``(B, C, H, W)`` tensor into ``(B, 4C, H/2, W/2)``
    by interleaving pixels, then applies a ``_BaseConv``.

    This matches the official Megvii YOLOX stem structure (``3x3`` conv
    with ``4*in_ch`` input channels) rather than a direct ``6x6`` conv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: int = 3,
        stride: int = 1,
        act_type: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = _BaseConv(in_ch * 4, out_ch, ksize=ksize, stride=stride, act_type=act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice into 4 sub-images
        patch_tl = x[..., ::2, ::2]
        patch_tr = x[..., ::2, 1::2]
        patch_bl = x[..., 1::2, ::2]
        patch_br = x[..., 1::2, 1::2]
        return self.conv(torch.cat([patch_tl, patch_bl, patch_tr, patch_br], dim=1))


class _SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling bottleneck (used in dark5 of official YOLOX).

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_sizes: Pooling kernel sizes. Default: (5, 9, 13).
        act_type: Activation function name. Default: "silu".
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes: Tuple[int, ...] = (5, 9, 13),
        act_type: str = "silu",
    ) -> None:
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = _BaseConv(in_ch, mid_ch, ksize=1, stride=1, act_type=act_type)
        self.pools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        )
        # concat: mid_ch * (1 + len(kernel_sizes))
        self.conv2 = _BaseConv(mid_ch * (1 + len(kernel_sizes)), out_ch, ksize=1, stride=1, act_type=act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pools], dim=1))


class _CSPLayer(nn.Module):
    """Cross Stage Partial layer with N bottlenecks.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        n: Number of bottleneck blocks. Default: 1.
        shortcut: Use residual in bottlenecks. Default: True.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        mid_ch = out_ch // 2
        self.conv1 = _BaseConv(in_ch, mid_ch, ksize=1, stride=1, act_type=act_type)
        self.conv2 = _BaseConv(in_ch, mid_ch, ksize=1, stride=1, act_type=act_type)
        self.conv3 = _BaseConv(mid_ch * 2, out_ch, ksize=1, stride=1, act_type=act_type)
        self.blocks = nn.Sequential(
            *[_Bottleneck(mid_ch, mid_ch, shortcut=shortcut, act_type=act_type, depthwise=depthwise) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.blocks(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat([x1, x2], dim=1))


class _CSPDarknet(nn.Module):
    """CSPDarknet53 backbone with configurable depth and width.

    Produces three feature maps at strides 8, 16, 32.

    Args:
        depth: Depth multiplier controlling number of bottleneck blocks.
        width: Width multiplier controlling channel counts.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        depth: float = 0.67,
        width: float = 0.75,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        base_channels = int(64 * width)
        base_depth = max(round(3 * depth), 1)
        kw = dict(act_type=act_type, depthwise=depthwise)

        # Stem (Focus module — matches official Megvii YOLOX architecture)
        self.stem = _Focus(3, base_channels, ksize=3, act_type=act_type)

        # Dark2 -> stride 4
        self.dark2 = nn.Sequential(
            _BaseConv(base_channels, base_channels * 2, ksize=3, stride=2, **kw),
            _CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, **kw),
        )
        # Dark3 -> stride 8, output[0]
        self.dark3 = nn.Sequential(
            _BaseConv(base_channels * 2, base_channels * 4, ksize=3, stride=2, **kw),
            _CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, **kw),
        )
        # Dark4 -> stride 16, output[1]
        self.dark4 = nn.Sequential(
            _BaseConv(base_channels * 4, base_channels * 8, ksize=3, stride=2, **kw),
            _CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, **kw),
        )
        # Dark5 -> stride 32, output[2]
        self.dark5 = nn.Sequential(
            _BaseConv(base_channels * 8, base_channels * 16, ksize=3, stride=2, **kw),
            _SPPBottleneck(base_channels * 16, base_channels * 16, act_type=act_type),
            _CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, **kw),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        out1 = self.dark3(x)   # stride 8
        out2 = self.dark4(out1)  # stride 16
        out3 = self.dark5(out2)  # stride 32
        return out1, out2, out3


class _PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network (PAFPN) neck.

    Takes three backbone feature maps and fuses them top-down
    then bottom-up.

    Args:
        depth: Depth multiplier for CSP layers.
        width: Width multiplier for channel counts.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        depth: float = 0.67,
        width: float = 0.75,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        base_channels = int(64 * width)
        base_depth = max(round(3 * depth), 1)
        kw = dict(act_type=act_type, depthwise=depthwise)

        ch = [base_channels * 4, base_channels * 8, base_channels * 16]

        # Top-down path
        self.lateral_conv0 = _BaseConv(ch[2], ch[1], ksize=1, stride=1, act_type=act_type)
        self.C3_p4 = _CSPLayer(ch[1] * 2, ch[1], n=base_depth, shortcut=False, **kw)
        self.reduce_conv1 = _BaseConv(ch[1], ch[0], ksize=1, stride=1, act_type=act_type)
        self.C3_p3 = _CSPLayer(ch[0] * 2, ch[0], n=base_depth, shortcut=False, **kw)

        # Bottom-up path
        self.bu_conv2 = _BaseConv(ch[0], ch[0], ksize=3, stride=2, **kw)
        self.C3_n3 = _CSPLayer(ch[0] * 2, ch[1], n=base_depth, shortcut=False, **kw)
        self.bu_conv1 = _BaseConv(ch[1], ch[1], ksize=3, stride=2, **kw)
        self.C3_n4 = _CSPLayer(ch[1] * 2, ch[2], n=base_depth, shortcut=False, **kw)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1, feat2, feat3 = inputs  # stride 8, 16, 32

        # Top-down
        p5 = self.lateral_conv0(feat3)
        p5_up = self.upsample(p5)
        p4 = self.C3_p4(torch.cat([p5_up, feat2], dim=1))

        p4_r = self.reduce_conv1(p4)
        p4_up = self.upsample(p4_r)
        p3 = self.C3_p3(torch.cat([p4_up, feat1], dim=1))

        # Bottom-up
        p3_down = self.bu_conv2(p3)
        n3 = self.C3_n3(torch.cat([p3_down, p4_r], dim=1))

        n3_down = self.bu_conv1(n3)
        n4 = self.C3_n4(torch.cat([n3_down, p5], dim=1))

        return p3, n3, n4


class _DecoupledHead(nn.Module):
    """YOLOX decoupled detection head for a single scale.

    Separate branches for classification, regression, and objectness.

    Args:
        num_classes: Number of detection classes.
        in_ch: Input channel count from FPN.
        mid_ch: Hidden channel count. Default: 256.
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        num_classes: int,
        in_ch: int,
        mid_ch: int = 256,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        kw = dict(act_type=act_type, depthwise=depthwise)

        # Shared stem
        self.stem = _BaseConv(in_ch, mid_ch, ksize=1, stride=1, act_type=act_type)

        # Classification branch
        self.cls_convs = nn.Sequential(
            _BaseConv(mid_ch, mid_ch, ksize=3, stride=1, **kw),
            _BaseConv(mid_ch, mid_ch, ksize=3, stride=1, **kw),
        )
        self.cls_pred = nn.Conv2d(mid_ch, num_classes, 1, 1, 0)

        # Regression branch
        self.reg_convs = nn.Sequential(
            _BaseConv(mid_ch, mid_ch, ksize=3, stride=1, **kw),
            _BaseConv(mid_ch, mid_ch, ksize=3, stride=1, **kw),
        )
        self.reg_pred = nn.Conv2d(mid_ch, 4, 1, 1, 0)

        # Objectness (shares reg branch features)
        self.obj_pred = nn.Conv2d(mid_ch, 1, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Classification
        cls_feat = self.cls_convs(x)
        cls_out = self.cls_pred(cls_feat)

        # Regression + objectness
        reg_feat = self.reg_convs(x)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        # During inference apply sigmoid to confidence outputs (matches official YOLOX).
        # During training keep raw logits — loss functions apply sigmoid internally.
        if not self.training:
            obj_out = obj_out.sigmoid()
            cls_out = cls_out.sigmoid()

        # Concatenate: (B, 5+C, H, W)
        output = torch.cat([reg_out, obj_out, cls_out], dim=1)
        return output


# ---------------------------------------------------------------------------
# Full YOLOX model
# ---------------------------------------------------------------------------


class YOLOXModel(DetectionModel):
    """Self-contained YOLOX model: CSPDarknet + PAFPN + Decoupled Head.

    This fallback implementation mirrors the official YOLOX architecture
    using standard PyTorch modules.

    Args:
        num_classes: Number of detection classes.
        depth: Depth multiplier. Default: 0.67 (YOLOX-M).
        width: Width multiplier. Default: 0.75 (YOLOX-M).
        act_type: Activation function name. Default: "silu".
        depthwise: Use depthwise separable conv. Default: False.
    """

    def __init__(
        self,
        num_classes: int,
        depth: float = 0.67,
        width: float = 0.75,
        act_type: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = _CSPDarknet(depth, width, act_type=act_type, depthwise=depthwise)
        self.neck = _PAFPN(depth, width, act_type=act_type, depthwise=depthwise)

        base_channels = int(64 * width)
        in_channels = [base_channels * 4, base_channels * 8, base_channels * 16]
        mid_ch = int(256 * width)

        self.heads = nn.ModuleList([
            _DecoupledHead(num_classes, ch, mid_ch, act_type=act_type, depthwise=depthwise)
            for ch in in_channels
        ])
        self._strides = [8, 16, 32]
        self._grid_cache: dict = {}
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        """Initialize classification and objectness biases for stable training."""
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
        for head in self.heads:
            nn.init.constant_(head.cls_pred.bias, bias_value)
            nn.init.constant_(head.obj_pred.bias, bias_value)

    @staticmethod
    def _remap_official_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remap official yolox_m.pth keys to YOLOXModel naming convention.

        Official YOLOX wraps CSPDarknet inside YOLOPAFPN (→ backbone.backbone.xxx)
        and uses a single YOLOXHead with ModuleList (→ head.cls_convs.i.j.xxx).
        Our YOLOXModel uses separate backbone/neck modules and per-scale heads.

        Mapping:
          backbone.backbone.*  →  backbone.*
          backbone.*           →  neck.*           (PAFPN layers)
          head.cls_convs.i.j.* →  heads.i.cls_convs.j.*
          head.cls_preds.i.*   →  heads.i.cls_pred.*
          head.obj_preds.i.*   →  heads.i.obj_pred.*
          head.reg_convs.i.j.* →  heads.i.reg_convs.j.*
          head.reg_preds.i.*   →  heads.i.reg_pred.*
          head.stems.i.*       →  heads.i.stem.*
        """
        def _fix_blocks(key: str) -> str:
            # Official YOLOX names bottleneck ModuleList "m"; ours is "blocks"
            return key.replace(".m.", ".blocks.")

        remapped: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("backbone.backbone."):
                remapped[_fix_blocks(k[len("backbone."):])] = v
            elif k.startswith("backbone."):
                remapped[_fix_blocks("neck." + k[len("backbone."):])] = v
            elif k.startswith("head."):
                parts = k.split(".", 3)  # ["head", <field>, <i>, <rest>]
                field = parts[1]
                if field in ("cls_convs", "reg_convs") and len(parts) == 4:
                    i, rest = parts[2], parts[3]
                    remapped[f"heads.{i}.{field}.{rest}"] = v
                elif field in ("cls_preds", "obj_preds", "reg_preds") and len(parts) >= 3:
                    i = parts[2]
                    rest = parts[3] if len(parts) == 4 else ""
                    dest_field = field.rstrip("s")  # cls_preds→cls_pred, etc.
                    remapped[f"heads.{i}.{dest_field}.{rest}".rstrip(".")] = v
                elif field == "stems" and len(parts) >= 3:
                    i = parts[2]
                    rest = parts[3] if len(parts) == 4 else ""
                    remapped[f"heads.{i}.stem.{rest}".rstrip(".")] = v
                else:
                    remapped[k] = v
            else:
                remapped[k] = v
        return remapped

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load state dict, auto-remapping official yolox package key format if detected.

        When ``strict=False`` also filters shape-mismatched entries (e.g.
        80-class COCO ``cls_preds`` loading into a ``num_classes=2`` head).
        """
        if any(k.startswith("backbone.backbone.") for k in state_dict):
            logger.info("Detected official YOLOX key format — remapping to YOLOXModel convention")
            state_dict = self._remap_official_keys(state_dict)
        if not strict:
            state_dict = _filter_shape_mismatched(state_dict, self)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full YOLOX model.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Decoded predictions of shape (B, N, 5 + num_classes)
            where N = sum of H_i * W_i across all scales.
            Per-anchor format: [cx, cy, w, h, obj, cls_0, ..., cls_C].
        """
        features = self.backbone(x)
        fpn_outs = self.neck(features)

        outputs = []
        for i, (feat, head) in enumerate(zip(fpn_outs, self.heads)):
            raw = head(feat)  # (B, 5+C, H_i, W_i)
            B, _, H, W = raw.shape
            # Reshape to (B, H*W, 5+C)
            raw = raw.permute(0, 2, 3, 1).reshape(B, H * W, -1)

            # Decode box predictions relative to grid (cached)
            stride = self._strides[i]
            cache_key = (H, W, i, x.device)
            if cache_key not in self._grid_cache:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=x.device, dtype=x.dtype),
                    torch.arange(W, device=x.device, dtype=x.dtype),
                    indexing="ij",
                )
                self._grid_cache[cache_key] = (
                    torch.stack([grid_x, grid_y], dim=-1).reshape(1, H * W, 2)
                )
            grid = self._grid_cache[cache_key]

            # cx, cy = (grid + raw_xy) * stride  — matches official YOLOX
            # (Megvii/YOLOX). Previously the custom path added "+ 0.5", which
            # worked self-consistently when training from scratch but produced
            # a half-grid-cell offset when loading Megvii pretrained weights —
            # boxes shifted by up to 16 px on stride 32. Removed for parity.
            decoded = raw.clone()
            # clamp offsets: (grid+offset)*stride must fit fp16 (<65504/32≈2047→offset≤1967)
            decoded[..., :2] = (grid + raw[..., :2].clamp(min=-100.0, max=100.0)) * stride
            # max=7.0: exp(7)*stride_max(32)=32891 < fp16_max(65504), prevents inf→NaN in GIoU under AMP
            decoded[..., 2:4] = torch.exp(raw[..., 2:4].clamp(min=-5.0, max=7.0)) * stride

            outputs.append(decoded)

        return torch.cat(outputs, dim=1)

    @property
    def output_format(self) -> str:
        """Output format identifier."""
        return "yolox"

    @property
    def strides(self) -> List[int]:
        """Detection strides."""
        return list(self._strides)

    def get_param_groups(
        self, lr: float, weight_decay: float
    ) -> List[Dict]:
        """Return six parameter groups: backbone/neck/head x decay/no_decay.

        Args:
            lr: Base learning rate.
            weight_decay: Weight-decay coefficient for the decay groups.

        Returns:
            List of six parameter-group dicts.
        """
        groups: Dict[str, List[nn.Parameter]] = {
            "backbone_decay": [],
            "backbone_no_decay": [],
            "neck_decay": [],
            "neck_no_decay": [],
            "head_decay": [],
            "head_no_decay": [],
        }

        component_map = {
            "backbone": self.backbone,
            "neck": self.neck,
            "head": self.heads,
        }

        for comp_name, comp_module in component_map.items():
            seen: set = set()
            for module in comp_module.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                    for p in module.parameters():
                        if id(p) not in seen:
                            groups[f"{comp_name}_no_decay"].append(p)
                            seen.add(id(p))
                else:
                    if hasattr(module, "bias") and isinstance(
                        module.bias, nn.Parameter
                    ) and id(module.bias) not in seen:
                        groups[f"{comp_name}_no_decay"].append(module.bias)
                        seen.add(id(module.bias))
                    if hasattr(module, "weight") and isinstance(
                        module.weight, nn.Parameter
                    ) and id(module.weight) not in seen:
                        groups[f"{comp_name}_decay"].append(module.weight)
                        seen.add(id(module.weight))

        return [
            {"params": groups["backbone_decay"], "lr": lr, "weight_decay": weight_decay, "name": "backbone_decay", "group_name": "backbone_decay"},
            {"params": groups["backbone_no_decay"], "lr": lr, "weight_decay": 0.0, "name": "backbone_no_decay", "group_name": "backbone_no_decay"},
            {"params": groups["neck_decay"], "lr": lr, "weight_decay": weight_decay, "name": "neck_decay", "group_name": "neck_decay"},
            {"params": groups["neck_no_decay"], "lr": lr, "weight_decay": 0.0, "name": "neck_no_decay", "group_name": "neck_no_decay"},
            {"params": groups["head_decay"], "lr": lr, "weight_decay": weight_decay, "name": "head_decay", "group_name": "head_decay"},
            {"params": groups["head_no_decay"], "lr": lr, "weight_decay": 0.0, "name": "head_no_decay", "group_name": "head_no_decay"},
        ]


# ---------------------------------------------------------------------------
# Registry: build function + variant registration
# ---------------------------------------------------------------------------


@register_model("yolox")
def build_yolox(config: dict) -> nn.Module:
    """Build a YOLOX model from config.

    Dispatches on ``config["model"]["impl"]``:

    * ``"custom"`` (default) — self-contained :class:`YOLOXModel`.
    * ``"official"`` — :class:`_OfficialYOLOXAdapter` wrapping the Megvii
      ``yolox`` package. Requires ``.venv-yolox-official/`` (see
      ``scripts/setup-yolox-venv.sh``).

    The variant name (e.g. ``yolox-m``) provides default depth/width, but
    explicit ``depth`` and ``width`` in ``config["model"]`` take precedence.

    Args:
        config: Full training config with a ``"model"`` section containing
            at least ``num_classes``.

    Returns:
        YOLOX model instance.
    """
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "yolox-m").lower()
    num_classes = model_cfg["num_classes"]
    impl = model_cfg.get("impl", "custom").lower()

    # Resolve default depth/width from variant, allow config override
    default_depth, default_width = _YOLOX_VARIANTS.get(arch, (0.67, 0.75))
    depth = model_cfg.get("depth", default_depth)
    width = model_cfg.get("width", default_width)

    if impl == "official":
        if model_cfg.get("act_type", "silu") != "silu" or model_cfg.get("depthwise", False):
            logger.warning(
                "model.impl=official ignores act_type / depthwise config keys "
                "(fixed to upstream defaults: silu, depthwise=False)."
            )
        model = _OfficialYOLOXAdapter(num_classes, depth, width)
        logger.info(
            "Built YOLOX via official Megvii package "
            "(depth=%.2f, width=%.2f, classes=%d).",
            depth, width, num_classes,
        )
        return model

    if impl != "custom":
        raise ValueError(
            f"Unknown model.impl={impl!r}. Expected 'custom' or 'official'."
        )

    act_type = model_cfg.get("act_type", "silu")
    depthwise = model_cfg.get("depthwise", False)
    model = YOLOXModel(num_classes, depth, width, act_type=act_type, depthwise=depthwise)
    logger.info(
        "Built YOLOX model using built-in architecture "
        "(depth=%.2f, width=%.2f, classes=%d).",
        depth,
        width,
        num_classes,
    )

    return model


# Register all YOLOX variant names to the canonical "yolox" builder
for _variant in _YOLOX_VARIANTS:
    _VARIANT_MAP[_variant] = "yolox"
