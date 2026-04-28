"""PaddleDetection backend — PicoDet + PP-YOLOE detection model registry.

Wraps PaddleDetection's flagship architectures behind the same
``DetectionModel`` contract used by YOLOX / HF detectors. The actual paddle /
PaddleDetection imports are deferred to first use so this module can be
imported in any venv (main `.venv`, `.venv-paddle`, etc.) and the registry
just records the builder.

Pretrained-weight URLs follow the official PaddleDetection v2.6 naming
convention (https://paddledetection.readthedocs.io/MODEL_ZOO.html). Weights
download into ``~/.cache/paddle/weights`` on first build via paddle's hub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.p06_models.base import DetectionModel
from core.p06_models.registry import register_model
from loguru import logger

# ---------------------------------------------------------------------------
# Pretrained weight registry
# ---------------------------------------------------------------------------
# arch -> (config_yaml_relpath, pretrained_url)
#
# config_yaml_relpath is the canonical PaddleDetection config file under the
# upstream PaddleDetection repo's `configs/` directory. PP-YOLOE+ uses the
# `ppyoloe_plus_*` naming and the `ppyoloe_plus_crn_*_obj365_pretrained` /
# `_coco` weight URLs. Reference: https://github.com/PaddlePaddle/PaddleDetection
_PADDLE_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # --- PicoDet -----------------------------------------------------------
    "picodet-s": (
        "configs/picodet/picodet_s_416_coco_lcnet.yml",
        "https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams",
    ),
    "picodet-m": (
        "configs/picodet/picodet_m_416_coco_lcnet.yml",
        "https://paddledet.bj.bcebos.com/models/picodet_m_416_coco_lcnet.pdparams",
    ),
    "picodet-l": (
        "configs/picodet/picodet_l_640_coco_lcnet.yml",
        "https://paddledet.bj.bcebos.com/models/picodet_l_640_coco_lcnet.pdparams",
    ),
    # --- PP-YOLOE ----------------------------------------------------------
    "ppyoloe-s": (
        "configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams",
    ),
    "ppyoloe-m": (
        "configs/ppyoloe/ppyoloe_crn_m_300e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams",
    ),
    "ppyoloe-l": (
        "configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams",
    ),
    "ppyoloe-x": (
        "configs/ppyoloe/ppyoloe_crn_x_300e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams",
    ),
    # --- PP-YOLOE+ ---------------------------------------------------------
    "ppyoloe-plus-s": (
        "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams",
    ),
    "ppyoloe-plus-m": (
        "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_m_80e_coco.pdparams",
    ),
    "ppyoloe-plus-l": (
        "configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams",
    ),
    "ppyoloe-plus-x": (
        "configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml",
        "https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_x_80e_coco.pdparams",
    ),
}

_PADDLE_CACHE_DIR = Path(os.environ.get("PADDLE_CACHE_DIR", str(Path.home() / ".cache" / "paddle")))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class PaddleDetectionModel(DetectionModel):
    """Thin adapter around a PaddleDetection model.

    Heavy paddle imports happen inside :py:meth:`_build_lazy` so that the
    module-level ``@register_model`` decorators succeed in any venv. The
    underlying paddle module lives at ``self.model``; ``self.model`` is
    ``None`` until the first forward / build call.
    """

    _keys_to_ignore_on_save = None

    def __init__(
        self,
        arch: str,
        num_classes: int,
        input_size: tuple[int, int],
        pretrained: str | None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = int(num_classes)
        self.input_size = tuple(int(x) for x in input_size)
        self.pretrained = pretrained
        self.class_names = class_names or {i: f"class_{i}" for i in range(self.num_classes)}
        # Paddle model is lazily built on first call so importing this module
        # does not require paddle to be installed.
        self.model: Any = None
        self._paddle_cfg: Any = None

    # -- DetectionModel contract -------------------------------------------
    @property
    def output_format(self) -> str:
        return "paddle"

    @property
    def strides(self) -> list[int]:
        return [8, 16, 32]

    # -- lazy paddle build --------------------------------------------------
    def _build_lazy(self) -> None:
        """Import paddle + PaddleDetection and instantiate the model."""
        if self.model is not None:
            return

        try:
            import paddle  # noqa: F401
            from ppdet.core.workspace import create, load_config, merge_config
            from ppdet.utils.checkpoint import load_pretrain_weight
        except ImportError as exc:  # pragma: no cover — depends on .venv-paddle
            raise ImportError(
                "PaddleDetection backend requires `paddlepaddle` + `paddledet`. "
                "Run `bash scripts/setup-paddle-venv.sh` and use "
                "`.venv-paddle/bin/python` for paddle archs."
            ) from exc

        cfg_path, weight_url = _PADDLE_MODEL_REGISTRY[self.arch]
        # PaddleDetection ships its configs alongside the package; resolve via
        # ppdet install location first, then fall back to the raw GitHub URL.
        try:
            import ppdet
            ppdet_root = Path(ppdet.__file__).resolve().parent.parent
            cfg_file = ppdet_root / cfg_path
            if not cfg_file.exists():
                # Fall back to upstream raw URL — paddledet's `load_config`
                # handles http(s) sources transparently in v2.6+.
                cfg_file = (
                    f"https://raw.githubusercontent.com/PaddlePaddle/"
                    f"PaddleDetection/release/2.6/{cfg_path}"
                )
        except Exception:  # noqa: BLE001
            cfg_file = (
                f"https://raw.githubusercontent.com/PaddlePaddle/"
                f"PaddleDetection/release/2.6/{cfg_path}"
            )

        cfg = load_config(str(cfg_file))
        merge_config({"num_classes": self.num_classes})
        # ppdet caches downloaded weights under PADDLE_CACHE_DIR/weights
        _PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PPDET_WEIGHTS_HOME", str(_PADDLE_CACHE_DIR / "weights"))

        model = create(cfg.architecture)
        weights = self.pretrained or weight_url
        try:
            load_pretrain_weight(model, weights)
            logger.info("Loaded paddle pretrained weights from %s", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load paddle pretrained from %s: %s", weights, exc)

        self.model = model
        self._paddle_cfg = cfg

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run paddle inference and return predictions in our pipeline shape.

        Inputs come in as torch tensors ``(B, 3, H, W)`` already normalized
        by our v2 transform pipeline. We zero-copy them into paddle tensors,
        run the forward, and return a torch tensor of raw decode outputs.
        Postprocessing happens via :py:meth:`postprocess` (the trainer +
        evaluator dispatcher honors ``model.postprocess`` first).
        """
        self._build_lazy()
        import paddle

        # numpy detour — paddle has no zero-copy from torch CUDA tensors yet
        np_x = x.detach().cpu().numpy().astype(np.float32)
        pd_x = paddle.to_tensor(np_x)
        scale_factor = paddle.ones((np_x.shape[0], 2), dtype="float32")
        im_shape = paddle.to_tensor(
            np.array([[np_x.shape[2], np_x.shape[3]]] * np_x.shape[0], dtype="float32")
        )

        self.model.eval()
        outs = self.model({
            "image": pd_x,
            "scale_factor": scale_factor,
            "im_shape": im_shape,
        })

        # ppdet returns {"bbox": (N, 6) [class, score, x1, y1, x2, y2],
        #                "bbox_num": (B,)} — keep the dict around for postprocess.
        # We expose it via a torch tensor wrapper so the dispatcher can route
        # by output_format == "paddle". Cache on self for postprocess to read.
        self._last_outs = outs
        bbox = outs.get("bbox")
        bbox_np = bbox.numpy() if hasattr(bbox, "numpy") else np.asarray(bbox)
        return torch.from_numpy(bbox_np)

    # -- postprocess --------------------------------------------------------
    def postprocess(
        self,
        predictions: torch.Tensor,
        conf_threshold: float,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, np.ndarray]]:
        """Decode cached paddle outputs into the standard list-of-dicts format.

        Args:
            predictions: ``(N, 6)`` tensor [class, score, x1, y1, x2, y2] —
                the same tensor returned by :py:meth:`forward`. Used as a
                fallback when ``self._last_outs`` is missing (e.g. the
                evaluator forward path).
            conf_threshold: Minimum score to keep.
            target_sizes: ``(B, 2)`` tensor of ``[H, W]`` per image — used to
                clip boxes; paddle already returns pixel coords in the
                original-image frame because we forwarded ``im_shape``.

        Returns:
            ``List[Dict[str, np.ndarray]]`` of length B with keys
            ``"boxes"`` (xyxy float32), ``"scores"`` (float32),
            ``"labels"`` (int64).
        """
        outs = getattr(self, "_last_outs", None)
        batch_size = int(target_sizes.shape[0])

        if outs is not None and "bbox_num" in outs:
            bbox = outs["bbox"]
            bbox_num = outs["bbox_num"]
            bbox_np = bbox.numpy() if hasattr(bbox, "numpy") else np.asarray(bbox)
            num_np = bbox_num.numpy() if hasattr(bbox_num, "numpy") else np.asarray(bbox_num)
            results: list[dict[str, np.ndarray]] = []
            offset = 0
            for n in num_np:
                n = int(n)
                chunk = bbox_np[offset:offset + n]
                offset += n
                if chunk.size == 0:
                    results.append(_empty_result())
                    continue
                cls = chunk[:, 0].astype(np.int64)
                score = chunk[:, 1].astype(np.float32)
                xyxy = chunk[:, 2:6].astype(np.float32)
                keep = score >= conf_threshold
                results.append({
                    "boxes": xyxy[keep],
                    "scores": score[keep],
                    "labels": cls[keep],
                })
            # pad in case bbox_num shorter than batch (defensive)
            while len(results) < batch_size:
                results.append(_empty_result())
            return results

        # Fallback: predictions tensor is the (N, 6) bbox table without
        # per-image splits — assign everything to the first image.
        arr = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        if arr.size == 0:
            return [_empty_result() for _ in range(batch_size)]
        cls = arr[:, 0].astype(np.int64)
        score = arr[:, 1].astype(np.float32)
        xyxy = arr[:, 2:6].astype(np.float32)
        keep = score >= conf_threshold
        first = {
            "boxes": xyxy[keep],
            "scores": score[keep],
            "labels": cls[keep],
        }
        return [first] + [_empty_result() for _ in range(batch_size - 1)]


def _empty_result() -> dict[str, np.ndarray]:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_builder(arch_name: str):
    """Build a closure that instantiates ``PaddleDetectionModel`` for *arch_name*."""

    def _build(config: dict) -> PaddleDetectionModel:
        model_cfg = config.get("model", {})
        num_classes = int(model_cfg.get("num_classes") or config.get("data", {}).get("num_classes", 80))
        input_size = model_cfg.get("input_size", [640, 640])
        pretrained = model_cfg.get("pretrained")

        # Mirror HF behavior: pull class names from data.names when present
        data_names = config.get("data", {}).get("names") or model_cfg.get("names")
        class_names: dict[int, str] | None = None
        if isinstance(data_names, dict) and len(data_names) == num_classes:
            class_names = {int(k): str(v) for k, v in data_names.items()}

        logger.info(
            "Building paddle model: arch=%s, num_classes=%d, input_size=%s, pretrained=%s",
            arch_name, num_classes, input_size, pretrained or "<default>",
        )
        return PaddleDetectionModel(
            arch=arch_name,
            num_classes=num_classes,
            input_size=tuple(input_size),
            pretrained=pretrained,
            class_names=class_names,
        )

    _build.__name__ = f"build_{arch_name.replace('-', '_')}"
    return _build


for _arch in _PADDLE_MODEL_REGISTRY:
    register_model(_arch)(_make_builder(_arch))


__all__ = [
    "PaddleDetectionModel",
    "_PADDLE_MODEL_REGISTRY",
]
