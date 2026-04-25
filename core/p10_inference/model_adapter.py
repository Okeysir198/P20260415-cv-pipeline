"""ModelAdapter — pluggable inference backend for error analysis.

Every concrete adapter wraps one model format and exposes a uniform
``predict_batch`` interface (BGR numpy in → list-of-dicts out).

To add a new format (e.g. PaddlePaddle):

1. Subclass :class:`ModelAdapter`.
2. Implement :meth:`can_handle` and :meth:`predict_batch`.
3. Decorate with ``@register_adapter``.
4. Either import your module before calling :func:`resolve_adapter`, or add
   the import to :func:`_load_builtin_adapters` at the bottom of this file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_ADAPTER_REGISTRY: list[type[ModelAdapter]] = []


def register_adapter(cls: type[ModelAdapter]) -> type[ModelAdapter]:
    """Class decorator — appends *cls* to the global adapter registry."""
    _ADAPTER_REGISTRY.append(cls)
    return cls


def resolve_adapter(
    model_path: str,
    data_config: dict,
    training_config: dict | None,
    *,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    device: Any | None = None,
) -> ModelAdapter:
    """Return the first registered adapter that can handle *model_path*.

    Adapters are tried in registration order.  Raises :class:`ValueError`
    if none match.
    """
    _load_builtin_adapters()
    for cls in _ADAPTER_REGISTRY:
        if cls.can_handle(model_path, data_config, training_config):
            logger.info("Using adapter {} for {}", cls.__name__, model_path)
            return cls(
                model_path=model_path,
                data_config=data_config,
                training_config=training_config,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
            )
    raise ValueError(
        f"No adapter can handle model_path={model_path!r}. "
        "Register a custom adapter with @register_adapter."
    )


class ModelAdapter(ABC):
    """Abstract base for all model format adapters.

    Subclasses receive the constructor arguments below; unused args should be
    accepted via ``**kwargs`` so the registry can forward them uniformly.
    """

    def __init__(
        self,
        model_path: str,
        data_config: dict,
        training_config: dict | None,
        conf_threshold: float,
        iou_threshold: float,
        device: Any | None,
    ) -> None: ...

    @abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """Run inference on a batch of BGR uint8 HWC images.

        Returns:
            One dict per image with keys:
            - ``boxes``:  (N, 4) float32 xyxy pixel coords in *original* image space
            - ``scores``: (N,) float32
            - ``labels``: (N,) int64
        """

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        model_path: str,
        data_config: dict,
        training_config: dict | None,
    ) -> bool:
        """Return True when this adapter is the right choice for *model_path*."""


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


@register_adapter
class PredictorAdapter(ModelAdapter):
    """Wraps :class:`core.p10_inference.predictor.DetectionPredictor`.

    Handles ``.pth``, ``.pt``, and ``.onnx`` checkpoints.
    """

    def __init__(self, model_path, data_config, training_config, conf_threshold,
                 iou_threshold, device) -> None:
        from core.p10_inference.predictor import DetectionPredictor  # noqa: PLC0415

        self._predictor = DetectionPredictor(
            model_path=model_path,
            data_config=data_config,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
        )

    @classmethod
    def can_handle(cls, model_path, data_config, training_config) -> bool:
        return Path(model_path).suffix.lower() in {".pth", ".pt", ".onnx"}

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        results = self._predictor.predict_batch(images)
        return [
            {
                "boxes": r["boxes"].astype(np.float32),
                "scores": r["scores"].astype(np.float32),
                "labels": r["labels"].astype(np.int64),
            }
            for r in results
        ]


@register_adapter
class HFAdapter(ModelAdapter):
    """Wraps a HuggingFace Hub object-detection model.

    Accepts either a Hub repo ID (``"PekingU/rtdetr_v2_r18vd"``) or a local
    directory containing ``config.json``.
    """

    def __init__(self, model_path, data_config, training_config, conf_threshold,
                 iou_threshold, device) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: PLC0415

        self._conf = conf_threshold
        self._device = torch.device(device) if isinstance(device, str) else (
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._processor = AutoImageProcessor.from_pretrained(model_path)

        # Our HF Trainer saves pytorch_model.bin with "hf_model." prefix
        # (HFModelWrapper.state_dict()). from_pretrained() silently reinitialises
        # all weights when keys don't match. Strip the prefix first, write a
        # temp bin, then load — avoids touching the original checkpoint.
        p = Path(model_path)
        bin_path = p / "pytorch_model.bin" if p.is_dir() else None
        if bin_path and bin_path.exists():
            sd = torch.load(bin_path, map_location="cpu", weights_only=False)
            if any(k.startswith("hf_model.") for k in sd):
                import tempfile, shutil  # noqa: PLC0415, E401
                sd_clean = {k.removeprefix("hf_model."): v for k, v in sd.items()}
                tmp_dir = Path(tempfile.mkdtemp())
                shutil.copy(p / "config.json", tmp_dir / "config.json")
                if (p / "preprocessor_config.json").exists():
                    shutil.copy(p / "preprocessor_config.json", tmp_dir / "preprocessor_config.json")
                torch.save(sd_clean, tmp_dir / "pytorch_model.bin")
                logger.info("Stripped hf_model. prefix — loading from temp dir {}", tmp_dir)
                self._model = AutoModelForObjectDetection.from_pretrained(str(tmp_dir))
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                self._model = AutoModelForObjectDetection.from_pretrained(model_path)
        else:
            self._model = AutoModelForObjectDetection.from_pretrained(model_path)

        self._model.to(self._device).eval()

    @classmethod
    def can_handle(cls, model_path, data_config, training_config) -> bool:
        p = Path(model_path)
        if p.suffix:
            return False
        return "/" in model_path or (p.is_dir() and (p / "config.json").exists())

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        target_sizes = torch.tensor(
            [[img.shape[0], img.shape[1]] for img in images], device=self._device
        )
        inputs = self._processor(images=rgb_images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs, threshold=self._conf, target_sizes=target_sizes
        )
        return [
            {
                "boxes": r["boxes"].cpu().numpy().astype(np.float32),
                "scores": r["scores"].cpu().numpy().astype(np.float32),
                "labels": r["labels"].cpu().numpy().astype(np.int64),
            }
            for r in results
        ]


@register_adapter
class TorchScriptAdapter(ModelAdapter):
    """Placeholder for raw TorchScript models.

    :class:`PredictorAdapter` (registered first) already handles ``.pt``
    files via :class:`~core.p10_inference.predictor.DetectionPredictor`, which
    internally supports TorchScript.  This adapter is intentionally a stub to
    show where you would add support for TorchScript models that do **not**
    follow the DetectionPredictor output contract (e.g. a custom export with a
    non-standard output tensor layout).

    To extend: override ``__init__`` to call ``torch.jit.load``, implement
    ``predict_batch`` with your own decode logic, and change ``can_handle`` to
    return ``True`` before ``PredictorAdapter`` is registered (register this
    adapter first).
    """

    def __init__(self, model_path, data_config, training_config, conf_threshold,
                 iou_threshold, device) -> None:
        raise NotImplementedError(
            "TorchScriptAdapter is a stub. Subclass it for models that do not "
            "follow the DetectionPredictor output contract."
        )

    @classmethod
    def can_handle(cls, model_path, data_config, training_config) -> bool:
        # PredictorAdapter is registered first and takes all .pt files; this
        # adapter intentionally never matches in the default registration order.
        return False

    def predict_batch(self, images):  # type: ignore[override]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lazy import guard — keeps the module importable without heavy dependencies
# until resolve_adapter() is actually called.
# ---------------------------------------------------------------------------

_builtins_loaded = False


def _load_builtin_adapters() -> None:
    """No-op: built-in adapters self-register via @register_adapter at import
    time.  Add third-party adapter imports here so they register before
    resolve_adapter() runs its loop."""
    pass
