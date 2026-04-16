"""Detection model predictor — single-image and batch inference with PyTorch or ONNX backends.

Supports .pt (PyTorch checkpoint) and .onnx (ONNX Runtime) model files.
Handles preprocessing, model inference, NMS postprocessing, and result
formatting for downstream alert logic and visualization.

Works with any model registered in the model registry (YOLOX, D-FINE,
RT-DETRv2, etc.).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p06_models import build_model
from core.p06_training.postprocess import postprocess as _postprocess_registry
from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from utils.device import get_device

logger = logging.getLogger(__name__)


def _remap_megvii_state_dict(
    src: dict, target_keys: set[str]
) -> dict:
    """Remap official Megvii YOLOX checkpoint keys to our model's naming.

    Megvii structure → Our structure:
    - ``backbone.backbone.*`` → ``backbone.*`` (CSPDarknet)
    - ``backbone.<pafpn_layer>`` → ``neck.<pafpn_layer>`` (PAFPN)
    - ``head.<layer>.N`` → ``heads.N.<layer>`` (per-scale decoupled head)
    - ``.m.`` → ``.blocks.`` (CSPLayer bottleneck list)
    - ``.preds.`` → ``.pred.`` (singular prediction layers)
    """
    import re

    # PAFPN layer names (checkpoint uses backbone.X, our model uses neck.X)
    _PAFPN_LAYERS = {
        "lateral_conv0", "reduce_conv1", "bu_conv1", "bu_conv2",
        "C3_p3", "C3_p4", "C3_n3", "C3_n4",
    }

    mapped: dict = {}
    for key, value in src.items():
        new_key = key

        # 1. CSPDarknet: backbone.backbone.* → backbone.*
        if new_key.startswith("backbone.backbone."):
            new_key = new_key.replace("backbone.backbone.", "backbone.", 1)

        # 2. PAFPN: backbone.<pafpn_layer>.* → neck.<pafpn_layer>.*
        elif new_key.startswith("backbone."):
            suffix = new_key[len("backbone."):]
            layer_name = suffix.split(".")[0]
            if layer_name in _PAFPN_LAYERS:
                new_key = "neck." + suffix

        # 3. Head: head.<layer>.N.* → heads.N.<layer>.*
        #    e.g. head.cls_convs.0.0.conv.weight → heads.0.cls_convs.0.conv.weight
        #    e.g. head.cls_preds.0.weight → heads.0.cls_pred.weight
        if new_key.startswith("head."):
            parts = new_key.split(".")
            # parts[0] = "head", parts[1] = layer, parts[2] = scale_idx, rest...
            layer = parts[1]
            scale_idx = parts[2]
            rest = ".".join(parts[3:])
            # cls_preds → cls_pred, reg_preds → reg_pred, obj_preds → obj_pred
            if layer.endswith("preds"):
                layer = layer[:-1]  # remove trailing 's'
            new_key = f"heads.{scale_idx}.{layer}"
            if rest:
                new_key += f".{rest}"

        # 4. CSPLayer bottleneck: .m. → .blocks.
        new_key = re.sub(r"\.m\.", ".blocks.", new_key)

        # 5. Head stems plural → singular: .stems. → .stem.
        new_key = new_key.replace(".stems.", ".stem.")

        mapped[new_key] = value

    # 6. BaseConv wrapper unwrapping — only apply when the unwrapped key
    #    exists in the target model (avoids breaking Focus → BaseConv nesting).
    final: dict = {}
    for key, value in mapped.items():
        new_key = key
        if new_key not in target_keys:
            candidate = re.sub(r"\.conv\.conv\.", ".conv.", new_key)
            candidate = re.sub(r"\.conv\.bn\.", ".bn.", candidate)
            if candidate in target_keys:
                new_key = candidate
        final[new_key] = value
    mapped = final

    # Log any remaining mismatches
    mapped_keys = set(mapped.keys())
    missing = target_keys - mapped_keys
    extra = mapped_keys - target_keys
    if missing:
        logger.warning(
            "Key remap: %d still missing, %d extra — loading with strict=False",
            len(missing), len(extra),
        )
    return mapped


class DetectionPredictor:
    """Run inference with a trained detection model (PyTorch or ONNX).

    Supports any model architecture registered in the model registry
    (YOLOX, D-FINE, RT-DETRv2, etc.).

    Handles: image preprocessing, model forward pass, NMS postprocessing,
    and result formatting.

    Args:
        model_path: Path to ``.pt`` checkpoint or ``.onnx`` model.
        data_config: Data config dict containing at least ``names``
            (dict[int, str]) and ``input_size`` ([H, W]).
        conf_threshold: Confidence threshold for filtering detections.
        iou_threshold: IoU threshold for NMS.
        device: Device string or ``torch.device`` (PyTorch models only).
            Auto-detected when *None*.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        data_config: dict,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[Any] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # --- Parse data config ---
        self.class_names: Dict[int, str] = {
            int(k): v for k, v in data_config["names"].items()
        }
        self.num_classes = data_config.get("num_classes", len(self.class_names))

        input_size = data_config.get("input_size", [640, 640])
        self.input_h = int(input_size[0])
        self.input_w = int(input_size[1])

        # Normalization params (default ImageNet)
        self.mean = np.array(
            data_config.get("mean", IMAGENET_MEAN), dtype=np.float32
        )
        self.std = np.array(
            data_config.get("std", IMAGENET_STD), dtype=np.float32
        )

        # --- Output format (overridden when loading registry-built models) ---
        self._output_format = "yolox"

        # --- Determine backend and load model ---
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            self.backend = "onnx"
            self._session = self._load_onnx_model(str(self.model_path))
            self._onnx_input_name = self._session.get_inputs()[0].name
            self._torch_model = None
            self.device = None
            logger.info("Loaded ONNX model from %s", self.model_path)
        elif suffix in (".pt", ".pth"):
            self.backend = "pytorch"
            if device is None:
                self.device = get_device()
            elif isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
            self._torch_model = self._load_pytorch_model(str(self.model_path))
            self._session = None
            logger.info(
                "Loaded PyTorch model from %s on %s", self.model_path, self.device
            )
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}'. Use .pt/.pth or .onnx."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single BGR image.

        Args:
            image: BGR image as ``np.ndarray`` (H, W, 3) uint8.

        Returns:
            Dict with keys:
                - ``boxes``:  (N, 4) float32 ``[x1, y1, x2, y2]`` in
                  original image coordinates.
                - ``scores``: (N,) float32 confidence scores.
                - ``labels``: (N,) int64 class indices.
                - ``class_names``: (N,) list of class name strings.
        """
        orig_h, orig_w = image.shape[:2]
        preprocessed = self._preprocess(image)

        if self.backend == "pytorch":
            outputs = self._infer_pytorch(preprocessed)
        else:
            outputs = self._infer_onnx(preprocessed)

        # Dispatch postprocessing by output format
        if self._output_format == "classification":
            return self._postprocess_classification(outputs)

        results = self._postprocess(outputs)

        # Scale boxes back to original image size
        if results["boxes"].shape[0] > 0:
            scale_x = orig_w / self.input_w
            scale_y = orig_h / self.input_h
            results["boxes"][:, [0, 2]] *= scale_x
            results["boxes"][:, [1, 3]] *= scale_y
            # Clip to image boundaries
            results["boxes"][:, [0, 2]] = np.clip(
                results["boxes"][:, [0, 2]], 0, orig_w
            )
            results["boxes"][:, [1, 3]] = np.clip(
                results["boxes"][:, [1, 3]], 0, orig_h
            )

        # Add human-readable class names
        results["class_names"] = [
            self.class_names.get(int(lbl), str(int(lbl))) for lbl in results["labels"]
        ]

        return results

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Run inference on a list of BGR images.

        Processes each image independently (no batched GPU forward pass)
        to keep the implementation simple and memory-safe.

        Args:
            images: List of BGR ``np.ndarray`` images.

        Returns:
            List of prediction dicts (same format as :meth:`predict`).
        """
        return [self.predict(img) for img in images]

    def predict_file(self, image_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load an image from disk and run inference.

        Args:
            image_path: Path to an image file (jpg, png, etc.).

        Returns:
            Prediction dict (same format as :meth:`predict`).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be decoded.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        return self.predict(image)

    def visualize(
        self,
        image: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """Draw predictions on an image.

        Args:
            image: Original BGR image.
            predictions: Output from :meth:`predict`.
            save_path: If provided, save the annotated image to this path.

        Returns:
            Annotated image (BGR ``np.ndarray``).
        """
        from core.p08_evaluation.visualization import draw_bboxes

        vis = draw_bboxes(
            image,
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
            self.class_names,
        )
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis)
            logger.info("Saved annotated image to %s", save_path)

        return vis

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize, and transpose a BGR image for YOLOX.

        YOLOX standard preprocessing:
        1. Resize to ``(input_h, input_w)`` with letterboxing-free resize.
        2. Convert BGR -> RGB.
        3. Scale to [0, 1], apply ImageNet normalization.
        4. Transpose HWC -> CHW.
        5. Add batch dimension -> (1, 3, H, W).

        Args:
            image: BGR uint8 image (H, W, 3).

        Returns:
            Preprocessed float32 array of shape (1, 3, input_h, input_w).
        """
        resized = cv2.resize(image, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
        return chw[np.newaxis, ...]  # add batch dim

    # ------------------------------------------------------------------
    # Backend inference
    # ------------------------------------------------------------------

    def _infer_pytorch(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run PyTorch forward pass.

        Args:
            preprocessed: (1, 3, H, W) float32 numpy array.

        Returns:
            Raw model output as numpy: (1, num_anchors, 5 + num_classes).
        """
        tensor = torch.from_numpy(preprocessed).to(self.device)
        with torch.no_grad():
            outputs = self._torch_model(tensor)

        # YOLOX may return a tuple; take the first element (the detection head)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        return outputs.cpu().numpy()

    def _infer_onnx(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime forward pass.

        Args:
            preprocessed: (1, 3, H, W) float32 numpy array.

        Returns:
            Raw model output as numpy: (1, num_anchors, 5 + num_classes).
        """
        outputs = self._session.run(None, {self._onnx_input_name: preprocessed})
        return outputs[0]

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(self, outputs: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode model outputs, apply NMS, and format results.

        Delegates to :func:`~core.p06_training.postprocess.postprocess`,
        which first checks for ``model.postprocess()`` (HF models) and then
        falls back to the registry (YOLOX, DETR, etc.).

        Args:
            outputs: Raw model output (1, num_anchors, 5 + num_classes)
                as a numpy array (from ONNX or PyTorch inference).

        Returns:
            Dict with ``boxes`` (N, 4) xyxy float32, ``scores`` (N,) float32,
            ``labels`` (N,) int64.
        """
        tensor = torch.from_numpy(outputs) if isinstance(outputs, np.ndarray) else outputs
        target_sizes = torch.tensor([[self.input_h, self.input_w]] * tensor.shape[0])

        results = _postprocess_registry(
            output_format=self._output_format,
            model=self._torch_model,
            predictions=tensor,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
            target_sizes=target_sizes,
        )

        return results[0] if results else self._empty_results()

    @staticmethod
    def _empty_results() -> Dict[str, np.ndarray]:
        """Return an empty detection results dict."""
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }

    def _postprocess_classification(self, outputs: np.ndarray) -> Dict[str, Any]:
        """Decode classification logits into class prediction.

        Args:
            outputs: Raw logits (1, num_classes) or (num_classes,).

        Returns:
            Dict with class_id, confidence, probabilities, class_name.
        """
        logits = outputs[0] if outputs.ndim == 2 else outputs  # remove batch dim
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        class_id = int(probs.argmax())
        return {
            "class_id": class_id,
            "confidence": float(probs[class_id]),
            "probabilities": probs.astype(np.float32),
            "class_name": self.class_names.get(class_id, str(class_id)),
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pytorch_model(self, path: str) -> "nn.Module":
        """Load a PyTorch detection checkpoint.

        Supports checkpoint formats:
        1. Registry-aware checkpoint with ``"config"`` key -- builds model
           via :func:`models.build_model` for any registered architecture.
        2. Full checkpoint dict with ``"model"`` key and no config --
           constructs a minimal config from data_config and builds via
           :func:`models.build_model`.
        3. Complete model object (``torch.save(model, ...)``).
        4. TorchScript model (``torch.jit.save``).

        Args:
            path: Path to the ``.pt`` / ``.pth`` file.

        Returns:
            Loaded ``nn.Module`` in eval mode on ``self.device``.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint cannot be loaded.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # --- Build model from checkpoint config via registry ---
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            ckpt_config = checkpoint["config"]
            model = build_model(ckpt_config)
            state_dict = checkpoint.get(
                "model", checkpoint.get("model_state_dict")
            )
            if state_dict is not None:
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._output_format = getattr(model, "output_format", "yolox")
            logger.info(
                "Built model via registry (output_format=%s)",
                self._output_format,
            )
            return model

        # --- State dict without config: build model from data_config ---
        state_dict = None
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

        if state_dict is not None:
            minimal_config = {
                "model": {
                    "arch": "yolox",
                    "num_classes": self.num_classes,
                    "input_size": [self.input_h, self.input_w],
                },
            }
            model = build_model(minimal_config)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                # Official Megvii COCO checkpoint uses different key naming.
                # Remap keys to match our model's state dict.
                model_keys = set(model.state_dict().keys())
                remapped = _remap_megvii_state_dict(state_dict, model_keys)
                model.load_state_dict(remapped, strict=False)
            model.to(self.device)
            model.eval()
            return model

        # --- Complete model object ---
        if isinstance(checkpoint, torch.nn.Module):
            checkpoint.to(self.device)
            checkpoint.eval()
            return checkpoint

        # --- TorchScript ---
        try:
            model = torch.jit.load(path, map_location=self.device)
            model.eval()
            return model
        except Exception:
            pass

        raise RuntimeError(
            f"Unable to load model from {path}. Supported formats: "
            "registry checkpoint with config, state_dict checkpoint, "
            "full model (torch.save(model, ...)), or TorchScript."
        )

    def _load_onnx_model(self, path: str) -> "ort.InferenceSession":
        """Load an ONNX model via ONNX Runtime.

        Args:
            path: Path to the ``.onnx`` file.

        Returns:
            ``ort.InferenceSession`` ready for inference.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        # Prefer GPU provider if available, fall back to CPU
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(path, providers=providers)
        except Exception as e:
            # CUDA init can fail mid-session (e.g. CUBLAS_ALLOC_FAILED on a
            # saturated shared GPU) even when the provider is reported as
            # "available". ORT's provider list does NOT auto-fallback in
            # that case — the constructor raises. Retry explicitly with
            # CPU only so the caller still gets a working session.
            if "CUDAExecutionProvider" not in providers:
                raise
            logger.warning(
                "CUDA init failed (%s); retrying ONNX session with CPU only.",
                str(e).splitlines()[0] if str(e) else type(e).__name__,
            )
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        logger.info("ONNX Runtime providers: %s", session.get_providers())
        return session
