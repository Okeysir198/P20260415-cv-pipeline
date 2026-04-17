"""Inpainter — lazy-loading wrapper for HuggingFace diffusion inpainting models.

Provides a unified interface for generating inpainted image variants using either
a locally-loaded Flux img2img pipeline (with client-side mask compositing) or an
external HTTP orchestrator service.  All heavy imports (torch, diffusers, PIL) are
guarded so the module can be imported in CPU-only / dependency-light environments
without errors.
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

import requests
import torch
from PIL import Image as PILImage

from utils.yolo_io import pil_to_b64

logger = logging.getLogger(__name__)

# Map config string to torch dtype
_DTYPE_MAP: Dict[str, Any] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class Inpainter:
    """Lazy-loading wrapper for diffusion-based inpainting.

    Supports two modes of operation:

    - **local**: Loads a Flux img2img pipeline on-device and runs inference
      directly, then composites with mask.
    - **service**: Sends image + mask to the Image Editor orchestrator service
      which handles SAM3 segmentation + Flux NIM generation.

    Models are loaded on first use and can be explicitly unloaded to free
    GPU memory.

    Args:
        config: Configuration dictionary with inpainting settings.  Expected
            structure mirrors the YAML example::

                inpainting:
                  mode: "local"
                  model: "black-forest-labs/FLUX.2-klein-4B"
                  device: null
                  torch_dtype: "bfloat16"
                  service_url: "http://localhost:8002"
                  defaults:
                    strength: 0.85
                    guidance_scale: 3.5
                    num_variants: 1
                    steps: 4
                    seed: null

            When *config* is ``None`` or empty, sensible defaults are used.

    Example::

        inpainter = Inpainter({"inpainting": {"mode": "local"}})
        results = inpainter.inpaint(image, mask, "a fire hydrant on the sidewalk")
        inpainter.unload()
    """

    _DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = (config or {}).get("inpainting", config or {})

        self._mode: str = cfg.get("mode", "local")
        self._model_id: str = cfg.get("model", self._DEFAULT_MODEL)
        self._torch_dtype_str: str = cfg.get("torch_dtype", "bfloat16")
        self._service_url: str = cfg.get("service_url", "http://localhost:8002")
        self._device: str = self._resolve_device(cfg.get("device"))

        # Default generation parameters
        defaults = cfg.get("defaults", {})
        self._default_strength: float = defaults.get("strength", 0.85)
        self._default_guidance_scale: float = defaults.get("guidance_scale", 3.5)
        self._default_num_variants: int = defaults.get("num_variants", 1)
        self._default_steps: int = defaults.get("steps", 4)
        self._default_seed: Optional[int] = defaults.get("seed")

        # Lazy-loaded pipeline slot
        self._pipeline = None

        logger.info(
            "Inpainter initialised — mode=%s, model=%s, device=%s",
            self._mode,
            self._model_id,
            self._device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        image: "PILImage.Image",  # type: ignore[name-defined]
        mask: Any,
        prompt: str,
        *,
        strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_variants: Optional[int] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> List[Any]:
        """Generate inpainted image variant(s).

        Args:
            image: PIL Image to inpaint.
            mask: Inpainting mask — either a boolean ``np.ndarray`` of shape
                ``(H, W)`` or a PIL Image (white = inpaint region).
            prompt: Text description of the desired content in the masked area.
            strength: Denoising strength in ``[0, 1]``.
            guidance_scale: Classifier-free guidance scale.
            num_variants: Number of image variants to generate.
            seed: Random seed for reproducibility.
            steps: Number of inference steps (Flux default: 4).

        Returns:
            List of PIL Images — one per requested variant.
        """
        _strength = strength if strength is not None else self._default_strength
        _guidance = guidance_scale if guidance_scale is not None else self._default_guidance_scale
        _num = num_variants if num_variants is not None else self._default_num_variants
        _seed = seed if seed is not None else self._default_seed
        _steps = steps if steps is not None else self._default_steps

        # Normalise mask to PIL Image (white = inpaint)
        mask_pil = self._prepare_mask(mask)

        if self._mode == "service":
            return self._inpaint_service(
                image, mask_pil, prompt,
                num_variants=_num,
                seed=_seed,
                steps=_steps,
            )
        else:
            return self._inpaint_local(
                image, mask_pil, prompt,
                strength=_strength,
                guidance_scale=_guidance,
                num_variants=_num,
                seed=_seed,
                steps=_steps,
            )

    def unload(self) -> None:
        """Free GPU memory by unloading the inpainting pipeline."""
        self._pipeline = None

        torch.cuda.empty_cache()

        logger.info("Unloaded inpainting pipeline")

    # ------------------------------------------------------------------
    # Private: local inference — Flux img2img + mask composite
    # ------------------------------------------------------------------

    def _inpaint_local(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        *,
        strength: float,
        guidance_scale: float,
        num_variants: int,
        seed: Optional[int],
        steps: int,
    ) -> List[Any]:
        """Run inpainting locally using Flux img2img + mask composite."""
        self._load_pipeline()

        results: List[Any] = []
        for i in range(num_variants):
            variant_seed = seed + i if seed is not None else None
            # Flux requires generator on CPU
            generator = None
            if variant_seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(variant_seed)

            output = self._pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            )
            edited = output.images[0]

            # Composite: only replace masked region
            result = self._mask_composite(image, edited, mask)
            results.append(result)

        logger.debug(
            "Generated %d variant(s) for prompt: %s", len(results), prompt[:80]
        )
        return results

    def _load_pipeline(self) -> None:
        """Lazy-load the Flux img2img pipeline."""
        if self._pipeline is not None:
            return

        from diffusers import FluxImg2ImgPipeline

        dtype = _DTYPE_MAP.get(self._torch_dtype_str, torch.bfloat16)

        logger.info("Loading Flux img2img pipeline: %s", self._model_id)
        self._pipeline = FluxImg2ImgPipeline.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
        ).to(self._device)
        logger.info("Flux pipeline loaded on %s", self._device)

    # ------------------------------------------------------------------
    # Private: service mode
    # ------------------------------------------------------------------

    def _inpaint_service(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        *,
        num_variants: int,
        seed: Optional[int],
        steps: int,
    ) -> List[Any]:
        """Send inpainting request to the Image Editor orchestrator service."""
        url = f"{self._service_url.rstrip('/')}/inpaint"

        image_b64 = pil_to_b64(image)
        mask_b64 = pil_to_b64(mask)

        payload = {
            "image": image_b64,
            "mask": mask_b64,
            "prompt": prompt,
            "num_variants": num_variants,
            "seed": seed,
            "steps": steps,
        }

        logger.info("Sending inpainting request to %s", url)
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        images_b64 = data.get("images", [])

        results: List[Any] = []
        for img_b64 in images_b64:
            img_bytes = base64.b64decode(img_b64)
            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            results.append(img)

        logger.debug(
            "Received %d variant(s) from service for prompt: %s",
            len(results),
            prompt[:80],
        )
        return results

    # ------------------------------------------------------------------
    # Mask composite helper
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_composite(
        original: "PILImage.Image",
        edited: "PILImage.Image",
        mask: "PILImage.Image",
    ) -> "PILImage.Image":
        """Blend original and edited images using mask (white = use edited).

        Since Flux does not support native inpainting, we run Flux img2img on
        the full image and then composite the result back using the mask so
        that only the masked region is replaced.

        Args:
            original: Original PIL Image.
            edited: Edited PIL Image from Flux.
            mask: Grayscale PIL mask (white=255 = inpaint region).

        Returns:
            Composited PIL Image.
        """
        # Ensure same size
        if edited.size != original.size:
            edited = edited.resize(original.size, PILImage.Resampling.LANCZOS)
        if mask.size != original.size:
            mask = mask.resize(original.size, PILImage.Resampling.NEAREST)

        # Convert to arrays
        mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
        alpha = mask_arr[:, :, np.newaxis]

        orig_arr = np.array(original, dtype=np.float32)
        edit_arr = np.array(edited, dtype=np.float32)

        result = orig_arr * (1.0 - alpha) + edit_arr * alpha
        return PILImage.fromarray(result.astype(np.uint8))

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        """Resolve the compute device string.

        Args:
            device: Explicit device string, or *None* for auto-detection.

        Returns:
            Device string suitable for ``.to()``.
        """
        if device is not None:
            return device

        from utils.device import get_device

        return str(get_device())

    @staticmethod
    def _prepare_mask(mask: Any) -> Any:
        """Convert a mask to PIL Image format (white = inpaint region).

        Args:
            mask: Either a boolean ``np.ndarray`` of shape ``(H, W)`` or a
                PIL Image.

        Returns:
            PIL Image mask with white (255) for inpaint regions and black (0)
            for preserve regions.
        """
        if isinstance(mask, np.ndarray):
            # Convert boolean mask to uint8 (True -> 255, False -> 0)
            mask_uint8 = (mask.astype(np.uint8)) * 255
            return PILImage.fromarray(mask_uint8, mode="L")

        # Assume it's already a PIL Image
        return mask


