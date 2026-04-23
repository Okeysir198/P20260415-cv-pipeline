"""Normalize-verification callback.

Provides :class:`NormalizeCheckCallback` — dual-backend callback that fires
once on ``on_train_start`` (pytorch) / ``on_train_begin`` (HF) and writes
``<save_dir>/data_preview/04_normalize_check.png``.

The PNG has 3 columns per sample:
  1. Raw (from disk + GT)
  2. Normalized tensor (model input, jet false-color ±3σ)
  3. Denormalized (inverse Normalize + GT — should look like a valid
     augmented image)

Combines the previous ``NormalizedInputPreviewCallback`` (stage-3 denormalize
sanity check) and ``TransformPipelineCallback`` (step-walk) into one focused
artifact. Catches:

- Double-normalize / missing-normalize pipelines (col 3 wrong colour cast).
- Box-format drift from cxcywh-normalized to pixel space (col 3 boxes wrong).
- Broken inverse-normalize algebra (col 3 clamped / saturated).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _AnyHook:
    """Base that turns every attribute access into a permissive no-op method.

    Lets one callback class satisfy both the pytorch CallbackRunner surface
    and the HF ``TrainerCallback`` surface without tracking either API's
    growing list. Only the hooks we actually implement override this.
    """

    def __getattr__(self, name):
        if name.startswith("on_"):
            def _noop(*args, **kwargs):
                return kwargs.get("control")  # HF expects control back; pytorch ignores
            return _noop
        raise AttributeError(name)


class NormalizeCheckCallback(_AnyHook):
    """Writes ``data_preview/04_normalize_check.png`` — 3-col per-sample
    verification: raw (disk + GT) | normalized tensor (jet false-color) |
    denormalized (inverse + GT).

    Fires once on train-start for both backends. Builds its own
    ``YOLOXDataset`` instances internally so it always reflects the real
    train-time transform pipeline (no dependency on the trainer's loader).
    """

    def __init__(
        self,
        save_dir: str | Path,
        *,
        data_config: dict,
        training_config: dict,
        base_dir: str,
        class_names: dict[int, str] | None = None,
        num_samples: int = 4,
        style: Any | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.training_config = training_config
        self.base_dir = base_dir
        self.class_names = class_names or {}
        self.num_samples = num_samples
        self.style = style

    def _render(self) -> None:
        from core.p05_data.transform_pipeline_viz import render_normalize_check

        render_normalize_check(
            out_path=self.save_dir / "data_preview" / "04_normalize_check.png",
            dataset=None,
            data_config=self.data_config,
            training_config=self.training_config,
            base_dir=self.base_dir,
            class_names=self.class_names,
            num_samples=self.num_samples,
            style=self.style,
        )

    # -------- pytorch backend --------
    def on_train_start(self, trainer):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover — never block training
            logger.warning("NormalizeCheckCallback skipped (pytorch): %s", e)

    # -------- HF backend --------
    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover
            logger.warning("NormalizeCheckCallback skipped (hf): %s", e)
        return control
