"""Transform-pipeline verification callback.

Provides :class:`TransformPipelineCallback` — dual-backend callback that fires
once on ``on_train_start`` (pytorch) / ``on_train_begin`` (HF) and writes
``<save_dir>/data_preview/04_transform_pipeline.png``.

The PNG is a ``K × N`` grid: one representative sample per class (first
occurrence in the train split, up to ``max_samples`` classes) walked through
every CPU transform step. Final column is ``Denormalize(Normalize)`` — a
visual sanity check on the algebraic inverse.

Catches:
  * double-normalize / missing-normalize pipelines (last col colour cast)
  * box-format drift (pixel ↔ cxcywh) across stages
  * broken inverse-normalize algebra (last col clamped/saturated)
  * surprise per-step aug output (visible per cell)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _AnyHook:
    """Base that turns every attribute access into a permissive no-op method.

    Lets one callback class satisfy both the pytorch ``CallbackRunner``
    surface and the HF ``TrainerCallback`` surface without tracking either
    API's growing list. Only the hooks we implement override this.
    """

    def __getattr__(self, name):
        if name.startswith("on_"):
            def _noop(*args, **kwargs):
                return kwargs.get("control")
            return _noop
        raise AttributeError(name)


class TransformPipelineCallback(_AnyHook):
    """Writes ``data_preview/04_transform_pipeline.png``.

    Fires once on train-start for both backends. Builds its own
    ``YOLOXDataset`` internally so it always reflects the real train-time
    pipeline (no dependency on the trainer's loader).
    """

    def __init__(
        self,
        save_dir: str | Path,
        *,
        data_config: dict,
        training_config: dict,
        base_dir: str,
        class_names: dict[int, str] | None = None,
        max_samples: int = 5,
        style: Any | None = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.data_config = data_config
        self.training_config = training_config
        self.base_dir = base_dir
        self.class_names = class_names or {}
        self.max_samples = max_samples
        self.style = style

    def _render(self) -> None:
        from core.p05_data.transform_pipeline_viz import render_transform_pipeline

        render_transform_pipeline(
            out_path=self.save_dir / "data_preview" / "04_transform_pipeline.png",
            data_config=self.data_config,
            training_config=self.training_config,
            base_dir=self.base_dir,
            class_names=self.class_names,
            max_samples=self.max_samples,
            style=self.style,
        )

    # -------- pytorch backend --------
    def on_train_start(self, trainer):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover
            logger.warning("TransformPipelineCallback skipped (pytorch): %s", e)

    # -------- HF backend --------
    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ARG002
        try:
            self._render()
        except Exception as e:  # pragma: no cover
            logger.warning("TransformPipelineCallback skipped (hf): %s", e)
        return control
