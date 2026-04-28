"""Evaluate a Paddle checkpoint — runs in `.venv-paddle/`.

Usage:
    .venv-paddle/bin/python core/p06_paddle/eval.py \\
        --config configs/_test/06_training_paddle_det.yaml \\
        --checkpoint <save_dir>/best.pdparams

Returns metrics by running upstream `Trainer.evaluate()`. For ONNX-based eval
(after `core/p06_paddle/export.py`), use the standard main-venv path:
    uv run core/p08_evaluation/evaluate.py --model <save_dir>/model.onnx --config <05_data>.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.p06_paddle._translator import (  # noqa: E402
    detection_overrides,
    load_our_yaml,
    ppdet_base_config_path,
)
from core.p06_paddle.train import _find_ppdet_config, _ensure_coco_annotations  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaddlePaddle native evaluator")
    p.add_argument("--config", required=True, type=Path,
                   help="06_training_paddle_*.yaml")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help=".pdparams checkpoint")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    our = load_our_yaml(args.config)

    arch = our["model"]["arch"]
    if not (arch.startswith("paddle-picodet-") or arch.startswith("paddle-ppyoloe-")):
        raise NotImplementedError(
            f"paddle eval v1 supports detection only — got arch={arch!r}"
        )

    _ensure_coco_annotations(our)

    from ppdet.core.workspace import load_config, merge_config
    from ppdet.engine import Trainer

    base_path = _find_ppdet_config(ppdet_base_config_path(arch))
    cfg = load_config(str(base_path))
    merge_config(detection_overrides(our))

    trainer = Trainer(cfg, mode="eval")
    trainer.load_weights(str(args.checkpoint))
    trainer.evaluate()

    # Best-effort metric scrape (ppdet API varies by version)
    metrics = {}
    for attr in ("_eval_metrics", "metric_results", "_metrics"):
        v = getattr(trainer, attr, None)
        if isinstance(v, dict):
            metrics.update({str(k): float(v[k]) for k in v if isinstance(v[k], (int, float))})
            break
    print(json.dumps({"metrics": metrics, "checkpoint": str(args.checkpoint)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
