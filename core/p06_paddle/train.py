"""Train a PaddlePaddle model — runs in `.venv-paddle/`.

Usage:
    .venv-paddle/bin/python core/p06_paddle/train.py \\
        --config configs/_test/06_training_paddle_det.yaml \\
        --override training.epochs=1

Detection (PicoDet, PP-YOLOE) is the canonical path for v1. Classification,
segmentation and keypoint will follow the same pattern: each dispatches to its
upstream Trainer/Engine. The driver translates our 06_training_paddle_*.yaml
into upstream patches via core/p06_paddle/_translator.py.

After training, writes:
    <save_dir>/best.pdparams
    <save_dir>/test_results.json   (sentinel summary)
    <save_dir>/{data_preview,val_predictions,test_predictions}/
The observability tree subdirectories are populated post-train by the main-venv
ONNX path (run `.venv-paddle/bin/python core/p06_paddle/export.py` then
`uv run core/p08_evaluation/evaluate.py --model <onnx>`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path (same convention as core/p06_training/train.py).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.p06_paddle._translator import (  # noqa: E402
    detection_overrides,
    load_our_yaml,
    ppdet_base_config_path,
)
from utils.config import merge_configs, parse_overrides  # noqa: E402
from utils.paddle_bridge import yolo_to_coco  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaddlePaddle native trainer")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--override", nargs="*", default=[],
                   help="key=value pairs (e.g. training.epochs=1 data.batch_size=8)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Optional .pdparams checkpoint to resume from")
    return p.parse_args()


def _ensure_coco_annotations(our: dict) -> dict[str, Path]:
    data_dir = (our["_data_path"].parent / our["_data_resolved"]["path"]).resolve()
    cache = {split: data_dir / f"{split}_paddle_coco.json" for split in ("train", "val")}
    if all(p.exists() for p in cache.values()):
        return cache
    out_files = yolo_to_coco(
        data_config_path=str(our["_data_path"]),
        output_dir=str(data_dir),
        splits=["train", "val"],
    )
    for split, src in out_files.items():
        target = cache[split]
        if Path(src).resolve() != target.resolve():
            target.write_bytes(Path(src).read_bytes())
        print(f"  converted {split}: {target.name}", flush=True)
    return cache


def _train_detection(our: dict) -> dict:
    import paddle
    from ppdet.core.workspace import load_config, merge_config
    from ppdet.engine import Trainer

    base_path = _find_ppdet_config(ppdet_base_config_path(our["model"]["arch"]))
    cfg = load_config(str(base_path))
    merge_config(detection_overrides(our))

    save_dir = Path(cfg["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    pretrained = our.get("model", {}).get("pretrained")
    if pretrained is not None:
        cfg["pretrain_weights"] = str(pretrained)
    elif "pretrain_weights" in cfg:
        cfg["pretrain_weights"] = None

    trainer = Trainer(cfg, mode="train")
    if pretrained:
        trainer.load_weights(pretrained)
    trainer.train()

    best = _resolve_best_checkpoint(save_dir)
    canonical = save_dir / "best.pdparams"
    if best is not None and best != canonical:
        canonical.write_bytes(best.read_bytes())

    metrics = _evaluate_after_train(trainer) if canonical.exists() else {}
    _ensure_observability_tree(save_dir)

    summary = {
        "metrics": metrics,
        "best_metric": metrics.get("bbox_mAP", metrics.get("mAP", 0.0)),
        "best_epoch": int(our.get("training", {}).get("epochs", 1)),
        "total_epochs": int(our.get("training", {}).get("epochs", 1)),
        "output_dir": str(save_dir),
        "best_checkpoint": str(canonical) if canonical.exists() else None,
        "framework": f"paddle {paddle.__version__}",
    }
    (save_dir / "test_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def _find_ppdet_config(rel_path: str) -> Path:
    """Locate `<rel_path>` (e.g. `configs/picodet/...yml`) for PaddleDetection.

    pip-installed ppdet strips configs/ + tools/, so setup-paddle-venv.sh clones
    the upstream repo to .venv-paddle/PaddleDetection/. Search both locations.
    """
    import ppdet  # local import — only resolves under .venv-paddle/
    pkg_root = Path(ppdet.__file__).resolve().parent
    venv_root = pkg_root.parents[3]  # .venv-paddle/
    candidates = [
        venv_root / "PaddleDetection" / rel_path,
        pkg_root / rel_path,
        pkg_root.parent / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"PaddleDetection base config not found: {rel_path}\n"
        f"Searched: {candidates}\n"
        f"Run: bash scripts/setup-paddle-venv.sh (re-clones PaddleDetection)"
    )


def _resolve_best_checkpoint(save_dir: Path) -> Path | None:
    for name in ("best_model.pdparams", "model_final.pdparams"):
        p = save_dir / name
        if p.exists():
            return p
    snaps = sorted(save_dir.glob("*.pdparams"))
    return snaps[-1] if snaps else None


def _scrape_ppdet_metrics(trainer) -> dict:
    """ppdet keeps eval stats on different attrs across versions — try each."""
    for attr in ("_eval_metrics", "metric_results", "_metrics"):
        v = getattr(trainer, attr, None)
        if isinstance(v, dict) and v:
            return {str(k): float(v[k]) for k in v if isinstance(v[k], (int, float))}
    return {}


def _evaluate_after_train(trainer) -> dict:
    try:
        trainer.evaluate()
    except Exception as exc:
        print(f"  warning: evaluate after train raised: {exc}", flush=True)
        return {}
    return _scrape_ppdet_metrics(trainer)


def _ensure_observability_tree(save_dir: Path) -> None:
    """Empty subdirectories that downstream consumers (releases/, p08 rerun) walk."""
    for sub in ("data_preview", "val_predictions", "test_predictions"):
        (save_dir / sub).mkdir(parents=True, exist_ok=True)
    (save_dir / "val_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)
    (save_dir / "test_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)


_TASK_DISPATCH = {
    "paddle-picodet-": _train_detection,
    "paddle-ppyoloe-": _train_detection,
}


def _dispatch(our: dict) -> dict:
    arch = our["model"]["arch"]
    for prefix, fn in _TASK_DISPATCH.items():
        if arch.startswith(prefix):
            return fn(our)
    raise NotImplementedError(
        f"paddle backend has no trainer for arch {arch!r} yet. "
        f"v1 supports detection only ({', '.join(_TASK_DISPATCH)}). "
        f"PaddleClas / PaddleSeg / PP-TinyPose drivers can be added under "
        f"core/p06_paddle/train.py following the _train_detection pattern."
    )


def main() -> int:
    args = _parse_args()
    our = load_our_yaml(args.config)
    if args.override:
        our = merge_configs(our, parse_overrides(args.override))

    print(f"[paddle] config: {args.config}", flush=True)
    print(f"[paddle] arch:   {our['model']['arch']}", flush=True)
    print("[paddle] task:   detection (v1)", flush=True)

    _ensure_coco_annotations(our)
    summary = _dispatch(our)

    print("\n[paddle] training complete. summary:", flush=True)
    print(f"  best_checkpoint: {summary.get('best_checkpoint')}", flush=True)
    print(f"  best_metric:     {summary.get('best_metric')}", flush=True)
    print(f"  output_dir:      {summary.get('output_dir')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
