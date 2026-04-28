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
import os
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
from utils.paddle_bridge import yolo_to_coco  # noqa: E402  (existing converter)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaddlePaddle native trainer")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--override", nargs="*", default=[],
                   help="key=value pairs (e.g. training.epochs=1 data.batch_size=8)")
    p.add_argument("--resume", type=Path, default=None,
                   help="Optional .pdparams checkpoint to resume from")
    return p.parse_args()


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    """In-place dotted-path overrides (mirrors utils.config.apply_overrides convention)."""
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"Bad override (missing '='): {kv!r}")
        path, value = kv.split("=", 1)
        # Coerce simple types
        if value.lower() in {"true", "false"}:
            v: object = value.lower() == "true"
        elif value.lstrip("-").replace(".", "", 1).isdigit():
            v = float(value) if "." in value else int(value)
        else:
            v = value
        node = config
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = v


def _ensure_coco_annotations(our: dict) -> dict[str, Path]:
    """Convert YOLO-format labels to COCO JSON. Idempotent — `yolo_to_coco`
    skips writing if the cache already exists in our patched annotation block,
    but here we always invoke it (the helper is fast on cache hit).
    """
    data_dir = (our["_data_path"].parent / our["_data_resolved"]["path"]).resolve()
    cache = {split: data_dir / f"{split}_paddle_coco.json" for split in ("train", "val")}
    # Skip if all caches exist; otherwise regenerate everything (the helper
    # writes all splits at once).
    if not all(p.exists() for p in cache.values()):
        out_files = yolo_to_coco(
            data_config_path=str(our["_data_path"]),
            output_dir=str(data_dir),
            splits=["train", "val"],
        )
        for split, p in out_files.items():
            target = cache[split] if split in cache else (data_dir / f"{split}_paddle_coco.json")
            if Path(p).resolve() != target.resolve():
                target.write_bytes(Path(p).read_bytes())
            print(f"  converted {split}: {target.name}", flush=True)
    return cache


def _train_detection(our: dict) -> dict:
    """Run upstream PaddleDetection Trainer for detection archs."""
    # Lazy paddle imports — only available in .venv-paddle/.
    import paddle
    from ppdet.core.workspace import load_config, merge_config
    from ppdet.engine import Trainer

    base_yml = ppdet_base_config_path(our["model"]["arch"])
    base_path = _find_ppdet_config(base_yml)
    cfg = load_config(str(base_path))

    patches = detection_overrides(our)
    merge_config(patches)

    save_dir = Path(cfg["save_dir"]).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optional pretrained weights — let ppdet handle the URL/path; if our config
    # sets pretrained: null, blank it on the merged config so train-from-scratch
    # works for tests.
    pretrained = our.get("model", {}).get("pretrained")
    if pretrained is not None:
        cfg["pretrain_weights"] = str(pretrained)
    elif "pretrain_weights" in cfg:
        cfg["pretrain_weights"] = None

    trainer = Trainer(cfg, mode="train")
    if pretrained:
        trainer.load_weights(pretrained)
    trainer.train()

    # ppdet writes checkpoint files keyed by epoch under save_dir. The "best"
    # checkpoint (per save_prediction_only convention) is `best_model.pdparams`
    # for newer ppdet, or the last-epoch snapshot otherwise. Symlink/copy to
    # the canonical name our downstream phases expect: `best.pdparams`.
    best = _resolve_best_checkpoint(save_dir, trainer)
    canonical = save_dir / "best.pdparams"
    if best is not None and best != canonical:
        canonical.write_bytes(best.read_bytes())

    # Run upstream eval to fill test_results.json
    metrics = _evaluate_after_train(cfg, trainer) if canonical.exists() else {}

    # Skeleton observability tree — main-venv ONNX path fills it later.
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


def _resolve_best_checkpoint(save_dir: Path, trainer) -> Path | None:
    """Return the path of the canonical best/last checkpoint produced by ppdet."""
    for name in ("best_model.pdparams", "model_final.pdparams"):
        p = save_dir / name
        if p.exists():
            return p
    # Fall back to highest-epoch snapshot
    snaps = sorted(save_dir.glob("*.pdparams"))
    return snaps[-1] if snaps else None


def _evaluate_after_train(cfg, trainer) -> dict:
    """Run trainer.evaluate() and pull numeric metrics out of the result."""
    try:
        trainer.evaluate()
    except Exception as exc:  # eval failures shouldn't kill the train run
        print(f"  warning: evaluate after train raised: {exc}", flush=True)
        return {}
    # ppdet's Trainer keeps stats on _eval_metrics, but the API has changed
    # across versions. Best-effort scrape:
    metrics = {}
    for attr in ("_eval_metrics", "metric_results", "_metrics"):
        v = getattr(trainer, attr, None)
        if isinstance(v, dict) and v:
            metrics.update({str(k): float(v[k]) for k in v if isinstance(v[k], (int, float))})
            break
    return metrics


def _ensure_observability_tree(save_dir: Path) -> None:
    """Empty subdirectories that downstream consumers (releases/, p08 rerun) walk."""
    for sub in ("data_preview", "val_predictions", "test_predictions"):
        (save_dir / sub).mkdir(parents=True, exist_ok=True)
    (save_dir / "val_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)
    (save_dir / "test_predictions" / "error_analysis").mkdir(parents=True, exist_ok=True)


_TASK_DISPATCH = {
    # Detection — canonical path for v1
    "paddle-picodet-": _train_detection,
    "paddle-ppyoloe-": _train_detection,
    # Other task families — implement when needed; raise a clear error for now.
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
        _apply_overrides(our, args.override)

    print(f"[paddle] config: {args.config}", flush=True)
    print(f"[paddle] arch:   {our['model']['arch']}", flush=True)
    print(f"[paddle] task:   detection (v1)", flush=True)

    _ensure_coco_annotations(our)
    summary = _dispatch(our)

    print(f"\n[paddle] training complete. summary:", flush=True)
    print(f"  best_checkpoint: {summary.get('best_checkpoint')}", flush=True)
    print(f"  best_metric:     {summary.get('best_metric')}", flush=True)
    print(f"  output_dir:      {summary.get('output_dir')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
