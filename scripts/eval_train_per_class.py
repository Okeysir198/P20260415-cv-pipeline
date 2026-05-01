"""Evaluate a trained detection model on the SAME train subset it saw, per-class.

Diagnostic: answers "did the model learn the training set?"
- If train per-class AP is high (~0.85+), generalization is the only issue (overfit).
- If train per-class AP is low, the model has a bias problem on that class.

Usage:
    uv run scripts/eval_train_per_class.py --run features/safety-fire_detection/runs/<ts> --subset 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

from core.p05_data.detection_dataset import build_dataloader  # noqa: E402
from core.p06_models import build_model  # noqa: E402
from core.p08_evaluation.sv_metrics import compute_map  # noqa: E402
from utils.checkpoint import strip_hf_prefix  # noqa: E402
from utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to a run dir containing pytorch_model.bin")
    ap.add_argument("--subset", type=float, default=0.1, help="Train subset fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run)
    # If pointing at a checkpoint-N subdir, configs live in the parent run dir.
    cfg_root = run_dir if (run_dir / "06_training.yaml").exists() else run_dir.parent
    train_cfg = load_config(cfg_root / "06_training.yaml")
    data_cfg = load_config(cfg_root / "05_data.yaml")
    # The data_cfg's `path:` is relative to the original configs dir, not run_dir.
    cfg_dir = ROOT / "features" / "safety-fire_detection" / "configs"
    data_cfg["_config_dir"] = cfg_dir
    train_cfg["data"]["subset"] = {"train": args.subset, "val": None, "test": None}
    train_cfg["data"]["batch_size"] = train_cfg["data"].get("eval_batch_size", 4)
    train_cfg["seed"] = args.seed

    print(f"loading model from {run_dir}/pytorch_model.bin")
    model = build_model(train_cfg)
    state = torch.load(run_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    state = strip_hf_prefix(state)
    # Saved state has wrapper-stripped keys like `model.encoder...`. Our wrapper
    # exposes the inner module as `hf_model`, so load into hf_model.
    inner = model.hf_model if hasattr(model, "hf_model") else model
    missing, unexpected = inner.load_state_dict(state, strict=False)
    print(f"  missing: {len(missing)}  unexpected: {len(unexpected)}")
    model = model.cuda().eval()

    print(f"building train loader (subset={args.subset}, seed={args.seed})")
    loader = build_dataloader(
        data_cfg, split="train", training_config=train_cfg, base_dir=cfg_dir,
    )
    print(f"  {len(loader)} batches × bs={train_cfg['data']['batch_size']} = ~{len(loader) * train_cfg['data']['batch_size']} samples")

    # Run inference, collect predictions + GT in evaluator format
    all_preds = []
    all_gts = []
    class_names = data_cfg["names"]
    n_classes = len(class_names)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["images"].cuda()
            targets = batch["targets"]  # list of tensors (N, 5) [cls,cx,cy,w,h] norm
            # Forward through hf_model wrapper
            if hasattr(model, "hf_model"):
                out = model(pixel_values=images)
            else:
                out = model(images)
            # Postprocess
            B, _, H, W = images.shape
            target_sizes = torch.tensor([[H, W]] * B, device=images.device)
            decoded = model.postprocess(out, args.conf, target_sizes)
            for b in range(B):
                # Pred: dict(boxes=[Nx4 xyxy], labels=[N], scores=[N])
                d = decoded[b]
                def _to_np(x, fallback):
                    if x is None:
                        return fallback
                    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
                all_preds.append({
                    "boxes": _to_np(d.get("boxes"), np.zeros((0, 4))),
                    "labels": _to_np(d.get("labels"), np.zeros((0,), dtype=np.int64)),
                    "scores": _to_np(d.get("scores"), np.zeros((0,))),
                })
                # GT in pixel xyxy
                t = targets[b]
                if t.numel() == 0:
                    all_gts.append({"boxes": [], "labels": []})
                    continue
                cls = t[:, 0].long().cpu().numpy()
                cx, cy, w, h = t[:, 1].cpu().numpy(), t[:, 2].cpu().numpy(), t[:, 3].cpu().numpy(), t[:, 4].cpu().numpy()
                x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                all_gts.append({
                    "boxes": np.stack([x1, y1, x2, y2], axis=1),
                    "labels": cls,
                })
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(loader)} batches done")

    print(f"\nEvaluating {len(all_preds)} predictions vs {len(all_gts)} GT sets")
    metrics = compute_map(all_preds, all_gts, n_classes)
    print(f"\n=== TRAIN per-class mAP (subset={args.subset}) ===")
    def _coerce(v):
        if hasattr(v, "tolist"):
            return v.tolist()
        if hasattr(v, "item"):
            try:
                return float(v)
            except Exception:
                return str(v)
        return v
    print(json.dumps({k: _coerce(v) for k, v in metrics.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
