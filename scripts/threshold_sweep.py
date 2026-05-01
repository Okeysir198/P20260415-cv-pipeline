"""Per-class confidence threshold sweep on val set.

Loads a trained checkpoint, runs inference once at a low threshold (0.01),
then post-hoc computes per-class TP/FP/FN/precision/recall/F1 at a sweep
of per-class thresholds. Identifies the best operating point per class.

Usage:
    uv run scripts/threshold_sweep.py --run features/.../checkpoint-N --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.device import auto_select_gpu  # noqa: E402

auto_select_gpu()

from core.p05_data.detection_dataset import build_dataloader  # noqa: E402
from core.p06_models import build_model  # noqa: E402
from utils.checkpoint import strip_hf_prefix  # noqa: E402
from utils.config import load_config  # noqa: E402

IOU_THRESH = 0.5


def _box_iou(a, b):
    """Vectorized IoU between (N,4) and (M,4) xyxy arrays. Returns (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def _eval_at_thresholds(all_preds, all_gts, n_classes, thresholds):
    """For each threshold, compute per-class TP/FP/FN. thresholds: list of (cls, thresh) → eval."""
    results = []
    for thresh_per_class in thresholds:  # list of length n_classes
        cls_tp = np.zeros(n_classes, dtype=int)
        cls_fp = np.zeros(n_classes, dtype=int)
        cls_fn = np.zeros(n_classes, dtype=int)

        for preds, gts in zip(all_preds, all_gts):
            # Filter preds by per-class threshold
            keep = np.zeros(len(preds["labels"]), dtype=bool)
            for c in range(n_classes):
                keep |= (preds["labels"] == c) & (preds["scores"] >= thresh_per_class[c])
            pb = preds["boxes"][keep]
            pl = preds["labels"][keep]
            ps = preds["scores"][keep]

            gb = np.asarray(gts["boxes"], dtype=np.float64).reshape(-1, 4)
            gl = np.asarray(gts["labels"], dtype=np.int64)

            # Greedy match: highest-score pred first per class
            order = np.argsort(-ps)
            gt_matched = np.zeros(len(gb), dtype=bool)

            for c in range(n_classes):
                # GT count for this class in this image
                gt_cls_mask = gl == c
                cls_fn[c] += int(gt_cls_mask.sum())  # tentative; subtract matches

            for idx in order:
                c = int(pl[idx])
                pred_box = pb[idx:idx+1]
                # Find best matching unmatched GT of same class
                gt_cls_idx = np.where((gl == c) & ~gt_matched)[0]
                if len(gt_cls_idx) == 0:
                    cls_fp[c] += 1
                    continue
                ious = _box_iou(pred_box, gb[gt_cls_idx]).ravel()
                best = int(ious.argmax())
                if ious[best] >= IOU_THRESH:
                    gt_matched[gt_cls_idx[best]] = True
                    cls_tp[c] += 1
                    cls_fn[c] -= 1
                else:
                    cls_fp[c] += 1

        results.append({"thresholds": thresh_per_class, "tp": cls_tp.copy(),
                        "fp": cls_fp.copy(), "fn": cls_fn.copy()})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to checkpoint dir (checkpoint-N)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--low-thresh", type=float, default=0.01,
                    help="Initial inference threshold (catch everything for post-hoc filter)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg_root = run_dir if (run_dir / "06_training.yaml").exists() else run_dir.parent
    train_cfg = load_config(cfg_root / "06_training.yaml")
    data_cfg = load_config(cfg_root / "05_data.yaml")
    cfg_dir = ROOT / "features" / "safety-fire_detection" / "configs"
    data_cfg["_config_dir"] = cfg_dir
    train_cfg["data"]["batch_size"] = train_cfg["data"].get("eval_batch_size", 4)
    # Disable the random subset so we eval on the full split
    train_cfg.setdefault("data", {}).setdefault("subset", {})[args.split] = None

    print(f"loading model from {run_dir}/pytorch_model.bin")
    model = build_model(train_cfg)
    state = torch.load(run_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)
    state = strip_hf_prefix(state)
    inner = model.hf_model if hasattr(model, "hf_model") else model
    missing, unexpected = inner.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    model = model.cuda().eval()

    print(f"building {args.split} loader (full)")
    loader = build_dataloader(
        data_cfg, split=args.split, training_config=train_cfg, base_dir=cfg_dir,
    )
    print(f"  {len(loader)} batches")

    n_classes = data_cfg["num_classes"]
    class_names = {int(k): v for k, v in data_cfg["names"].items()}

    all_preds = []
    all_gts = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["images"].cuda()
            targets = batch["targets"]
            out = model(pixel_values=images) if hasattr(model, "hf_model") else model(images)
            B, _, H, W = images.shape
            target_sizes = torch.tensor([[H, W]] * B, device=images.device)
            decoded = model.postprocess(out, args.low_thresh, target_sizes)

            for b in range(B):
                d = decoded[b]
                def _np(x, fb):
                    if x is None:
                        return fb
                    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
                all_preds.append({
                    "boxes": _np(d.get("boxes"), np.zeros((0, 4))),
                    "labels": _np(d.get("labels"), np.zeros((0,), dtype=np.int64)),
                    "scores": _np(d.get("scores"), np.zeros((0,))),
                })
                t = targets[b]
                if t.numel() == 0:
                    all_gts.append({"boxes": [], "labels": []})
                    continue
                cls = t[:, 0].long().cpu().numpy()
                cx, cy, w, h = (t[:, k].cpu().numpy() for k in range(1, 5))
                x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                all_gts.append({"boxes": np.stack([x1, y1, x2, y2], axis=1),
                                "labels": cls})
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(loader)} batches")

    # Sweep per-class thresholds. Apply same threshold to both classes for each value.
    sweep_vals = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    print(f"\n{'='*80}")
    print(f"GLOBAL SWEEP (same threshold all classes)")
    print(f"{'='*80}")
    print(f"{'thr':>6}  ", end="")
    for c in range(n_classes):
        nm = class_names[c]
        print(f"{nm}_TP {nm}_FP {nm}_FN {nm}_P {nm}_R {nm}_F1   ", end="")
    print()
    for v in sweep_vals:
        thr = [v] * n_classes
        r = _eval_at_thresholds(all_preds, all_gts, n_classes, [thr])[0]
        print(f"{v:>6.3f}  ", end="")
        for c in range(n_classes):
            tp, fp, fn = r["tp"][c], r["fp"][c], r["fn"][c]
            p = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1 = 2 * p * rc / max(p + rc, 1e-9)
            print(f"{tp:>4d} {fp:>4d} {fn:>4d} {p:>5.3f} {rc:>5.3f} {f1:>5.3f}   ", end="")
        print()

    # Per-class sweep: vary smoke threshold while keeping fire at 0.10
    print(f"\n{'='*80}")
    print(f"SMOKE-ONLY SWEEP (fire fixed at 0.10)")
    print(f"{'='*80}")
    print(f"{'smoke_thr':>10}  smoke_TP  smoke_FP  smoke_FN  smoke_P  smoke_R  smoke_F1")
    for v in sweep_vals:
        thr = [0.10, v]  # fire=0.10, smoke=v (assuming fire is class 0)
        r = _eval_at_thresholds(all_preds, all_gts, n_classes, [thr])[0]
        c = 1
        tp, fp, fn = r["tp"][c], r["fp"][c], r["fn"][c]
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-9)
        print(f"{v:>10.3f}  {tp:>8d}  {fp:>8d}  {fn:>8d}  {p:>7.3f}  {rc:>7.3f}  {f1:>8.3f}")


if __name__ == "__main__":
    main()
