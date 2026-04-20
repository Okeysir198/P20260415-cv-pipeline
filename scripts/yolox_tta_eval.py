"""TTA (Test-Time Augmentation) evaluation for the official YOLOX adapter.

For each val image runs inference at N scales (default: 512/640/768), each
both original and horizontally flipped. All predictions are scaled back to
the 640x640 reference frame, merged, then class-wise NMS is applied.

Compare against baseline (no-TTA) mAP printed by p08/evaluate.py.

Usage (official venv, needs YOLOXDataset + official yolox pkg):
    ./.venv-yolox-official/bin/python scripts/yolox_tta_eval.py \\
        --ckpt features/safety-fire_detection/runs/2026-04-19_131107_06_training/best.pth \\
        --data-config features/safety-fire_detection/configs/05_data.yaml \\
        --scales 512,640,768 \\
        --flip \\
        --out-dir features/safety-fire_detection/eval/yolox_official_ep51_tta
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from core.p06_models.yolox import build_yolox
from core.p08_evaluation.sv_metrics import compute_map
from utils.config import load_config
from utils.progress import ProgressBar


def _postprocess_yolox(
    raw: torch.Tensor, conf: float, nms_iou: float
) -> List[dict]:
    """(B, N, 5+C) → list of dicts with xyxy boxes (at input scale), scores, labels."""
    out = []
    for b in range(raw.shape[0]):
        pred = raw[b]
        obj = pred[:, 4]
        cls_scores, cls_ids = pred[:, 5:].max(dim=1)
        scores = obj * cls_scores
        mask = scores >= conf
        if not mask.any():
            out.append({"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)})
            continue
        pred, scores, cls_ids = pred[mask], scores[mask], cls_ids[mask]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        keep = batched_nms(boxes, scores, cls_ids, nms_iou)
        out.append({
            "boxes": boxes[keep].cpu().numpy(),
            "scores": scores[keep].cpu().numpy(),
            "labels": cls_ids[keep].cpu().numpy().astype(np.int64),
        })
    return out


def _resize_for_scale(images: torch.Tensor, target_size: int) -> torch.Tensor:
    """Stretch-resize ``(B, 3, H, W)`` batch to ``(B, 3, target_size, target_size)``."""
    return F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)


def _hflip_boxes(boxes: np.ndarray, img_w: float) -> np.ndarray:
    """Unflip xyxy boxes after a horizontal flip."""
    out = boxes.copy()
    out[:, 0] = img_w - boxes[:, 2]
    out[:, 2] = img_w - boxes[:, 0]
    return out


def _scale_boxes(boxes: np.ndarray, src_size: float, dst_size: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    return boxes * (dst_size / src_size)


def _merge_tta(
    all_predictions: List[List[dict]], ref_size: float, nms_iou: float
) -> List[dict]:
    """Concat predictions across TTA passes (already in ref_size space) then NMS."""
    n_images = len(all_predictions[0])
    merged = []
    for i in range(n_images):
        boxes = np.concatenate([p[i]["boxes"] for p in all_predictions if len(p[i]["boxes"])], axis=0) \
            if any(len(p[i]["boxes"]) for p in all_predictions) else np.zeros((0, 4))
        scores = np.concatenate([p[i]["scores"] for p in all_predictions if len(p[i]["scores"])], axis=0) \
            if any(len(p[i]["scores"]) for p in all_predictions) else np.zeros(0)
        labels = np.concatenate([p[i]["labels"] for p in all_predictions if len(p[i]["labels"])], axis=0) \
            if any(len(p[i]["labels"]) for p in all_predictions) else np.zeros(0, dtype=np.int64)
        if boxes.shape[0] == 0:
            merged.append({"boxes": boxes, "scores": scores, "labels": labels})
            continue
        bt = torch.from_numpy(boxes).float()
        st = torch.from_numpy(scores).float()
        lt = torch.from_numpy(labels).long()
        keep = batched_nms(bt, st, lt, nms_iou)
        merged.append({
            "boxes": bt[keep].numpy(), "scores": st[keep].numpy(), "labels": lt[keep].numpy().astype(np.int64),
        })
    return merged


def _gt_list(dataset, indices: List[int], ref_size: int) -> List[dict]:
    """Pull ground-truths at reference (ref_size) scale."""
    out = []
    for i in indices:
        _, t, _ = dataset[i]
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        if arr.shape[0] == 0:
            out.append({"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=np.int64)})
            continue
        cx = arr[:, 1] * ref_size
        cy = arr[:, 2] * ref_size
        w = arr[:, 3] * ref_size
        h = arr[:, 4] * ref_size
        out.append({
            "boxes": np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1),
            "labels": arr[:, 0].astype(np.int64),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--scales", default="512,640,768")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ref-size", type=int, default=640,
                    help="Reference size for merging (should match training input_size).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-ema", action="store_true")
    args = ap.parse_args()

    scales = [int(s) for s in args.scales.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = load_config(args.data_config)
    input_size = tuple(data_cfg["input_size"])
    mean = torch.tensor(data_cfg.get("mean", IMAGENET_MEAN)).view(1, 3, 1, 1).to(device)
    std = torch.tensor(data_cfg.get("std", IMAGENET_STD)).view(1, 3, 1, 1).to(device)

    # Dataset — val with training-parity transforms (already normalized)
    transforms = build_transforms(
        config={}, is_train=False, input_size=input_size,
        mean=data_cfg.get("mean", IMAGENET_MEAN), std=data_cfg.get("std", IMAGENET_STD),
    )
    dataset = YOLOXDataset(
        data_config=data_cfg, split=args.split,
        transforms=transforms, base_dir=str(Path(args.data_config).parent),
    )

    # Model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["model"].pop("pretrained", None)
    model = build_yolox(cfg).to(device).eval()
    state = ckpt["ema_state_dict"]["ema_model"] if args.use_ema else ckpt["model_state_dict"]
    model.load_state_dict(state, strict=False)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=lambda batch: {
            "images": torch.stack([b[0] for b in batch]),
            "indices": [offset + i for offset, (_, _, _) in enumerate(batch) for i in [0]],
        },
    )

    all_preds_ref: List[List[dict]] = []  # one list per TTA pass; each inner list has per-image dicts in dataset order
    pass_count = len(scales) * (2 if args.flip else 1)

    # We need indices of each image. Simpler: iterate the dataset directly
    # with a DataLoader keyed by index order.
    indices_full = list(range(len(dataset)))

    # Collect predictions per TTA pass
    from core.p05_data.detection_dataset import collate_fn as det_collate
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=det_collate,
    )

    for pass_idx, scale in enumerate(scales):
        for flip in ([False, True] if args.flip else [False]):
            pass_name = f"s{scale}_{'flip' if flip else 'orig'}"
            per_image_preds = []
            with ProgressBar(total=len(loader), desc=f"TTA [{pass_name}]") as pbar:
                with torch.no_grad():
                    for batch in loader:
                        images = batch["images"].to(device)  # already normalized at ref_size
                        # The dataset applied normalize; but we need to resize to `scale`.
                        # Denormalize is not needed — normalize is input-independent of spatial size.
                        # If scale != ref, interpolate directly on normalized tensor.
                        if scale != args.ref_size:
                            images_s = F.interpolate(images, size=(scale, scale),
                                                     mode="bilinear", align_corners=False)
                        else:
                            images_s = images
                        if flip:
                            images_s = torch.flip(images_s, dims=[-1])

                        raw = model(images_s)  # (B, N_scale, 5+C) with boxes in `scale`-space pixels
                        dets = _postprocess_yolox(raw, args.conf, args.nms_iou)

                        for d in dets:
                            # Scale boxes back to ref_size
                            boxes = _scale_boxes(d["boxes"], src_size=scale, dst_size=args.ref_size)
                            # Unflip if needed
                            if flip:
                                boxes = _hflip_boxes(boxes, img_w=args.ref_size)
                            per_image_preds.append({
                                "boxes": boxes, "scores": d["scores"], "labels": d["labels"],
                            })
                        pbar.update()
            all_preds_ref.append(per_image_preds)

    # Merge TTA predictions
    merged = _merge_tta(all_preds_ref, ref_size=args.ref_size, nms_iou=args.nms_iou)

    # Compute mAP against GT at ref_size
    gt = _gt_list(dataset, indices_full, args.ref_size)
    metrics = compute_map(merged, gt, iou_threshold=0.5, num_classes=data_cfg["num_classes"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mAP@0.5": metrics["mAP"],
        "per_class_ap": metrics.get("per_class_ap", {}),
        "precision": metrics.get("precision", {}),
        "recall": metrics.get("recall", {}),
        "n_images": len(merged),
        "scales": scales,
        "flip": args.flip,
        "passes": pass_count,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "use_ema": args.use_ema,
    }
    (out_dir / "tta_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print()
    print("=" * 65)
    print(f"  TTA Evaluation — {pass_count} passes (scales={scales}, flip={args.flip})")
    print("=" * 65)
    print(f"  mAP@0.5: {metrics['mAP']:.4f}")
    for cls_id, ap in metrics.get("per_class_ap", {}).items():
        name = data_cfg["names"].get(int(cls_id), str(cls_id)) if isinstance(data_cfg.get("names"), dict) else str(cls_id)
        print(f"  {name:>12s}  AP={ap:.4f}")
    print(f"  Saved: {out_dir / 'tta_metrics.json'}")


if __name__ == "__main__":
    main()
