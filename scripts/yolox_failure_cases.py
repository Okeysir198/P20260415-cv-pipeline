"""Render GT + predictions for the worst-ranked val images of YOLOX-M.

Uses error_analysis.json's hardest_images (ranked by error count at conf=0.001)
and draws:
  - Green boxes: ground truth
  - Red boxes: predictions at the optimal F1 threshold

Usage (must run from the yolox venv — needs official yolox package + repo):
    ./.venv-yolox-official/bin/python scripts/yolox_failure_cases.py \\
        --ckpt features/safety-fire_detection/runs/2026-04-19_131107_06_training/best.pth \\
        --data-config features/safety-fire_detection/configs/05_data.yaml \\
        --analysis features/safety-fire_detection/eval/yolox_official_ep51/error_analysis.json \\
        --out features/safety-fire_detection/eval/yolox_official_ep51/failure_cases.png \\
        --top 6 --conf 0.42
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from torchvision.ops import batched_nms

from core.p05_data.detection_dataset import YOLOXDataset
from core.p05_data.transforms import build_transforms
from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
from core.p06_models.yolox import build_yolox
from utils.config import load_config

COCO_FIRE = ["fire", "smoke"]


def _denorm(img_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalize, back to [0, 255] BGR uint8."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # CHW RGB → HWC RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _postprocess(raw: torch.Tensor, conf: float, nms_iou: float = 0.45):
    """(1, N, 5+C) → dict with xyxy boxes, scores, labels (already sigmoid'd)."""
    pred = raw[0]
    obj = pred[:, 4]
    cls_scores, cls_ids = pred[:, 5:].max(dim=1)
    scores = obj * cls_scores
    mask = scores >= conf
    pred, scores, cls_ids = pred[mask], scores[mask], cls_ids[mask]
    if pred.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    keep = batched_nms(boxes, scores, cls_ids, nms_iou)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(), cls_ids[keep].cpu().numpy()


def _draw(img: np.ndarray, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, img_idx: int):
    img = img.copy()
    h, w = img.shape[:2]
    # GT boxes — green, thin
    for box, lbl in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(img, f"GT:{COCO_FIRE[int(lbl)]}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    # Predictions — red, thin
    for box, score, lbl in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(img, f"{COCO_FIRE[int(lbl)]}:{score:.2f}", (x1, min(h - 4, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    # Header banner
    banner = np.full((32, w, 3), 30, dtype=np.uint8)
    text = f"img #{img_idx} | {len(gt_boxes)} GT | {len(pred_boxes)} pred @conf>={CONF_THRESH}"
    cv2.putText(banner, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return np.vstack([banner, img])


CONF_THRESH = 0.42  # optimal F1 threshold from error_analysis


def main():
    global CONF_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--conf", type=float, default=CONF_THRESH)
    args = ap.parse_args()
    CONF_THRESH = args.conf

    data_cfg = load_config(args.data_config)
    with open(args.analysis) as f:
        ea = json.load(f)
    hardest = ea["summary"]["hardest_images"][: args.top]
    hard_idx = [h["image_idx"] for h in hardest]

    # Build val dataset with same transforms training used
    transforms = build_transforms(
        config={}, is_train=False,
        input_size=tuple(data_cfg["input_size"]),
        mean=data_cfg.get("mean", IMAGENET_MEAN),
        std=data_cfg.get("std", IMAGENET_STD),
    )
    dataset = YOLOXDataset(
        data_config=data_cfg, split="val",
        transforms=transforms, base_dir=str(Path(args.data_config).parent),
    )

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = ckpt["model_state_dict"]
    cfg = ckpt["config"]
    cfg["model"].pop("pretrained", None)
    model = build_yolox(cfg).cuda().eval()
    model.load_state_dict(state, strict=False)

    input_h, input_w = data_cfg["input_size"]
    panels = []
    for idx in hard_idx:
        image_t, target_t, _ = dataset[idx]
        img_bgr = _denorm(image_t)

        # GT (normalized cxcywh → pixel xyxy at input size)
        gt = target_t.numpy()
        if gt.shape[0] > 0:
            cx, cy, w, h = gt[:, 1]*input_w, gt[:, 2]*input_h, gt[:, 3]*input_w, gt[:, 4]*input_h
            gt_boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
            gt_labels = gt[:, 0].astype(int)
        else:
            gt_boxes, gt_labels = np.zeros((0, 4)), np.zeros(0, dtype=int)

        # Predictions
        with torch.no_grad():
            raw = model(image_t.unsqueeze(0).cuda())
        pb, ps, pl = _postprocess(raw, args.conf)

        panels.append(_draw(img_bgr, gt_boxes, gt_labels, pb, ps, pl, idx))

    # Grid layout: 2 rows × (top/2) cols, equal-sized
    n = len(panels)
    cols = 3
    rows = (n + cols - 1) // cols
    h_max = max(p.shape[0] for p in panels)
    w_max = max(p.shape[1] for p in panels)
    padded = [
        np.pad(p, ((0, h_max - p.shape[0]), (0, w_max - p.shape[1]), (0, 0)),
               constant_values=20)
        for p in panels
    ]
    # Fill grid
    grid_rows = []
    for r in range(rows):
        row = padded[r*cols:(r+1)*cols]
        if len(row) < cols:
            row += [np.full_like(padded[0], 20)] * (cols - len(row))
        grid_rows.append(np.hstack(row))
    grid = np.vstack(grid_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Saved: {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
