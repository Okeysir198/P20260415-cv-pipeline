"""Temperature scaling for RT-DETRv2 confidence calibration.

Fits a single scalar T by minimizing NLL on val-set predictions matched to GT.
DETR-family models rank correctly but compress sigmoid scores into ~[0, 0.2].
Applying score = sigmoid(logit / T) with T < 1 expands the range without
changing mAP (sigmoid(x/T) is strictly monotonic in x).

Workflow:
1. Load trained RT-DETRv2 checkpoint.
2. Run inference on val split at very low threshold (0.001).
3. Match each prediction to GT by IoU > 0.5 + correct class → label {TP=1, FP=0}.
4. Fit T via LBFGS on BCE(sigmoid(logit/T), label).
5. Report T, plus sample inference with T applied (samples_rtdetr_t_scaled/).

Saves: <ckpt_dir>/temperature.json = {"T": float, "n_tp": int, "n_fp": int, ...}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/ct-admin/Documents/Langgraph/TEST/ai")
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from utils.config import load_config

RUN_DIR = REPO / "features/safety-fire_detection/runs/2026-04-21_154828_06_training"
CKPT = RUN_DIR / "checkpoint-2930"
DATA_CFG = REPO / "features/safety-fire_detection/configs/05_data.yaml"
SAMPLES = REPO / "features/safety-fire_detection/samples"
OUT = SAMPLES.parent / "samples_rtdetr_t_scaled"
OUT.mkdir(exist_ok=True)
T_OUT = CKPT / "temperature.json"

NAMES = {0: "fire", 1: "smoke"}
COLORS = {0: (255, 80, 60), 1: (120, 160, 255)}
IOU_MATCH = 0.5
TOP_K_PER_IMAGE = 50         # top-scoring queries per image (RT-DETR post-process norm)
DISPLAY_CONF = 0.15
MAX_VAL_IMAGES = 500         # plenty for fitting a single scalar


def _load_model(device: str):
    model = AutoModelForObjectDetection.from_pretrained(str(CKPT), ignore_mismatched_sizes=True)
    sd = torch.load(CKPT / "pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = {k.removeprefix("hf_model."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"load mismatch: {len(missing)} miss / {len(unexpected)} unex"
    processor = AutoImageProcessor.from_pretrained(str(CKPT))
    return model.to(device).eval(), processor


def _load_val_items() -> list[tuple[Path, Path]]:
    data_cfg = load_config(str(DATA_CFG))
    ds_root = (DATA_CFG.parent / data_cfg["path"]).resolve()
    img_dir = ds_root / data_cfg["val"]
    lbl_dir = ds_root / data_cfg["val"].replace("images", "labels")
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        items.append((img, lbl))
    return items[:MAX_VAL_IMAGES]


def _parse_yolo_labels(lbl: Path, W: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (boxes_xyxy_pixels, class_ids) tensors. Empty if no labels."""
    if not lbl.exists():
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    lines = [ln.strip().split() for ln in lbl.read_text().splitlines() if ln.strip()]
    if not lines:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    arr = np.array([[float(v) for v in ln[:5]] for ln in lines])  # cls, cx, cy, w, h (norm)
    cls = arr[:, 0].astype(np.int64)
    cx, cy, w, h = arr[:, 1] * W, arr[:, 2] * H, arr[:, 3] * W, arr[:, 4] * H
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return torch.from_numpy(boxes).float(), torch.from_numpy(cls)


def collect_logits_labels(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """For each val image, run model and match predictions to GT.

    Returns:
        logits: (N,) pre-sigmoid class scores from filtered matcher queries.
                We take max-class-logit per query above the score threshold.
        labels: (N,) binary {1=TP matched to correct-class GT, 0=FP/no-match}.
    """
    model, processor = _load_model(device)
    items = _load_val_items()
    all_logits, all_labels = [], []

    for img_path, lbl_path in tqdm(items, desc="collect val preds"):
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_boxes, gt_cls = _parse_yolo_labels(lbl_path, W, H)

        inp = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inp)
        # logits: (1, num_queries, num_classes) pre-sigmoid
        # boxes:  (1, num_queries, 4) in normalized cxcywh
        logits = out.logits[0].cpu()                       # (Q, C)
        boxes_norm = out.pred_boxes[0].cpu()               # (Q, 4) cxcywh norm
        # convert boxes to xyxy pixels
        cx, cy, w, h = boxes_norm.unbind(-1)
        boxes_xyxy = torch.stack([
            (cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H,
        ], dim=-1)
        scores = logits.sigmoid()                          # (Q, C)
        max_scores, max_cls = scores.max(-1)               # (Q,)
        # Keep only top-K queries per image — matches what the post-processor
        # would output at inference. Fitting T on all 300 queries (98% no-obj)
        # drowns the signal and drives T → 1.
        k = min(TOP_K_PER_IMAGE, len(max_scores))
        topk_vals, topk_idx = max_scores.topk(k)
        q_boxes = boxes_xyxy[topk_idx]
        q_cls = max_cls[topk_idx]
        q_logits = logits[topk_idx].gather(-1, max_cls[topk_idx].unsqueeze(-1)).squeeze(-1)

        if gt_boxes.numel() == 0:
            # All queries are FP
            all_logits.append(q_logits)
            all_labels.append(torch.zeros(len(q_logits)))
            continue

        # IoU between each kept query and every GT box
        ious = box_iou(q_boxes, gt_boxes)                  # (Q', G)
        best_iou, best_gt = ious.max(-1)                   # (Q',)
        matched_cls = gt_cls[best_gt]                      # (Q',)
        is_tp = (best_iou >= IOU_MATCH) & (matched_cls == q_cls)
        all_logits.append(q_logits)
        all_labels.append(is_tp.float())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return logits, labels


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit T by minimizing BCE(sigmoid(logits/T), labels) via LBFGS."""
    logits = logits.double()
    labels = labels.double()
    logT = torch.zeros(1, dtype=torch.double, requires_grad=True)  # T = exp(logT), init T=1
    opt = torch.optim.LBFGS([logT], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = logT.exp()
        loss = F.binary_cross_entropy_with_logits(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def run_sample_inference_with_t(T: float, device: str) -> None:
    """Apply learned T to sample images: score = sigmoid(logit / T)."""
    model, processor = _load_model(device)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    samples = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    summary = []
    for path in samples:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        inp = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inp)
        logits = out.logits[0].cpu()
        boxes_norm = out.pred_boxes[0].cpu()
        scores_t = (logits / T).sigmoid()                   # T-scaled
        max_scores, max_cls = scores_t.max(-1)
        keep = max_scores > DISPLAY_CONF
        sel_scores = max_scores[keep]
        sel_cls = max_cls[keep]
        cx, cy, w, h = boxes_norm[keep].unbind(-1)
        sel_boxes = torch.stack([
            (cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H,
        ], dim=-1)

        draw = ImageDraw.Draw(img)
        dets = []
        for s, c, b in zip(sel_scores, sel_cls, sel_boxes):
            s, c = float(s), int(c)
            x1, y1, x2, y2 = [float(v) for v in b]
            name = NAMES.get(c, str(c))
            color = COLORS.get(c, (0, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            tag = f"{name} {s:.2f}"
            tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
            draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw + 8, y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - th - 4)), tag, fill=(255, 255, 255), font=font)
            dets.append({"class": name, "score": round(s, 3), "box": [round(v, 1) for v in (x1, y1, x2, y2)]})

        img.save(OUT / path.name, quality=92)
        summary.append({"image": path.name, "detections": dets})
        tag = ", ".join(f"{d['class']}:{d['score']}" for d in dets) or "-"
        print(f"{path.name:30s}  {len(dets):2d} det  [{tag}]")

    (OUT / "summary.json").write_text(json.dumps({"T": T, "results": summary}, indent=2))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RT-DETRv2 temperature scaling ===")
    print(f"Checkpoint: {CKPT}")

    print("\n[1/3] Collecting val logits + GT matches …")
    logits, labels = collect_logits_labels(device)
    n_tp = int(labels.sum())
    n_fp = int(len(labels) - n_tp)
    print(f"  collected {len(logits)} queries: {n_tp} TP, {n_fp} FP")

    # Quick sanity: mean pos/neg logit
    pos_logits = logits[labels == 1]
    neg_logits = logits[labels == 0]
    print(f"  pre-T:  pos logit mean {pos_logits.mean():.3f}  (sigmoid≈{pos_logits.mean().sigmoid():.3f})")
    print(f"  pre-T:  neg logit mean {neg_logits.mean():.3f}  (sigmoid≈{neg_logits.mean().sigmoid():.3f})")

    print("\n[2/3] Fitting temperature …")
    T = fit_temperature(logits, labels)
    print(f"  ➜ T* = {T:.4f}")

    # Post-T summary
    print(f"  post-T: pos logit mean {(pos_logits / T).mean():.3f}  (sigmoid≈{(pos_logits / T).mean().sigmoid():.3f})")
    print(f"  post-T: neg logit mean {(neg_logits / T).mean():.3f}  (sigmoid≈{(neg_logits / T).mean().sigmoid():.3f})")

    T_OUT.write_text(json.dumps({
        "T": T, "n_tp": n_tp, "n_fp": n_fp,
        "pos_logit_mean_pre": float(pos_logits.mean()),
        "neg_logit_mean_pre": float(neg_logits.mean()),
        "iou_threshold": IOU_MATCH,
        "val_images_used": MAX_VAL_IMAGES,
    }, indent=2))
    print(f"  saved: {T_OUT}")

    print(f"\n[3/3] Sample inference with T={T:.3f} @ conf≥{DISPLAY_CONF} …")
    run_sample_inference_with_t(T, device)
    print(f"\nWrote annotated samples to {OUT}")


if __name__ == "__main__":
    main()
