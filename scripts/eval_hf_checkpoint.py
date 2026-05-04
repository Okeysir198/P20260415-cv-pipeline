"""One-off: evaluate HF Trainer checkpoint on test split.

Usage:
    uv run python scripts/eval_hf_checkpoint.py \\
        --ckpt features/safety-fire_detection/runs/rtdetr_r50_<ts>/checkpoint-44573 \\
        --data-config features/safety-fire_detection/configs/05_data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchmetrics.detection import MeanAveragePrecision  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: E402

from utils.checkpoint import strip_hf_prefix  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402


def _load_yolo_label(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(p) for p in parts[:5]])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--score-threshold", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"[load] {args.ckpt}")
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = AutoModelForObjectDetection.from_pretrained(args.ckpt)
    state = torch.load(f"{args.ckpt}/pytorch_model.bin", map_location="cpu", weights_only=False)
    state = strip_hf_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()

    data_cfg = load_config(args.data_config)
    base = Path(args.data_config).parent
    # data_cfg has `path` (root) + `<split>: <split>/images` (split-relative).
    # img_dir = path / split-key (e.g. ".../fire_detection/test/images");
    # label_dir = sibling 'labels' dir alongside images.
    root = resolve_path(data_cfg["path"], base)
    img_dir = root / data_cfg[args.split]
    label_dir = img_dir.parent / "labels"

    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"[data] {len(img_paths)} images in {img_dir}")

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    BATCH = 4
    with torch.no_grad():
        for i in range(0, len(img_paths), BATCH):
            batch_paths = img_paths[i:i + BATCH]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            sizes = [(img.size[1], img.size[0]) for img in images]  # (H, W)

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)

            target_sizes = torch.tensor(sizes, device=device)
            preds = processor.post_process_object_detection(
                outputs, threshold=args.score_threshold, target_sizes=target_sizes
            )
            preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]

            targets = []
            for p, (H, W) in zip(batch_paths, sizes, strict=True):
                lbl = _load_yolo_label(label_dir / f"{p.stem}.txt")
                if lbl.shape[0] == 0:
                    targets.append({"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)})
                    continue
                cls = torch.from_numpy(lbl[:, 0]).long()
                cx, cy, w, h = lbl[:, 1] * W, lbl[:, 2] * H, lbl[:, 3] * W, lbl[:, 4] * H
                xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
                targets.append({"boxes": torch.from_numpy(xyxy).float(), "labels": cls})
            metric.update(preds, targets)

            if (i // BATCH) % 25 == 0:
                print(f"  [{i + len(batch_paths)}/{len(img_paths)}]", flush=True)

    raw = metric.compute()
    classes = raw.get("classes")
    id2label = model.config.id2label or {}
    print()
    print(f"=== {Path(args.ckpt).name} on {args.split} ===")
    print(f"  mAP@[0.5:0.95]      : {raw['map'].item():.4f}")
    print(f"  mAP@0.5             : {raw['map_50'].item():.4f}")
    print(f"  mAP@0.75            : {raw['map_75'].item():.4f}")
    if classes is not None:
        for cid, ap in zip(classes.tolist(), raw["map_per_class"].tolist(), strict=True):
            name = id2label.get(int(cid), str(int(cid)))
            print(f"  AP[{name:>6s}]          : {ap:.4f}")


if __name__ == "__main__":
    main()
