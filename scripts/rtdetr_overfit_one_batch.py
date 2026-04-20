"""Overfit a SINGLE RT-DETR batch — the canonical sanity check.

If the model can't drive loss to near-zero on 8 images after 500 steps with a
reasonable LR, there's a fundamental pipeline issue. If it can, the plateau on
585 images must be a data-scale / optimization dynamics issue, not a bug.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.device import auto_select_gpu  # noqa: E402
auto_select_gpu()

import torch  # noqa: E402

from core.p06_training.trainer import DetectionTrainer  # noqa: E402


def main():
    config_path = ROOT / "features" / "safety-fire_detection" / "configs" / "06_training_rtdetr.yaml"
    overrides = {
        "data": {"subset": {"train": 0.05, "val": 0.05}, "batch_size": 8},
        "augmentation": {
            "mosaic": False, "mixup": False, "copypaste": False,
            "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
            "fliplr": 0, "flipud": 0, "degrees": 0, "translate": 0, "shear": 0,
            "contrast": 0, "scale": [1, 1],
        },
        "training": {
            "val_viz": {"enabled": False}, "data_viz": {"enabled": False},
            "aug_viz": {"enabled": False}, "val_full_interval": 0,
        },
        "logging": {"wandb_project": None},
    }
    trainer = DetectionTrainer(str(config_path), overrides=overrides)
    trainer.train_loader, trainer.val_loader = trainer._build_dataloaders()
    trainer.model = trainer._build_model()
    trainer.model.train()

    # Grab ONE batch, keep on GPU
    batch = next(iter(trainer.train_loader))
    images = batch["images"].to(trainer.device)
    targets_raw = [t.to(trainer.device) for t in batch["targets"]]
    input_h, input_w = trainer._model_cfg["input_size"]
    targets_pixel = trainer._scale_targets_to_pixels(targets_raw, input_h, input_w)

    base_model = trainer._base_model
    hf_labels = base_model._format_targets(targets_pixel, input_h, input_w)
    gt_counts = [(l["class_labels"].shape[0]) for l in hf_labels]
    print(f"Batch has {sum(gt_counts)} total GT boxes across {len(gt_counts)} images: {gt_counts}")

    # Pure AdamW overfit — canonical DETR-style
    opt = torch.optim.AdamW(
        [p for p in base_model.hf_model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.0,
    )

    print(f"\n=== OVERFIT SINGLE BATCH — 300 steps, lr=1e-4, no warmup/aug ===")
    print(f"{'step':>5s}  {'loss':>10s}  {'vfl':>8s}  {'bbox':>8s}  {'giou':>8s}  {'lr':>10s}")
    for step in range(300):
        opt.zero_grad()
        outputs = base_model.hf_model(pixel_values=images, labels=hf_labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.hf_model.parameters(), 0.1)
        opt.step()

        if step < 10 or step % 20 == 0:
            ld = outputs.loss_dict
            print(
                f"{step:>5d}  {outputs.loss.item():>10.4f}  "
                f"{ld.get('loss_vfl', torch.tensor(0.)).item():>8.4f}  "
                f"{ld.get('loss_bbox', torch.tensor(0.)).item():>8.4f}  "
                f"{ld.get('loss_giou', torch.tensor(0.)).item():>8.4f}  "
                f"{opt.param_groups[0]['lr']:>10.6f}"
            )

    print("\n=== SWITCH TO EVAL MODE AND PREDICT ON SAME BATCH ===")
    base_model.hf_model.eval()
    with torch.no_grad():
        outputs = base_model.hf_model(pixel_values=images)
    logits = outputs.logits
    sig = logits.sigmoid()
    scores, labels = sig.max(dim=-1)
    for b in range(min(3, images.shape[0])):
        top_scores, top_indices = scores[b].topk(min(10, scores.shape[1]))
        top_labels = labels[b][top_indices]
        print(f"  img {b}: GT count={gt_counts[b]}, top-5 scores={top_scores[:5].tolist()}, labels={top_labels[:5].tolist()}")


if __name__ == "__main__":
    main()
