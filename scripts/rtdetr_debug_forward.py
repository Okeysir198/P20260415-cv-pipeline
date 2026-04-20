"""Debug what RT-DETRv2 actually receives and predicts on our training data.

Loads a small batch, forwards through the HF model, and prints:
- pixel_values range + shape
- label structure (class IDs, box values)
- outputs.logits stats (are they diverging?)
- outputs.pred_boxes range
- outputs.loss + loss_dict (how much of which component?)
- matched queries per image (via the auxiliary decoder outputs if available)
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

    batch = next(iter(trainer.train_loader))
    images = batch["images"].to(trainer.device)
    targets_raw = [t.to(trainer.device) for t in batch["targets"]]

    print(f"\n=== INPUT TENSORS ===")
    print(f"images shape: {tuple(images.shape)}  dtype={images.dtype}")
    print(f"images min/max/mean: {images.min().item():.4f} / {images.max().item():.4f} / {images.mean().item():.4f}")

    print(f"\n=== RAW TARGETS (normalized cxcywh from dataloader) ===")
    for i, t in enumerate(targets_raw[:3]):
        if t.numel() == 0:
            print(f"  img {i}: EMPTY")
            continue
        print(f"  img {i}: {t.shape[0]} boxes, cls_ids={t[:, 0].cpu().tolist()}")
        print(f"          cx range {t[:, 1].min().item():.3f}-{t[:, 1].max().item():.3f}, cy {t[:, 2].min().item():.3f}-{t[:, 2].max().item():.3f}")
        print(f"          w  range {t[:, 3].min().item():.3f}-{t[:, 3].max().item():.3f}, h  {t[:, 4].min().item():.3f}-{t[:, 4].max().item():.3f}")

    # Run the trainer's scale step
    input_h, input_w = trainer._model_cfg["input_size"]
    targets_pixel = trainer._scale_targets_to_pixels(targets_raw, input_h, input_w)
    print(f"\n=== AFTER _scale_targets_to_pixels (H={input_h} W={input_w}) ===")
    for i, t in enumerate(targets_pixel[:3]):
        if t.numel() == 0:
            continue
        print(f"  img {i}: cx range {t[:, 1].min().item():.1f}-{t[:, 1].max().item():.1f}, w  {t[:, 3].min().item():.1f}-{t[:, 3].max().item():.1f}")

    # Run _format_targets to see the normalized labels HF actually gets
    base_model = trainer._base_model
    hf_labels = base_model._format_targets(targets_pixel, input_h, input_w)
    print(f"\n=== AFTER _format_targets (HF labels — normalized cxcywh) ===")
    for i, lbl in enumerate(hf_labels[:3]):
        if lbl["boxes"].numel() == 0:
            print(f"  img {i}: EMPTY")
            continue
        print(f"  img {i}: class_labels={lbl['class_labels'].cpu().tolist()}")
        b = lbl["boxes"]
        print(f"          box cx {b[:, 0].min().item():.3f}-{b[:, 0].max().item():.3f}, w {b[:, 2].min().item():.3f}-{b[:, 2].max().item():.3f}")

    # Forward
    print(f"\n=== HF FORWARD OUTPUTS ===")
    with torch.no_grad():
        outputs = base_model.hf_model(pixel_values=images, labels=hf_labels)
    print(f"  logits:     shape={tuple(outputs.logits.shape)}, range [{outputs.logits.min().item():.3f}, {outputs.logits.max().item():.3f}]")
    sig = outputs.logits.sigmoid()
    print(f"  sigmoid(logits): max per class 0={sig[..., 0].max().item():.4f}, class 1={sig[..., 1].max().item():.4f}")
    print(f"  pred_boxes: shape={tuple(outputs.pred_boxes.shape)}, range [{outputs.pred_boxes.min().item():.4f}, {outputs.pred_boxes.max().item():.4f}]")
    print(f"  loss:       {outputs.loss.item():.4f}")
    print(f"  loss_dict:")
    for k, v in outputs.loss_dict.items():
        print(f"    {k}: {v.item():.4f}")

    # Check HF config
    cfg = base_model.hf_model.config
    print(f"\n=== MODEL CONFIG RELEVANT FIELDS ===")
    for k in ("num_labels", "num_queries", "num_denoising", "eos_coefficient",
             "focal_alpha", "matcher_class_cost", "matcher_bbox_cost", "matcher_giou_cost",
             "weight_loss_vfl", "weight_loss_bbox", "weight_loss_giou",
             "freeze_backbone_batch_norms"):
        print(f"  {k}: {getattr(cfg, k, '(not set)')}")


if __name__ == "__main__":
    main()
