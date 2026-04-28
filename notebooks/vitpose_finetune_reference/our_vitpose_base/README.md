# our_vitpose_base — in-repo ViTPose run

Same recipe as `../reference_vitpose_base/`, routed through
`core/p06_training/train.py --backend hf` for apples-to-apples
comparison.

## Status — 2026-04-28

**✅ Smoke-verified end-to-end** with the just-landed core wiring:

- `@register_model("hf_keypoint")` in `core/p06_models/hf_model.py` —
  `HFKeypointModel` wrapper around `VitPoseForPoseEstimation` with a
  weighted heatmap-MSE loss exposed through the standard HF Trainer
  hook (`outputs["loss"]`).
- `KeypointTopDownDataset` in `core/p05_data/keypoint_dataset.py` —
  reads YOLO-pose `.txt` labels, expands the dataset to one row per
  labeled person, expands each bbox to the model's H:W aspect with
  `bbox_padding`, hands the crop to `VitPoseImageProcessor`, encodes
  per-joint 2D Gaussian heatmaps + visibility weight.
- Keypoint branch in `core/p06_training/hf_trainer.py::_build_datasets`
  + `keypoint` added to `_SUPPORTED_HF_TASKS` + `label_names =
  ["target_heatmap", "target_weight"]` so HF Trainer reports
  `eval_loss` correctly.
- `dump_coco_keypoints.py` in this folder — pulls
  `rom1x38/COCO_keypoints` (HF parquet mirror) and writes
  `dataset_store/training_ready/coco_keypoints/{train,val}/{images,labels}/`
  in YOLO-pose format.

Verified smoke result on 28 train images / 9 val images
(`--max-train 60 --max-val 30`):

```
train_loss = 0.00210, eval_loss = 0.00224, 16 steps, ~22 samples/s
```

## Run

```bash
# 1. Dump a working subset of COCO keypoints (smoke = ~30 s, full = hours)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/our_vitpose_base/dump_coco_keypoints.py \
  --max-train 5000 --max-val 1000

# 2. Train through the in-repo HF backend
uv run core/p06_training/train.py \
  --config notebooks/vitpose_finetune_reference/our_vitpose_base/06_training.yaml
```

## Files

| File | Purpose |
|---|---|
| `05_data.yaml` | COCO-keypoints data contract — paths, names, 17-kpt schema, flip_indices, OKS sigmas, input size |
| `06_training.yaml` | ViTPose-base + HF Trainer recipe matching the reference port (lr 5e-4, 30 ep, bs 32, bf16, cosine warmup 500); `metric_for_best_model: loss` until OKS-AP lands |
| `dump_coco_keypoints.py` | Streams `rom1x38/COCO_keypoints` and writes YOLO-pose layout to `dataset_store/training_ready/coco_keypoints/`. Caps via `--max-train` / `--max-val` |

## Parity invariants vs `../reference_vitpose_base/`

- Same checkpoint (`usyd-community/vitpose-base-simple`).
- Same `input_size: [256, 192]`.
- Same `lr=5e-4`, `epochs=30`, `batch_size=32`, `weight_decay=0.01`,
  `warmup_steps=500`, `scheduler: cosine`.
- Same heatmap params (`stride=4`, `sigma=2.0`).
- Same loss (heatmap MSE × visibility weight).
- Same `metric_for_best_model: loss` (eval heatmap MSE) until the OKS-AP
  metric module lands in `core/p08_evaluation/`.
- Same `bf16: true`. Flip both off for fp32 bit-parity if needed.

## Known limitations

- **Post-train viz callbacks (data_preview, val_predictions, error
  analysis) emit warnings and skip rendering for the keypoint task.**
  They assume the legacy YOLO-pose target shape (`{boxes, keypoints}`
  per full-frame image), but the top-down dataset emits per-person
  crops + heatmap targets. Training itself runs cleanly; the viz
  pipeline needs a top-down-aware branch in
  `core/p06_training/hf_callbacks.py` and
  `core/p05_data/run_viz.py::_stats_keypoint`. Tracking as a follow-up.
- **No OKS-AP metric** in `core/p08_evaluation/` yet — `loss` is used
  for best-checkpoint selection. Real OKS-AP comparison vs the
  reference run requires a pycocotools wrapper (~150 LOC) that
  decodes heatmaps via `processor.post_process_pose_estimation`,
  scales coords back to image space using stored bbox info, and runs
  `COCOeval(..., "keypoints")`.
