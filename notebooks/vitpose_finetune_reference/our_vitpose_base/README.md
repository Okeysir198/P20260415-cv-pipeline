# our_vitpose_base — in-repo ViTPose run (⚠ blocked)

Same recipe as `../reference_vitpose_base/`, routed through
`core/p06_training/train.py --backend hf` for apples-to-apples
comparison once the keypoint training arch lands in `core/`.

## Status — 2026-04-28

**Blocked.** `core/p06_models/` does not yet expose a trainable HF
keypoint arch. The configs in this folder define the contract that
arch must satisfy.

What exists already:
- `core/p05_data/keypoint_dataset.py` — YOLO-pose loader, transforms,
  flip_indices support.
- `core/p06_training/_common.py` + `trainer.py` + `hf_trainer.py` —
  `task: keypoint` is enumerated as a valid task; dataset wiring works.
- `core/p06_models/{pose_base,pose_registry,mediapipe_pose,rtmpose}.py`
  — inference-only pose estimators (cannot be trained).

## Wiring TODO

To unblock this folder, three pieces are needed:

### 1. Register `hf_keypoint` arch — `core/p06_models/hf_model.py`

Add an `@register_model("hf_keypoint")` builder that loads
`VitPoseForPoseEstimation` (or any HF `AutoModelForKeypointDetection`
backbone), exposes `forward_with_loss(pixel_values, target_heatmap,
target_weight) → (loss, outputs)`, and respects
`model.ignore_mismatched_sizes` for the head reshape.

Reference for the loss shape — see
`../reference_vitpose_base/finetune.py::_PoseTrainer.compute_loss`
(weighted heatmap MSE).

### 2. Top-down crop adapter — `core/p05_data/keypoint_dataset.py`

`KeypointDataset` currently emits full-frame YOLO-pose tensors. ViTPose
needs **per-person crops**. Add a `top_down: bool` switch (driven by
`model.top_down` in 06_training.yaml) that, for each annotation:
1. Reads the parent image + bbox.
2. Calls `_expand_bbox(bbox, ratio=H:W, padding=model.bbox_padding)`.
3. Lets the HF processor crop+resize to `model.input_size`.
4. Maps keypoints into crop space.
5. Synthesizes heatmap targets (`training.heatmap.{stride,sigma}`) +
   visibility mask.

Reuse the heatmap encoder from `reference_vitpose_base/finetune.py`.

### 3. OKS-AP metric — `core/p08_evaluation/`

Add a keypoint metric module that:
- Decodes predicted heatmaps via the processor's
  `post_process_pose_estimation`.
- Scales coordinates back to the parent image.
- Runs `COCOeval(coco_gt, coco_dt, "keypoints")` and exposes the
  result under the key `AP` (consumed by HF Trainer's
  `metric_for_best_model: AP`).
- Honours `evaluation.oks_sigmas_from_data: true` by reading
  `oks_sigmas` from `05_data.yaml`.

Once these three land, the configs here should run unchanged.

## Run (once unblocked)

```bash
uv run core/p06_training/train.py \
  --config notebooks/vitpose_finetune_reference/our_vitpose_base/06_training.yaml
```

## Files

| File | Purpose |
|---|---|
| `05_data.yaml` | COCO-keypoints data contract — paths, names, 17-kpt schema, flip_indices, OKS sigmas, input size |
| `06_training.yaml` | ViTPose-base + HF Trainer recipe matching the reference port (lr 5e-4, 30 ep, bs 32, bf16, cosine warmup 500) |

## Parity invariants vs `../reference_vitpose_base/`

- Same checkpoint (`usyd-community/vitpose-base-simple`).
- Same `input_size: [256, 192]`.
- Same `lr=5e-4`, `epochs=30`, `batch_size=32`, `weight_decay=0.01`,
  `warmup_steps=500`, `scheduler: cosine`.
- Same heatmap params (`stride=4`, `sigma=2.0`).
- Same loss (heatmap MSE × visibility weight).
- Same metric (`AP` via pycocotools keypoint eval).
- Same `bf16: true`. Flip both off for fp32 bit-parity if needed.
