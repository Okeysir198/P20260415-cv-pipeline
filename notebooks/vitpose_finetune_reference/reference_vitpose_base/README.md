# reference_vitpose_base — HF ViTPose 2D-pose / keypoint reference

Runnable `.py` port of HF's ViTPose docs example. Used as the known-good
baseline to diff against our in-repo keypoint training path
(`our_vitpose_base/`, currently blocked on arch wiring).

## Contents

| File | Purpose |
|---|---|
| `finetune.py` | ViTPose-base fine-tune on `rom1x38/COCO_keypoints` (top-down, GT person boxes). Flags: `--seed`, `--tag`, `--epochs`, `--subset` (persons; 0=full set), `--lr`, `--batch-size`, `--num-workers`, `--bf16`. Writes to `runs/{TAG_}seed{SEED}/`. |
| `inference.py` | Post-training viz — loads `best/`, overlays GT + pred skeletons for N val crops in a single PNG. Stub for full OKS-AP eval (TODO: pycocotools wrapper once full-data run lands). |

Upstream sources:
- ViTPose model card: <https://huggingface.co/usyd-community/vitpose-base-simple>
- HF docs: <https://huggingface.co/docs/transformers/en/model_doc/vitpose>
- Paper (Xu et al. 2022): <https://arxiv.org/abs/2204.12484>

## Run

```bash
# Smoke (verified end-to-end on this branch — ~30 s)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \
  --seed 42 --subset 30 --epochs 1 --batch-size 8 --tag smoke

# Default smoke (5k persons, ~30 min, monotonic loss)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \
  --seed 42 --subset 5000

# Full COCO train (~149k person instances, ~4–6 h on a 5090)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \
  --seed 42 --subset 0 --epochs 30

# Inference grid (GT|Pred skeletons side-by-side)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/inference.py \
  --run-dir notebooks/vitpose_finetune_reference/reference_vitpose_base/runs/seed42 \
  --n 16
```

`--subset N > 0` streams the dataset (no full ~19 GB train pull) and
caps at N persons; `--subset 0` triggers the full non-streaming load.

### Smoke result (verified 2026-04-28)

| Run | train_loss | eval_loss | wall |
|---|---|---|---|
| `--subset 30 --epochs 1 --batch-size 8` | 0.00205 | 0.00240 | ~30 s end-to-end |

### 10% learning-ability run (verified 2026-04-28)

`--subset 15000 --epochs 15 --batch-size 32 --bf16`, GPU 1 (5090), 20 min total wall.

```
ep 1   eval_loss = 0.001286
ep 4   eval_loss = 0.001201    -7.3% vs ep1   warmup ends
ep 6   eval_loss = 0.001133    -12%
ep 9   eval_loss = 0.001084
ep11   eval_loss = 0.001078    ★ MIN  -16.2% vs ep1
ep15   eval_loss = 0.001091    slight upward drift = mild overfitting
final  train_loss = 0.000488   (3.7x reduction from ep0)
```

Read: model genuinely fits (train loss drops 3.7×, eval loss bottoms
at ep11 then plateaus + drifts up). On a 10% slice, **11 epochs is
the sweet spot** — beyond that gains are zero or slightly negative.
For full-data runs expect 15–20 epochs to be the right horizon, which
matches the qubvel-style guidance that the canonical 30 ep over-trains
on a single seed. No training instability — gradient norms 0.0007–0.005
throughout. Best ckpt under `runs/learn10pct_seed42/best/`.

### Headline OKS-AP from a full 30-epoch run

**Pending** — requires the offline pycocotools wrapper (~150 LOC; see
`inference.py` stub). Will be filled in once that lands.

## Hyperparameter recipe (baked into `finetune.py`)

| Knob | Value | Note |
|---|---|---|
| `model` | `usyd-community/vitpose-base-simple` | ViTPose-base, 17-kpt simple head |
| `dataset` | `rom1x38/COCO_keypoints` | Public parquet mirror of COCO 2017 person keypoints; no `HF_TOKEN` |
| `epochs` | 30 | |
| `lr` | 5e-4 | AdamW, weight_decay 0.01 |
| `warmup` | 500 steps, cosine decay | |
| `per_device_train_batch_size` | 32 | |
| `input_size` | (256, 192) (H, W) | ViTPose default; non-square |
| `heatmap_size` | (64, 48) | input/4 |
| `heatmap_sigma` | 2.0 px | Gaussian per joint |
| Loss | weighted heatmap MSE | weight = visibility mask `v > 0` |
| `bf16` | on | safe; flip off for fp32 parity |

## Top-down recipe (key invariant)

ViTPose **does not detect people**. The processor crops a person from
the input image using the `boxes=` argument, resizes to `(256, 192)`,
and ImageNet-normalizes. For training we use **GT bboxes from COCO's
`bbox` field**; for full pipeline OKS-AP at deployment time you swap in
detector predictions.

The `rom1x38/COCO_keypoints` schema delivers **multiple persons per
row** (`bboxes: list[[x,y,w,h]]`, `keypoints: list[[[x,y,v]*17]]`).
`_flatten_persons()` expands these into one row per labeled person and
drops persons with zero visible keypoints.

`_expand_bbox()` widens the COCO `[x, y, w, h]` to ViTPose's H:W aspect
ratio with a 1.25× padding factor — matches mmpose's
`TopDownAffine`/`GetBBoxCenterScale` behaviour.

## Heatmap targets

Standard top-down recipe: each visible joint becomes a 2D Gaussian
peaked at the resized crop coordinate, σ = 2 px on a 64×48 grid. Joints
with `visibility == 0` (unlabeled) are masked from the loss via
`target_weight`. The processor's `post_process_pose_estimation` decodes
heatmaps back to image-space coordinates at inference time.

## Full OKS-AP eval (post-training)

`inference.py` ships a viz-only path that prints a notice and writes
`oks_ap.json: {"status": "viz_only", "n": <N>}`. For real OKS-AP
numbers you need the official COCO val annotations JSON
(`person_keypoints_val2017.json`) + `pycocotools`:

```bash
# in .venv-notebook/
.venv-notebook/bin/pip install pycocotools
# Wrapper (TODO — to be added once full-data run lands):
#   1. predict keypoints for every val annotation
#   2. dump a COCO-format predictions JSON
#   3. COCOeval(coco_gt, coco_dt, "keypoints").{evaluate,accumulate,summarize}
```

The HF dataset row carries `image_id` for joining back to the
official annotation set.

## Conversion notes vs upstream HF docs

1. Stripped shell installs (`!pip install ...`) and `notebook_login()`.
2. `display(...)` removed.
3. `push_to_hub=False`, `report_to="none"`.
4. Dataset: `rom1x38/COCO_keypoints` (public parquet mirror; no token).
   Original `keremberke/coco-keypoints` mirror referenced in earlier
   tutorials is no longer on the Hub.
5. Loss ported as a plain `Trainer.compute_loss` override (upstream
   relies on `mmpose`'s `KeypointMSELoss` which is heavier-weight).
6. Custom `_PoseTrainer.prediction_step` to surface `eval_loss` —
   ViTPose's output object has no `.loss` attribute, so the default
   eval loop reports nothing. This override recomputes the loss at
   eval time so `metric_for_best_model="loss"` works.
7. Heavy module-level work (model + processor load, dataset
   flatten + map) is gated inside `_main()` so `inference.py`'s
   `from finetune import ...` stays cheap.

## Re-fetching the upstream snippet

The HF ViTPose docs page is the authoritative upstream reference (no
frozen `.ipynb` exists — the docs page itself carries the runnable
code blocks):

```bash
curl -L https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/vitpose.md \
  -o vitpose.md
```
