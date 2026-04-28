# CLAUDE.md — notebooks/vitpose_finetune_reference/

Isolated reproduction of HF ViTPose-base fine-tuning on COCO-keypoints
(`usyd-community/vitpose-base-simple`, top-down, 17 keypoints).

**Purpose**: known-good baseline to diff against when our in-repo
keypoint-detection pipeline (HF backend, `hf_keypoint`) lands and we
need to verify parity.

Upstream:
- Docs: <https://huggingface.co/docs/transformers/en/model_doc/vitpose>
- Model card: <https://huggingface.co/usyd-community/vitpose-base-simple>
- Paper: <https://arxiv.org/abs/2204.12484>

## Status (2026-04-28)

| Side | State | Notes |
|---|---|---|
| `reference_vitpose_base/` | ✅ smoke-verified end-to-end | Tested with `.venv-notebook/` (`transformers 5.6.0.dev0`) — 30 persons × 1 epoch trains in ~3 s; inference grid renders correct GT/pred skeletons. |
| `our_vitpose_base/` | 🟡 blocked | Configs scaffolded; needs `hf_keypoint` arch + builder in `core/p06_models/hf_model.py`, plus a keypoint loss path + OKS metric in `core/p06_training/` and `core/p08_evaluation/`. See `our_vitpose_base/README.md` for the explicit TODO list. |

The `core/p05_data/keypoint_dataset.py` loader and the
`task: keypoint` branches in `core/p06_training/{trainer,hf_trainer,_common}.py`
already exist — only the model registry entry and metric/loss glue are missing.

## Layout

```
.
├── CLAUDE.md                       (this file)
├── README.md                       (human-facing overview)
├── .gitignore
│
├── reference_vitpose_base/         upstream .py port (finetune.py + inference.py + runs/)
└── our_vitpose_base/               same experiment via core/p06_training/ (configs + status README)
```

Same two-kind folder convention as the other `notebooks/*_finetune_reference/`
folders.

## Venv

> **⚠ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT
> the main `.venv/`. Always invoke via `.venv-notebook/bin/python ...`.

Setup via `scripts/setup-notebook-venv.sh`. ViTPose was added in
`transformers 4.45`; the current `.venv-notebook/` ships `5.6.0.dev0`
which is fine. Older pins (< 4.45) 404 on `from_pretrained`.

## Dataset

### COCO-keypoints — `rom1x38/COCO_keypoints`
- Public parquet mirror; no token required.
- Schema (per row, **multi-person** per image):
  `{image: PIL, image_id: int, bboxes: list[[x,y,w,h]], keypoints: list[[[x,y,v]*17]]}`.
  Pixel-space keypoint coords; visibility `v ∈ {0=unlabeled, 1=occluded, 2=visible}`.
- The reference loader flattens to one row per labeled person (drops
  persons with zero visible keypoints).
- 17 keypoints per person (COCO order: nose, eyes, ears, shoulders,
  elbows, wrists, hips, knees, ankles).
- ViTPose is **top-down** — input is a person crop produced by an
  upstream detector. The reference recipe uses GT person boxes from
  the COCO `bbox` field for training and val. Replace with detector
  predictions only when measuring full pipeline OKS-AP.
- Train split = 22 parquet shards (~19 GB images). Smoke runs use
  `--subset N` which switches to streaming and caps at N persons —
  `--subset 30 --epochs 1` finishes in seconds without downloading shards.

## Key config invariants (for `our_vitpose_base/` once wired)

```yaml
seed: 42

model:
  arch: hf_keypoint                # ⚠ NOT YET REGISTERED — see our_vitpose_base/README.md
  pretrained: usyd-community/vitpose-base-simple
  num_keypoints: 17
  input_size: [256, 192]           # ViTPose default (H, W) — non-square
  ignore_mismatched_sizes: true    # decoder head reshapes per num_keypoints

tensor_prep:
  input_size: [256, 192]
  rescale: true
  normalize: imagenet
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  applied_by: hf_processor         # VitPoseImageProcessor owns rescale+normalize

training:
  bf16: true                       # safe on ViTPose (regression + heatmap MSE)
  amp:  false
  # Reference uses `loss` (eval heatmap MSE) for in-loop best selection.
  # Real metric is OKS-AP via offline pycocotools eval — wire under key
  # `AP` once the in-repo p08 keypoint metric lands.
  metric_for_best_model: AP    # target — until the OKS-AP metric lands,
                               # use `loss` (matches reference smoke today)

logging:
  report_to: none
  # Do NOT set run_name — HF Trainer uses it as the output folder name (ghost-folder footgun).
```

## Observability

Once `our_vitpose_base/` runs, both sides will produce `data_preview/`,
`val_predictions/{epochs/,best.png}`, `val_predictions/error_analysis/`,
`test_predictions/`, and `test_results.json` — same tree as the other
references. Keypoint-specific viz (skeleton overlay, per-joint PCK
heatmaps) routes through `utils/viz.py::draw_keypoints` (already used
by the inference path).

## Conversion gotchas (when porting upstream → `.py`)

1. Strip shell installs (`!pip`), `notebook_login()`, `push_to_hub=True`.
2. `display(...)` commented out.
3. ViTPose **expects person-crop input, not full image** — the
   processor's `boxes=` arg drives the crop. For training, feed GT boxes
   from the COCO annotation; for inference, feed detector outputs.
4. Output is heatmaps (per joint, `H/4 × W/4`), decoded by the
   processor's `post_process_pose_estimation`. Do not bypass it.
5. COCO keypoints reshape to `(17, 3)` — `[x, y, v]` per joint.
   Visibility `v=0` → not labeled (mask from loss); `v=1` → occluded
   but labeled; `v=2` → visible.
6. **HF Trainer doesn't see eval_loss for free.** ViTPose's output is
   `VitPoseEstimatorOutput(heatmaps=...)` with no `.loss` attribute, so
   the default `prediction_step` reports nothing on eval. The reference
   subclass overrides `prediction_step` to recompute loss manually —
   keep that override when porting any HF model whose forward returns a
   loss-less output object.
7. **Module side effects must be gated.** `inference.py` imports helpers
   from `finetune.py` — model load + dataset flatten are wrapped in
   `_main()` so the import doesn't trigger them.

## Speed notes (reference-side)

ViTPose-base is ~89M params, smaller crops (256×192). Estimated
throughput on a 5090 at `batch_size=32` is ~3–5 it/s (from public
ViTPose benchmarks; not yet measured locally). Full COCO train (149k
person instances) at 30 epochs ≈ 4–6 hours; will be confirmed once the
first full run lands.

`--subset N` switches the dataset loader to **streaming mode** and stops
after N persons, so smoke runs do not download the full 19 GB train
shard set. Verified smoke: `--subset 30 --epochs 1 --batch-size 8`
runs to completion in ≈30 s including dataset flatten + 1 train + 1
eval pass + checkpoint save. Pass `--subset 0` for the full
non-streaming load.

## Invariants for reference-vs-ours comparison

- Same dataset (same COCO split + person-box source).
- Same pretrained checkpoint (`usyd-community/vitpose-base-simple`).
- Same input size (`[256, 192]`) + ImageNet normalize via the
  processor (not a manual `Normalize` on top).
- Same batch / lr / epochs / warmup.
- Same offline OKS-AP eval (pycocotools keypoint).
- In-loop selection uses `eval_loss` (weighted heatmap MSE) on both
  sides until the in-repo OKS-AP metric lands.

## Known gotchas (carry-overs from repo-level CLAUDE.md)

- **`report_to: none`** on HF Trainer — wandb callback hard-fails without `wandb login`.
- **Do not set `logging.run_name`** — HF Trainer derives `output_dir` from it (ghost dirs).
- **HF-backend checkpoints save with `hf_model.` key prefix** — strip
  before `from_pretrained` reload / Optimum export.
- **`ignore_mismatched_sizes: true`** required because the keypoint
  head's final conv reshapes per `num_keypoints`.
- **bf16 vs fp32**: ViTPose's heatmap MSE is bf16-safe in practice.
  Flip to fp32 only for bit-identical upstream parity.
