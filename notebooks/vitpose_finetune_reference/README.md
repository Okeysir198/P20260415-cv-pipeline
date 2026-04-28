# ViTPose fine-tune reference scripts + in-repo pipeline comparisons

Central ground-truth folder for any 2D human-pose / keypoint-detection (HF
`VitPoseForPoseEstimation`, ViTPose family) work on HF public datasets.

**Experiment: COCO-keypoints fine-tune** — top-down 2D pose recipe ported
from the HF `transformers` ViTPose docs + community fine-tune examples.

Mirrors the layout of `notebooks/detr_finetune_reference/`,
`notebooks/image_classification_finetune_reference/`, and
`notebooks/segformer_finetune_reference/`:

1. **Reference run** — runnable port of the upstream HF docs / cookbook
   for `VitPoseForPoseEstimation`. Known-good baseline for diffing
   against our own pipeline.
2. **In-repo pipeline run** — the same recipe executed through
   `core/p06_training/train.py --backend hf` *(currently blocked on
   wiring an `hf_keypoint` arch into `core/p06_models/` — see CLAUDE.md
   "Status" section)*.

## Layout

```
notebooks/vitpose_finetune_reference/
├── README.md                         (this file)
├── CLAUDE.md                         Claude-facing notes on recipes / gotchas / status
├── .gitignore                        ignores runs/, .venv*/, .ipynb_checkpoints
│
├── reference_vitpose_base/           upstream ViTPose docs/example .py port
│   ├── finetune.py                   COCO-keypoints recipe (top-down, GT-box crops)
│   ├── inference.py                  val-prediction overlay grid (skeleton + dots)
│   └── README.md
│
└── our_vitpose_base/                 same recipe via core/p06_training/  ⚠ blocked
    ├── 05_data.yaml
    ├── 06_training.yaml
    └── README.md                     status + TODO list for wiring the arch
```

`reference_*/` holds the **upstream baseline**. `our_*/` holds **the same
experiment run through `core/p06_training/`** for apples-to-apples comparison
once the arch is registered.

## Setup (once)

Same venv as the other reference folders:

```bash
bash scripts/setup-notebook-venv.sh
```

> **⚠ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT
> the main `.venv/`. Always invoke via `.venv-notebook/bin/python ...`.

ViTPose checkpoints are public — no `HF_TOKEN` required for the default
`usyd-community/vitpose-base-simple` checkpoint. COCO-keypoints data is
streamed from the public parquet mirror `rom1x38/COCO_keypoints` (smoke
runs avoid the full ~19 GB train pull by streaming + capping with
`--subset`).

## Run

```bash
# Reference smoke — verified end-to-end (≈30 s, streams a 30-person subset)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \
  --seed 42 --subset 30 --epochs 1 --batch-size 8 --tag smoke

# Reference full run (≈4–6 h on a 5090, full COCO train)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/finetune.py \
  --seed 42 --subset 0 --epochs 30

# Inference grid (GT + pred skeletons side-by-side)
.venv-notebook/bin/python \
  notebooks/vitpose_finetune_reference/reference_vitpose_base/inference.py \
  --run-dir notebooks/vitpose_finetune_reference/reference_vitpose_base/runs/seed42 \
  --n 16

# Ours (HF backend) — blocked: requires `hf_keypoint` arch in core/p06_models/
# See our_vitpose_base/README.md for the wiring TODO.
uv run core/p06_training/train.py \
  --config notebooks/vitpose_finetune_reference/our_vitpose_base/06_training.yaml
```

## Upstream references

- ViTPose model card (base-simple): <https://huggingface.co/usyd-community/vitpose-base-simple>
- HF `transformers` ViTPose docs: <https://huggingface.co/docs/transformers/en/model_doc/vitpose>
- Original paper (Xu et al. 2022): <https://arxiv.org/abs/2204.12484>
- COCO-keypoints dataset (parquet mirror): <https://huggingface.co/datasets/rom1x38/COCO_keypoints>
  (canonical source: official COCO 2017 keypoints — `cocodataset.org/#keypoints-2017`)
