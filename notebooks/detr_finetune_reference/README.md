# DETR-family fine-tune reference scripts

Direct ports of qubvel's HF reference notebooks for RT-DETRv2 and D-FINE,
converted to plain Python so they can be run, diffed, and compared against our
in-repo `DetectionTrainer` pipeline as a known-good baseline.

## Layout

```
notebooks/detr_finetune_reference/
├── README.md                       (this file)
├── requirements.txt                pinned deps
├── rtdetr_v2_finetune_cppe5.py     RT-DETRv2 fine-tune on CPPE-5 (runnable)
├── dfine_finetune_cppe5.py         D-FINE fine-tune on CPPE-5 (runnable)
├── rtdetr_v2_inference.py          RT-DETRv2 inference (runnable)
├── dfine_inference.py              D-FINE inference (runnable)
├── data_loader.py                  YOLO → HF Dataset bridge for when we later
│                                   want to swap CPPE-5 → our features
└── reference/                      Untouched original .ipynb notebooks
    ├── RT_DETR_v2_finetune_on_a_custom_dataset.ipynb
    ├── DFine_finetune_on_a_custom_dataset.ipynb
    ├── RT_DETR_v2_inference.ipynb
    └── DFine_inference.ipynb
```

## Setup

```bash
bash scripts/setup-notebook-venv.sh
# creates .venv-notebook/ with albumentations==1.4.6 + torchmetrics + HF transformers (git)
```

## Phase 1 — replicate the reference result on CPPE-5

Run the notebooks AS-IS on their native CPPE-5 dataset (~1k training images,
5 classes). Goal: confirm the reference code trains cleanly in our environment,
so any gap vs our in-repo pipeline is attributable to our code, not the
notebook recipe.

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py
.venv-notebook/bin/python notebooks/detr_finetune_reference/dfine_finetune_cppe5.py
```

Expected outcome (from qubvel's published results):
- **RT-DETRv2-R50** on CPPE-5 @ 40 epochs: val `mAP` ≈ 0.34
- **D-FINE** on CPPE-5 @ 30 epochs: val `mAP` ≈ 0.33

## Phase 2 — swap CPPE-5 for our features (optional, after Phase 1 passes)

Only after the reference scripts reproduce the expected CPPE-5 numbers:

1. In each script, replace the CPPE-5 loading block (around line 55-65):
   ```python
   # REMOVE:
   # from datasets import load_dataset
   # dataset = load_dataset("cppe-5")
   # if "validation" not in dataset: ...

   # ADD:
   from data_loader import load_feature_dataset
   dataset = load_feature_dataset("fire_detection", subset=0.05)
   ```
2. Nothing else changes. `data_loader.py` emits a schema byte-compatible with CPPE-5
   (COCO bbox format, `ClassLabel` category feature), so the downstream
   Albumentations / `image_processor` / `CPPE5Dataset` / `Trainer` code all work
   verbatim.

## Conversion notes — what differs from the original notebooks

The `.py` files are direct ports of the `.ipynb` via `jupyter nbconvert --to script`,
with three mechanical cleanups applied:

1. **Shell installs removed** — `!pip install …` / `get_ipython().system(…)` lines
   stripped. Deps live in `requirements.txt` and are installed by
   `scripts/setup-notebook-venv.sh`.
2. **Jupyter `display(...)` commented out** — visualization cells don't run in
   plain Python. Training/eval behavior unchanged.
3. **`datasets` 4.x syntax fix** — the notebook accesses
   `ds.features["objects"].feature["category"].names` (valid in datasets 2.x),
   which raises `AttributeError` on datasets 4.x. The `.py` uses
   `ds.features["objects"]["category"].feature.names` instead. Same result.

No training-behavior-affecting changes. Original notebooks are preserved under
`reference/` for byte-level verification.

## Canonical training args (from qubvel's notebooks — DO NOT change)

| Arg | RT-DETRv2 | D-FINE |
|---|---|---|
| `num_train_epochs` | 40 | 30 |
| `learning_rate` | 5e-5 | 5e-5 |
| `warmup_steps` | 300 | 300 |
| `max_grad_norm` | 0.1 | 0.1 |
| `per_device_train_batch_size` | 8 | 8 |
| `checkpoint` | `PekingU/rtdetr_v2_r50vd` | `ustc-community/dfine-large-coco` |
| `image_size` | 480 | 480 |

## Results log

| Date | Script | Dataset | Subset | Epochs | Final mAP50 | Notes |
|---|---|---|---|---|---|---|
| — | `rtdetr_v2_finetune_cppe5.py` | CPPE-5 | full (1k) | 40 | TBD | expected ≈ 0.34 per qubvel |
| — | `dfine_finetune_cppe5.py` | CPPE-5 | full (1k) | 30 | TBD | expected ≈ 0.33 per qubvel |
