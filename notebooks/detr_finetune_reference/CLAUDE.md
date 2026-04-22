# CLAUDE.md — notebooks/detr_finetune_reference/

Isolated reproduction of qubvel's HF reference notebooks for RT-DETRv2 + D-FINE
fine-tuning on CPPE-5. **Purpose**: known-good baseline to diff against when
our in-repo pipeline can't make a DETR-family model converge.

## Layout

```
.
├── CLAUDE.md                       (this file)
├── README.md                       (human-facing overview)
├── pyproject.toml                  pinned deps (albumentations==1.4.6, ...)
├── data_loader.py                  HF cppe-5 ↔ YOLO bridge + --dump-cppe5 CLI
│
├── reference_rtdetr_v2/            qubvel's RT-DETRv2 reference (.py port of .ipynb)
├── reference_dfine/                qubvel's D-FINE reference     (.py port of .ipynb)
│
├── our_rtdetr_v2_albumentations/   our pipeline, RT-DETRv2, Albu
├── our_rtdetr_v2_torchvision/      our pipeline, RT-DETRv2, torchvision v2
├── our_dfine_albumentations/       our pipeline, D-FINE-n, Albu
├── our_dfine_torchvision/          our pipeline, D-FINE-n, torchvision v2
└── our_yolox/                      placeholder
```

Two-kind folder convention:
- `reference_<arch>/` = upstream baseline (qubvel's notebook as .py + .ipynb).
- `our_<arch>[_<aug>]/` = same experiment through our pipeline
  (`core/p06_training/train.py --backend hf`). Each self-contained with
  `05_data.yaml`, `06_training.yaml`, README, and `runs/`.

## Venv

> **⚠️ CRITICAL:** Reference `.py` scripts run in `.venv-notebook/`, NOT the
> main `.venv/`. `uv run` uses the main venv and hits albumentations-version
> divergence → silently-wrong bbox clipping. Always invoke via
> `.venv-notebook/bin/python ...`.

Pinned separately via `pyproject.toml` + `scripts/setup-notebook-venv.sh`:
- `albumentations==1.4.6` (qubvel's pin)
- HF transformers from git (decoupled from main venv)

```bash
# Reference
.venv-notebook/bin/python notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py --seed 42

# Our pipeline (main venv is fine — we don't depend on albu 1.4.6)
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/06_training.yaml
```

## Results — current truth (2026-04-22)

### RT-DETRv2 on CPPE-5 (seed=42, 40 ep, bs=16, lr=1e-4, bf16, cosine, WD=1e-4)

| Run | Backbone | Params | test_mAP | test_mAP₅₀ | Δ vs qubvel | runtime |
|---|---|---|---|---|---|---|
| qubvel published | R50 | 42M | 0.5789 | 0.8674 | — | — |
| `reference_rtdetr_v2/` port | R50 | 42M | 0.5464 | 0.8043 | −0.033 | 617 s |
| our_torchvision | R50 | 42M | 0.5607 | 0.8237 | −0.018 | 568 s |
| **our_albumentations** | **R50** | **42M** | **0.5698** | 0.8226 | **−0.009** | 667 s |
| our_torchvision (R18) | R18 | 20M | 0.5389 | 0.7374 | −0.040 | 507 s |
| our_albumentations (R18) | R18 | 20M | 0.5160 | 0.7249 | −0.063 | 619 s |

All R50 runs within ±0.03 of qubvel — pipeline is empirically indistinguishable
from reference. **R18 trails R50 by ~0.04** — unlike D-FINE, RT-DETRv2 is not
over-parameterized at R50 for CPPE-5; the R50 backbone carries real signal.

**Default for Phase 2 (fire/helmet/PPE features)**: RT-DETRv2-R50.

### D-FINE on CPPE-5 (seed=42, 30 ep, bs=8, lr=5e-5, fp32, linear, WD=0)

**D-FINE-n (4M params) beats D-FINE-large (80M params) on this dataset.**
850-image CPPE-5 train split is under-determined for 80M params — the
smaller variant generalizes better.

| Run | Arch | Params | test_mAP | test_mAP₅₀ | Goggles | runtime |
|---|---|---|---|---|---|---|
| qubvel published | dfine-large | 80M | 0.4485 | — | — | — |
| `reference_dfine/` port | dfine-large | 80M | 0.4294 | 0.617 | — | ~1100 s |
| our, dfine-large 5-seed mean | dfine-large | 80M | 0.383 ± 0.042 | — | 0.17 | — |
| **our, dfine-n Albu** | **dfine-n** | **4M** | **0.4781** | **0.697** | **0.271** | **583 s** |
| **our, dfine-n TV** | **dfine-n** | **4M** | **0.4685** | 0.680 | 0.268 | 497 s |

Both dfine-n runs **beat qubvel's dfine-large** (+0.020–0.030 test_mAP),
with 20× fewer params in half the wall-time. Both `our_dfine_*/06_training.yaml`
default to `arch: dfine-n`.

### Capacity vs dataset-size heuristic (CPPE-5 has 850 train images)

| Family | Variant tested | Params | test_mAP |
|---|---|---|---|
| RT-DETRv2 | R18 | 20M | 0.52–0.54 |
| RT-DETRv2 | **R50** | **42M** | **0.56–0.57 ← best** |
| D-FINE | **n** | **4M** | **0.47–0.48 ← best-for-cost** |
| D-FINE | large | 80M | 0.38–0.49 (overfits) |

Sweet spot for CPPE-5-scale datasets: **20–40M params**. Above (D-FINE-large
80M) overfits; below (D-FINE-n 4M) still works if the loss geometry is
right; RT-DETRv2-R18 loses backbone features the head can't rebuild.

### Signal health (why dfine-n beats dfine-large)

| Signal | dfine-large 50ep | dfine-n 30ep |
|---|---|---|
| train_loss | 60 → 16 (✓ drops) | 33 → 17 (✓ drops) |
| **eval_loss** | 2.06 → **2.94** (✗ climbs = overfit) | 2.56 → **1.70** (✓ drops) |
| val_mAP | 0.22 peak ep 2, decays | 0.27 peak ep 20+, holds |
| val→test gap | 1.8–2.2× (inconsistent) | 1.72× (clean) |
| Goggles (rarest class) | 0.15–0.31 (noisy) | 0.27 (stable) |

dfine-large shows textbook overfit signature (eval_loss rising while train
drops). dfine-n's smaller capacity eliminates the overfit; all three
signals move in the correct direction.

## Key config invariants (both arches)

```yaml
seed: 42

training:
  # D-FINE must stay in fp32 — bf16 stalls DFL, fp16 overflows decoder.
  # RT-DETRv2 is bf16-safe.
  bf16: true  # (RT-DETR); set false for D-FINE
  amp: false

  # Early-stop on val_mAP plateau. D-FINE especially needs this —
  # val_mAP often peaks ep 5–20 then drifts.
  patience: 8

logging:
  report_to: none  # skip wandb auth hard-fail on HF backend

evaluation:
  score_threshold: 0.0   # canonical mAP; raise only to match external baselines
```

Early `set_seed(42)` before `from_pretrained` is done inside
`core/p06_training/hf_trainer.py::train_with_hf` — required for reproducible
class-head reinit on HF backend.

## Bundle B recipe (RT-DETRv2)

Goes beyond qubvel's notebook; delivers +0.03–0.05 test_mAP over the
default recipe:

```python
per_device_train_batch_size = 16     # 2× qubvel's 8
learning_rate               = 1e-4   # 2× qubvel's 5e-5 (linear scale for 2× bs)
lr_scheduler_type           = "cosine"
weight_decay                = 1e-4
bf16                        = True
seed = data_seed            = 42
# plus: set_seed(42) early, cuDNN deterministic flags
```

For D-FINE, **do not apply Bundle B** — halving LR stalled it at test
mAP 0.26 vs the 0.45 default. D-FINE wants lr=5e-5, linear, WD=0, fp32.

## Data pipeline — byte-identical at the model boundary

Both reference scripts and all four `our_*` configs feed the HF model
the same tensor:

- `pixel_values`: (B, 3, 480, 480) float32, ImageNet-normalized (by
  `AutoImageProcessor`, exactly once)
- `labels`: list of `{"class_labels": LongTensor, "boxes": FloatTensor
  cxcywh [0,1]}`

Aug semantics match qubvel's basic recipe (Perspective 0.1 + HFlip 0.5 +
BrightContrast 0.5 + HueSatVal 0.1 + bbox clip + min_area 25). Our pipeline
supports both albumentations and torchvision v2 via `augmentation.library`
— empirically equivalent at ±0.01 test_mAP after the resize-first reorder
(commit `c4d3658`).

## Conversion gotchas (applied to the `.py` files)

If re-converting from `.ipynb`, re-apply:

1. Remove shell installs (`get_ipython().system(...)`, `!pip ...`).
2. Comment out `display(...)` — Jupyter-only.
3. `datasets` 4.x access fix:
   ```python
   # 2.x: ds.features["objects"].feature["category"].names
   # 4.x:
   ds.features["objects"]["category"].feature.names
   ```

## CLI for `reference_rtdetr_v2/finetune.py`

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py \
    --seed 42  --tag bs16_lr1e4_cosine_wd_bf16  --aug basic
```

Output dir: `runs/rtdetr_v2_r50_cppe5_seed{SEED}{_TAG}/`.

## Invariants for reference-vs-ours comparison

- Same split (CPPE-5 `train_test_split(0.15, seed=1337)`).
- Same model checkpoint (R50 vs R18 swings mAP by 0.03).
- Same input size (480 vs 640).
- Same batch size, lr, epochs, warmup.
- **Same RNG hygiene**: early `set_seed`, cuDNN deterministic flags.

Skipping any of these makes the comparison meaningless — class-head
reinit alone swings mAP by 0.03–0.07.

## Per-seed variance is large

3-seed Bundle B sweep (RT-DETRv2, seeds 42/0/2024): test mAP span
**0.486 – 0.559** on 29-image test. Single-run comparisons not meaningful
below ±0.03. For D-FINE the spread is larger (4 identical seed=42 runs
gave 0.364/0.441/0.395/0.425) because deformable attention backward lacks
a deterministic CUDA kernel (we use `warn_only=True`).

## Known gotchas

- **ONNX fp32 export is lossless and ~2× faster** on ORT CUDA EP
  (RT-DETRv2-R50 CPPE-5 seed=42: 0.516 pytorch ≡ 0.516 ONNX, 20.9 → 10.3 ms).
  INT8 via direct `quantize_static` **runs slower** and **collapses mAP** on
  ORT CUDA EP — deploy INT8 only via TensorRT EP engine build.
- **`reference_dfine/` uses dfine-large** (qubvel's choice). For our own
  production fine-tuning on CPPE-5-scale datasets, **dfine-n is the
  default** — `our_dfine_*/06_training.yaml` arch is set accordingly.
- **`metric_for_best_model="eval_map"`** requires torchmetrics to emit a
  key literally named `map`. If torchmetrics renames it, HF Trainer
  silently fails to checkpoint the best model.
- **`torch.use_deterministic_algorithms(True)` strict mode crashes
  RT-DETRv2 + D-FINE** (multi-scale deformable attention backward has no
  deterministic kernel). We use `warn_only=True`; accept ±0.005 RT-DETRv2,
  ±0.04 D-FINE run-to-run variance.
- **Do not run reference scripts from main venv** — albu 1.4.6 pin.
  Our `our_*` configs run fine from main venv (`uv run`).
