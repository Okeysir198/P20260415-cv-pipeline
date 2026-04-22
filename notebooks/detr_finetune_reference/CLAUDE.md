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
├── our_rtdetr_v2_albumentations/   our pipeline, RT-DETRv2-R50, Albu
├── our_rtdetr_v2_torchvision/      our pipeline, RT-DETRv2-R50, torchvision v2
├── our_dfine_albumentations/       our pipeline, D-FINE-n, Albu
├── our_dfine_torchvision/          our pipeline, D-FINE-n, torchvision v2
├── our_yolox/                      our pipeline, YOLOX-m (official), Albu, 50 ep
└── our_yolox_torchvision/          our pipeline, YOLOX-m (official), TV + Mosaic + MixUp, 100 ep
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

## Results — 6-way CPPE-5 reference reproduction (seed=42, 2026-04-23)

All six `our_*/` runs were rerun sequentially on 2 GPUs (one model per GPU, 3 batches) after the observability overhaul landed. Configs use codebase defaults for every viz block (see `core/p06_training/CLAUDE.md` → "Post-train observability").

### Detection performance

| Run | Backend | Arch | Epochs | Aug | test_mAP | test_mAP₅₀ |
|---|---|---|---|---|---|---|
| qubvel RT-DETRv2 | — | R50 | 40 | Albu basic | 0.5789 | 0.8674 |
| qubvel D-FINE | — | dfine-large | 30 | Albu basic | 0.4485 | — |
| **our_rtdetr_v2_torchvision** | HF | R50 | 40 | torchvision v2 | **0.5600** | **0.8231** |
| **our_rtdetr_v2_albumentations** | HF | R50 | 40 | Albumentations | **0.5483** | **0.8264** |
| **our_dfine_torchvision** | HF | dfine-n | 30 | torchvision v2 | **0.4532** | **0.6828** |
| **our_dfine_albumentations** | HF | dfine-n | 30 | Albumentations | **0.4473** | **0.6778** |
| **our_yolox_torchvision** | pytorch | yolox-m (official) | 100 | TV + Mosaic + MixUp | — | **0.8668** |
| **our_yolox** | pytorch | yolox-m (official) | 50 | Albumentations | — | **0.7388** |

Takeaways:
- **RT-DETRv2-R50** is the strongest DETR recipe on CPPE-5 — inside ±0.03 of qubvel on both aug libraries. Default for Phase 2 features.
- **D-FINE-n** beats `reference_dfine/` port (0.4294) at 4M params vs 80M. Both our D-FINE configs pin `arch: dfine-n`; do not use `dfine-large` on sub-2k-image training sets.
- **YOLOX-m + Mosaic + MixUp + 100 ep** (our_yolox_torchvision) scores **val mAP₅₀ 0.867** — highest of any run in this folder. Wins when speed matters more than test-split framing.

### Capacity heuristic (CPPE-5 = 850 train images)

| Family | Variant | Params | test_mAP₅₀ |
|---|---|---|---|
| YOLOX (Mosaic, 100 ep) | m | 25M | **0.867** |
| RT-DETRv2 | R50 | 42M | **0.823–0.826** |
| D-FINE | **n** | **4M** | **0.677–0.683** |
| RT-DETRv2 | R18 | 20M | 0.72–0.73 (earlier runs) |
| D-FINE | large | 80M | 0.38–0.49 (overfits) |

Sweet spot for CPPE-5-scale (~1k-image) detection: 4M–42M params. Above that overfits; below that still works if the loss geometry is right (dfine-n's distribution focal).

## Per-run observability (every run)

Each `our_*/runs/seed42/` now holds a full 3-axis report — identical layout on both backends. See `core/p06_training/CLAUDE.md` for the tree + config toggles. Expect ~2–3 GB per run (checkpoints + preview PNGs + error_analysis gallery).

Known gotcha: `_finalize_training` on HF backend can create a stray `notebooks/.../runs/` nested dir when `logging.save_dir` is relative and the override path contains `notebooks/`. Safe to `rm -rf our_*/notebooks/` post-hoc — the real artifacts live under `runs/seed42/`.

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
