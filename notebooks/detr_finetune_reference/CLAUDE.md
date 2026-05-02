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
CUDA_VISIBLE_DEVICES=0 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/06_training_r50_norm_false_ema_false.yaml
```

## Results — normalize × EMA ablation (seed=42, 2026-05-02)

Full 2×2 EMA×normalize sweep for RT-DETRv2 (R50 + R18) and D-FINE (n/s/m), all
torchvision augmentation, 20 runs total. Each config has its own file under
`our_rtdetr_v2_torchvision/` and `our_dfine_torchvision/` (naming: `06_training_<arch>_norm_{true,false}_ema_{true,false}.yaml`).

### RT-DETRv2 results (40 epochs, Bundle B: lr=1e-4, cosine, wd=1e-4, bs=16, bf16)

| Arch | normalize | ema | test_mAP₅₀ | save_dir |
|---|---|---|---|---|
| R50 | **false** | false | **0.834** ← best R50 | `r50_norm_false_ema_false` |
| R50 | false | true | 0.804 | `r50_norm_false_ema_true` |
| R50 | true | false | 0.826 | `r50_norm_true_ema_false` |
| R50 | true | true | 0.745 | `r50_norm_true_ema_true` |
| R18 | **false** | false | **0.741** ← best R18 | `r18_norm_false_ema_false` |
| R18 | false | true | 0.734 | `r18_norm_false_ema_true` |
| R18 | true | false | 0.670 | `r18_norm_true_ema_false` |
| R18 | true | true | 0.654 | `r18_norm_true_ema_true` |
| qubvel R50 (reference) | true | false | 0.867 | — |

### D-FINE results (30 epochs, lr=5e-5, linear, wd=0, bs=8, fp32)

| Arch | normalize | ema | test_mAP₅₀ | save_dir |
|---|---|---|---|---|
| dfine-n (4M) | false | **true** | **0.710** ← best dfine-n | `dfine_n_norm_false_ema_true` |
| dfine-n (4M) | false | false | 0.681 | `dfine_n_norm_false_ema_false` |
| dfine-n (4M) | true | false | 0.557 | `dfine_n_norm_true_ema_false` |
| dfine-n (4M) | true | true | 0.556 | `dfine_n_norm_true_ema_true` |
| dfine-s (16M) | false | false | 0.603 | `dfine_s_norm_false_ema_false` |
| dfine-s (16M) | false | true | 0.565 | `dfine_s_norm_false_ema_true` |
| dfine-s (16M) | true | false | 0.462 | `dfine_s_norm_true_ema_false` |
| dfine-s (16M) | true | true | 0.433 | `dfine_s_norm_true_ema_true` |
| dfine-m (31M) | false | true | 0.431 | `dfine_m_norm_false_ema_true` |
| dfine-m (31M) | false | false | 0.400 | `dfine_m_norm_false_ema_false` |
| dfine-m (31M) | true | true | 0.325 | `dfine_m_norm_true_ema_true` |
| dfine-m (31M) | true | false | 0.323 | `dfine_m_norm_true_ema_false` |
| qubvel dfine-large (80M, reference) | true | false | ~0.60 | — |

### Key findings

**1. normalize=false is universally better on CPPE-5** (fire dataset already uses false):

| Model | norm_false best | norm_true best | delta |
|---|---|---|---|
| R50 | **0.834** | 0.826 | +0.008 |
| R18 | **0.741** | 0.670 | **+0.071** |
| dfine-n | **0.710** | 0.557 | **+0.153** |
| dfine-s | **0.603** | 0.462 | **+0.141** |
| dfine-m | **0.431** | 0.325 | **+0.106** |

**2. EMA effect varies by arch** (norm_false baseline):

| Model | ema=false | ema=true | delta |
|---|---|---|---|
| R50 | **0.834** | 0.804 | −0.030 (hurts) |
| R18 | **0.741** | 0.734 | −0.007 (neutral) |
| dfine-n | 0.681 | **0.710** | +0.029 (helps) |
| dfine-s | **0.603** | 0.565 | −0.038 (hurts) |
| dfine-m | 0.400 | **0.431** | +0.031 (helps) |

EMA helps dfine at extremes (n=4M, m=31M) but hurts dfine-s (16M) and both RT-DETRv2 variants on clean CPPE-5. On noisier datasets (fire), EMA is more beneficial.

**3. Capacity sweep — dfine-n dominates at 850-image scale:**

| Family | Variant | Params | best test_mAP₅₀ |
|---|---|---|---|
| RT-DETRv2 | **R50** | 42M | **0.834** ← best overall |
| RT-DETRv2 | R18 | 20M | 0.741 |
| D-FINE | **n** | **4M** | **0.710** ← best-for-cost |
| D-FINE | s | 16M | 0.603 |
| D-FINE | m | 31M | 0.431 |
| YOLOX (Mosaic, 100 ep) | m | 25M | **0.867** (val, earlier run) |

dfine-s and dfine-m overfit on 850 images — both are worse than 4M dfine-n. Use dfine-n for CPPE-5-scale datasets; dfine-s may recover with more data (~5k+).

### Recommended configs for CPPE-5-scale fine-tuning

- **Best accuracy**: `our_rtdetr_v2_torchvision/06_training_r50_norm_false_ema_false.yaml` (0.834)
- **Best cost/accuracy**: `our_dfine_torchvision/06_training_dfine_n_norm_false_ema_true.yaml` (0.710, 4M params)
- **Fastest**: R18 norm_false ema_false (0.741, ~20M params, 1.5× faster than R50)

## Per-run observability (every run)

Each run dir holds a full 3-axis report — identical layout on both backends. See `core/p06_training/CLAUDE.md` for the tree + config toggles. Expect ~2–3 GB per run (checkpoints + preview PNGs + error_analysis gallery).

Known gotcha: `_finalize_training` on HF backend can create a stray `notebooks/.../runs/` nested dir when `logging.save_dir` is relative and the override path contains `notebooks/`. Safe to `rm -rf our_*/notebooks/` post-hoc — the real artifacts live under `runs/<save_dir>/`.

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

## Data pipeline

Augmentation runs first (on raw pixel values), then the HF `AutoImageProcessor`
resizes → rescales (/255) → optionally normalizes (mean/std). Normalization is
applied **after** augmentation — color-based aug (HSV, brightness/contrast) always
sees raw pixel scale.

- `pixel_values`: (B, 3, 480, 480) float32; rescaled to [0,1] by processor.
  Whether ImageNet mean/std is also applied depends on `tensor_prep.normalize`.
- `labels`: list of `{"class_labels": LongTensor, "boxes": FloatTensor cxcywh [0,1]}`
- `augmentation.normalize: false` in all configs — aug pipeline never applies mean/std.

Aug semantics match qubvel's basic recipe (Perspective 0.1 + HFlip 0.5 +
BrightContrast 0.5 + HueSatVal 0.1 + bbox clip). Our pipeline supports both
albumentations and torchvision v2 via `augmentation.library`.

**normalize=false is the recommended default** — ablation shows it beats normalize=true
on CPPE-5 across all 5 architectures tested (+0.008 to +0.153 mAP₅₀). The reference
notebook uses normalize=true but our results exceed it at normalize=false.

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
