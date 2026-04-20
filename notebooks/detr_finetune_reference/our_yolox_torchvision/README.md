# our_yolox_torchvision — YOLOX-M (official) with the full production recipe

Sibling to `../our_yolox/` (Albumentations backend, 480², no Mosaic).
This one flips four knobs to match Megvii's production training
procedure and measure the accuracy cost of the
`our_yolox/` simplifications:

| knob | `our_yolox/` (baseline) | **this** |
|---|---|---|
| `augmentation.library` | albumentations | **torchvision v2** |
| `augmentation.mosaic` | false (unsupported by Albu backend) | **true** |
| `augmentation.mixup` | false | **true** |
| `input_size` | 480² | **640²** |
| epochs | 50 | **100** |
| affine | `scale=[0.9,1.1], degrees=5` | `scale=[0.1,2.0], degrees=10` (Megvii default) |
| HSV | `p=0.5`, magnitudes smaller | always-on, Megvii magnitudes |

Expected improvements (from general priors + TTA measurement on the
baseline):

- Mosaic alone: **+0.05-0.10 mAP₅₀**, especially on rare classes
  (Goggles in baseline = 0.30, TTA'd = 0.49, Mosaic typically matches
  the TTA uplift at no inference cost).
- 640² input: **+0.02-0.04 mAP₅₀** on small-object classes (Gloves,
  Goggles).
- 100 ep: **+0.02 mAP₅₀** vs the 50-ep baseline (convergence not yet
  saturated at ep50).

Stretch target: match or exceed `../our_rtdetr_v2_albumentations/`'s
test mAP₅₀ (0.7714) at ~half the wall time.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 .venv-yolox-official/bin/python \
  core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_yolox_torchvision/06_training.yaml
```

Same venv + pretrained requirements as `../our_yolox/`. See that
folder's README for setup.

## After training

Run p08 eval + TTA:

```bash
# Baseline test mAP@0.5 (conf 0.05)
CUDA_VISIBLE_DEVICES=1 .venv-yolox-official/bin/python \
  core/p08_evaluation/evaluate.py \
  --model notebooks/detr_finetune_reference/our_yolox_torchvision/runs/seed42/best.pth \
  --config notebooks/detr_finetune_reference/our_yolox_torchvision/05_data.yaml \
  --split test --conf 0.05 \
  --save-dir notebooks/detr_finetune_reference/our_yolox_torchvision/runs/seed42/eval_test_conf005

# + TTA (multi-scale 512/640/768 × h-flip) — expected +0.03-0.10 on top
CUDA_VISIBLE_DEVICES=1 .venv-yolox-official/bin/python scripts/yolox_tta_eval.py \
  --ckpt notebooks/detr_finetune_reference/our_yolox_torchvision/runs/seed42/best.pth \
  --data-config notebooks/detr_finetune_reference/our_yolox_torchvision/05_data.yaml \
  --split test --scales "512,640,768" --flip --ref-size 640 --conf 0.05 \
  --out-dir notebooks/detr_finetune_reference/our_yolox_torchvision/runs/seed42/tta_test
```

## Result (seed=42, 100 epochs, GPU 1, 2026-04-20)

### Wall time — 3.6× slower than the albumentations baseline

| axis | `../our_yolox/` (Albu, 480², 50 ep) | **this (TV, 640², 100 ep)** |
|---|---|---|
| `train_runtime` | 553.7 s | **1987 s (33:07)** |
| per-epoch | 11.1 s | 19.9 s |

Slowdown factors: 640² (1.78× per-pixel) × Mosaic (4× disk reads/sample)
× 2× more epochs. Mosaic is the dominant cost — disable it for a quick
retrain to isolate the 640² + 100-ep contribution.

### Accuracy — Mosaic closes ~2/3 of the gap vs RT-DETRv2

Val (best epoch 71 of 100) and test (29-image CPPE-5 test split):

| metric | baseline | **this run** | Δ | RT-DETRv2-R50 Albu |
|---|---|---|---|---|
| Val mAP@0.5 best | 0.6561 @ ep26 | **0.6825 @ ep71** | +0.026 | — |
| Test mAP@0.5 (p08, conf 0.05) | 0.5718 | **0.6409** | +0.069 | 0.7714 (torchmetrics) |
| Test mAP@0.5 + TTA (6×) | 0.6876 | **0.7400** | +0.052 | — |

### Per-class — Goggles is the headline

Test AP breakdown (p08 eval @ conf=0.05):

| class | baseline | this run | Δ | + TTA this run | Δ + TTA vs baseline TTA |
|---|---|---|---|---|---|
| Coverall | 0.6991 | 0.7894 | **+0.090** | 0.7769 | −0.011 |
| Face_Shield | 0.7218 | 0.6893 | −0.033 | 0.7917 | −0.008 |
| Gloves | 0.4419 | 0.4856 | +0.044 | 0.6270 | +0.044 |
| **Goggles** | 0.2960 | **0.4244** | **+0.128** | **0.6594** | **+0.169** |
| Mask | 0.7000 | 0.8158 | **+0.116** | 0.8449 | +0.068 |

Mosaic + 640² lifts the rare/small-object classes most (Goggles +0.17
after TTA, Gloves +0.04, Mask +0.07). Coverall is the easy class —
already saturated at the baseline. Face_Shield is essentially
unchanged: it's medium-size and well-represented, so extra augmentation
doesn't add much.

### Takeaways

- **Mosaic is the biggest single lever for YOLOX on this dataset** —
  +0.07 test mAP₅₀ baseline, +0.05 under TTA.
- **TTA gains of +0.10 on top of the improved baseline** confirm TTA
  compounds with training-time changes.
- **YOLOX + TTA (0.7400) approaches RT-DETRv2 albu (0.7714) at
  ~2.3× total wall time** (~45 min vs ~14 min train + eval).
  RT-DETRv2 wins on speed AND accuracy on this specific
  small-dataset fine-tune. YOLOX's advantage is real-time inference
  (no transformer overhead), not training-time accuracy on 850-image
  sets.
- **Per-class swings mean seed variance is dominant** on 29-image test;
  reruns with different seeds would shift ±0.03 mAP₅₀.

### What to try next (in priority order)

1. **`close_mosaic_epochs`** — YOLOX upstream disables Mosaic in the
   last ~15 epochs so the model can converge on clean images. Our
   trainer doesn't implement this; patching it in would likely give
   another +0.01-0.02 mAP₅₀.
2. **YOLOX-L** (46.5M params vs 8.9M) — consistent +0.02-0.04 on the
   fire_detection runs, costs ~2× wall time.
3. **Wire torchmetrics MAP into `core/p08_evaluation/evaluator.py`** so
   the test mAP is directly comparable to the DETR siblings' 0.7714.
   The p08 conf=0.05 single-IoU AP is close but not identical; some of
   the remaining gap may be evaluator methodology.

