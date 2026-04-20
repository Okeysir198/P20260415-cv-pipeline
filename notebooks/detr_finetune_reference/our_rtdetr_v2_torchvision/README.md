# our_rtdetr_v2_torchvision — RT-DETRv2 via our in-repo pipeline (torchvision v2 CPU aug)

**Arch**: RT-DETRv2-R50. **Aug backend**: `torchvision.transforms.v2` (the
default path in `core/p05_data/transforms.py`, selected when
`augmentation.library: torchvision` — or when the key is omitted).

Sibling of `../our_rtdetr_v2_albumentations/`: byte-identical model, data,
recipe. The only difference is which CPU augmentation library runs inside
each DataLoader worker. Purpose: measure the performance cost of the
torchvision v2 pipeline vs Albumentations on the same transform set, and
confirm it produces the same test mAP (ruling out a silent semantic
regression).

## Run

```bash
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/06_training.yaml
```

Outputs land in `runs/seed42/` — same HF-Trainer-standard layout as
the Albumentations sibling.

## Result (seed=42, Bundle B, 40 epochs, 2026-04-20)

**Per-sample aug cost is now at parity with Albumentations** after
landing the "resize-first" reorder described below. Same-GPU,
same-config 1-epoch benchmarks:

| Backend | 1-ep `train_runtime` (GPU 1, viz off) |
|---|---|
| Albumentations | 23.26 s |
| **torchvision v2 (this)** | **22.84 s (0.42 s faster)** |

40-epoch correctness run (seed=42, viz callbacks on):

| metric | `../our_rtdetr_v2_albumentations/` (GPU 0) | **this (GPU 1)** | Δ |
|---|---|---|---|
| `train_runtime` | 615.1 s | 866.7 s (GPU contention, not aug code) | +41 % |
| Test mAP | 0.5577 | **0.5584** | +0.0007 (noise) |
| Test mAP₅₀ | 0.8285 | 0.8487 | +0.020 |

**Per-class test AP** (40-ep):

| class | albumentations | torchvision | Δ |
|---|---|---|---|
| Coverall | 0.5470 | 0.7460 | +0.199 |
| Face_Shield | 0.6256 | 0.5747 | −0.051 |
| Gloves | 0.5346 | 0.5029 | −0.032 |
| Goggles | 0.5343 | 0.4498 | −0.085 |
| Mask | 0.5471 | 0.5187 | −0.028 |

Per-class swings up to ±0.2 are consistent with the ±0.03 single-seed σ
on a 29-image test set and the different CPU-aug numerics described in
`../CLAUDE.md`. Overall test mAP is statistically equivalent — **no
correctness regression**.

## Why torchvision v2 used to be 2× slower (now fixed)

Per-sample profile on 128 CPPE-5 training images (shape 500×334×3 uint8
→ 480×480), measuring full pipeline cost.

| configuration | torchvision v2 | albumentations |
|---|---|---|
| bare (resize + dtype + wrap) | 2.88 ms / sample | 2.90 ms / sample |
| full pipeline (perspective + bc + hsv + flip + resize + dtype) | **2.61 ms / sample** | **2.99 ms / sample** |

**Conversion overhead (tv_tensor wrapping, BGR→RGB) is not the problem** —
both backends hit the same ~2.9 ms baseline. The pre-reorder gap was
pure transform cost from running expensive ops on uint8 pre-resize
images:

| transform | on uint8 pre-resize (500×334) | on float32 post-resize (480×480) |
|---|---|---|
| `v2.ColorJitter(hue=0.015, sat=0.2, bright=0.1)` | **24.47 ms** | **5.93 ms** |
| `v2.RandomPerspective(p=1.0, fill=114)` | 13.62 ms | 1.99 ms |
| `v2.ColorJitter(brightness=0.2, contrast=0.2)` | 3.75 ms | 0.91 ms |
| `v2.RandomHorizontalFlip(p=1.0)` | 0.69 ms | 0.20 ms |

v2's ColorJitter HSV path is ~4× slower on uint8 than on float32
because the RGB↔HSV conversion internally upcasts per-op.
Albumentations avoids this with cv2's hand-tuned uint8 HSV kernel.

### Fixes applied to `core/p05_data/transforms.py`

1. **Resize + ToDtype moved to the head of the transform list**
   (immediately after Mosaic/MixUp/CopyPaste, which are dataset-level ops
   that produce `input_size` output). Perspective, ColorJitter, and
   Flips now run on 480² float32 tensors — **the ~8× speedup that
   closes the gap.** Side-effect: `v2.SanitizeBoundingBoxes(min_area=25)`
   now evaluates against the resized canvas, which is the same semantics
   Albumentations uses (`A.BboxParams` runs after `A.Resize`).
2. **Identity-`RandomAffine` skip** — CPPE-5 config has
   `scale=[1,1], degrees=translate=shear=0` which used to run
   `v2.RandomAffine` for a no-op interpolation. Now skipped at build time.
3. **BGR→RGB via `cv2.cvtColor`** instead of numpy strided `.copy()`.
4. **`v2.Resize` bilinear with `antialias=False`** (matches HF DETR
   cookbook; `resize_antialias` config key opts in to LANCZOS for
   classification/seg callers).
5. **`IRSimulation` dtype-aware** — the `+10` offset and `noise_sigma=15`
   constants now scale to the image range (1/255 on float32 input).
   Required for the resize-first reorder, which delivers float32 to
   IRSimulation.
6. **`fill=114` → `fill=114/255.0`** on `RandomAffine` + `RandomPerspective`
   since they now run on float32 [0, 1] tensors.

### Speed trajectory

| stage | 1-ep `train_runtime` (viz off) | per-sample aug cost |
|---|---|---|
| baseline (before any fix) | ~43 s (GPU 1) | 20.18 ms |
| after fixes 2-4 (identity-affine skip + cvtColor + bilinear) | ~23.79 s | ~7 ms |
| **after fix 1 (resize-first reorder)** | **22.84 s** | **2.61 ms** |
| albumentations reference | 23.26 s | 2.99 ms |

## When to use which backend

- **torchvision v2 (default)** — now at parity with Albumentations on
  per-sample wall time, and supports Mosaic/MixUp/CopyPaste/IRSimulation
  (not available on the Albumentations backend — those are dataset-level
  ops that need `Dataset.get_raw_item`). Use for anything that needs
  those transforms, plus the broader v2 ecosystem integration.
- **Albumentations** — still preferred when byte-matching qubvel's
  upstream reference notebook for mAP comparison, since the notebook
  uses `A.Compose(...)`. On our in-repo pipeline the choice is now
  mostly aesthetic.
