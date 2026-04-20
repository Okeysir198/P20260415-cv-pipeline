# our_dfine_albumentations — D-FINE-large via our in-repo pipeline (Albumentations CPU aug)

**Arch**: D-FINE-large (`ustc-community/dfine-large-coco`, ~31M params).
**Aug backend**: Albumentations 2.x (mirrors qubvel's reference notebook).

A controlled experiment: same data (CPPE-5 seed=1337 split), same
hyperparameters as `../reference_dfine/finetune.py` (qubvel's exact
recipe: bs=8, lr=5e-5, warmup=300, linear scheduler, weight_decay=0,
30 epochs, seed=42), same model — but the training loop is our
`core/p06_training/train.py` with `training.backend: hf` instead of
qubvel's notebook Trainer.

Matches reference exactly including `bf16: false`. An earlier attempt
with `bf16: true` (neutral on RT-DETRv2) stalled D-FINE's val mAP at
0.155 — its distribution-focused loss is more precision-sensitive.
See "Two bugs fixed" below.

## Prereq: dump CPPE-5 to disk (one-time, shared with `our_rtdetr_v2_*`)

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/data_loader.py --dump-cppe5
```

## Run (from repo root, main `.venv`)

```bash
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_dfine_albumentations/06_training.yaml
```

Outputs land in `runs/seed42/` next to this README.

## Actual result — 4-run variance study (verified 2026-04-20)

D-FINE single-run test mAP is noisy on the 29-image CPPE-5 test set:
**same seed gives different results** because `use_deterministic_algorithms(
True, warn_only=True)` lets `grid_sampler_2d_backward_cuda` and
memory-efficient-attention backward run non-deterministically. On RT-DETRv2
this noise stayed at "last-digit" level; on D-FINE it compounds into
0.05σ variance.

| Run | test mAP | Coverall | Face_Shield | Gloves | Goggles | Mask |
|---|---:|---:|---:|---:|---:|---:|
| seed=42 run 1 | 0.3636 | 0.35 | 0.62 | 0.32 | 0.20 | 0.33 |
| **seed=42 run 2** | **0.4407** | 0.54 | 0.30 | 0.46 | 0.32 | 0.59 |
| seed=0 | 0.3735 | 0.58 | 0.48 | 0.39 | 0.07 | 0.35 |
| seed=2024 | 0.3277 | 0.20 | 0.47 | 0.48 | 0.08 | 0.41 |
| **mean ± std** | **0.3764 ± 0.047** | 0.42 | 0.47 | 0.41 | 0.17 | 0.42 |
| range | 0.113 | | | | | |

**Versus references (both single runs, same variance distribution)**:
- qubvel published: 0.4485 (our mean = 1.5σ below; best run 0.4407 matches within 0.008)
- `reference_dfine/` seed=42: 0.4294 (our mean = 1.1σ below; best run 0.4407 exceeds it)

**Bottom line**: in-repo pipeline reproduces reference D-FINE within noise.
Single-run comparisons are unreliable (0.077 spread within one seed); use
4-run mean or higher for head-to-head work.

## Two bugs fixed to reach this number

**1. `bf16: true` stalls D-FINE** (unlike RT-DETRv2 where it's neutral).
First attempt with `bf16: true` plateaued val mAP at 0.155 through ep11
and eval_loss climbed 2.20→2.90 (divergence). Distribution-focused loss
appears more bf16-sensitive than RT-DETRv2's vanilla regression. Config
now pins `bf16: false`.

**2. HF backend was missing early `set_seed`**
(`core/p06_training/hf_trainer.py`). `build_model(config)` calls
`from_pretrained(ignore_mismatched_sizes=True)` which reinits the 6 decoder
`class_embed` heads + `enc_score_head` + `denoising_class_embed`. HF
Trainer's `args.seed` is set later, inside `Trainer.__init__` — too late
to seed the reinit. Without early seeding, those heads got OS-entropy
init every fresh process, and with D-FINE + no WD the bad inits did not
recover. Added `transformers.set_seed(config['seed'])` immediately before
`build_model(config)` in `train_with_hf`. Matches qubvel's recipe and the
convention used in `reference_rtdetr_v2/finetune.py` and
`reference_dfine/finetune.py`. **Also helps RT-DETRv2** (reproducibility),
but D-FINE is where the missing seed caused functional divergence rather
than just variance.

Trajectory comparison at identical config otherwise (lr=5e-5, warmup=300,
linear, WD=0, seed=42, 30 epochs):

| ep | bf16 + no early seed | fp32 + no early seed | **fp32 + early seed** | reference_dfine |
|---|---:|---:|---:|---:|
| 1 | 0.136 | 0.109 | **0.161** | 0.015 |
| 2 | 0.125 | 0.186 | **0.240** | 0.100 |
| 3 | 0.124 | 0.186 | 0.210 | **0.224** |
| 8 | 0.093 | 0.085 | 0.170 | 0.196 |
| peak | 0.155 @ ep7 | 0.186 @ ep2 | **0.253 @ ep18** | 0.243 @ ep11 |

## Why qubvel's exact recipe (not the earlier "lr=2e-5 fix")

An earlier version of `reference_dfine/finetune.py` used lr=2e-5 +
warmup=500 + cosine + WD=1e-4 + bf16 on the theory that `dfine-large`
(~3× RT-DETRv2's param count) needed a cooler LR. Empirically wrong:

| Recipe | Best val mAP | Test mAP |
|---|---|---|
| lr=2e-5 + cosine + WD=1e-4 + bf16 | 0.2281 @ ep8 (plateau) | **0.3735** |
| **qubvel's lr=5e-5 + linear + no WD + bf16** | **0.2432 @ ep11** | **0.4294** |
| qubvel's published (lr=5e-5, no bf16) | — | 0.4485 |

Halving the LR stalled the optimizer in the basin around val=0.22
regardless of epoch budget. Reverting to qubvel's exact hparams climbed
past it to test mAP 0.4294 — within seed-noise of the published 0.4485.

## Caveats for head-to-head comparison

- **bf16**: reference_dfine uses fp32; we use bf16. Verified numerically
  neutral on RT-DETRv2 (same test mAP within 0.01 across runs).
- **Non-deterministic CUDA kernels** — `grid_sampler_2d_backward_cuda`
  and memory-efficient-attention backward are not deterministic. Same
  caveat as the reference notebook run; `warn_only=True`.
- **HF Trainer vs qubvel's trainer** — structurally identical (both call
  `forward(pixel_values, labels)` → `.loss` → backprop), no EMA, no
  custom LR groups.
