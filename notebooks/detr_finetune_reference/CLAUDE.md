# CLAUDE.md — notebooks/detr_finetune_reference/

Isolated reproduction of qubvel's HF reference notebooks for RT-DETRv2 + D-FINE
fine-tuning, converted to plain `.py` scripts. **Purpose**: known-good baseline
to diff against when our in-repo `core/p06_training/DetectionTrainer` can't
make a DETR-family model converge — we need to attribute the gap to our code
vs the arch.

## Layout

```
.
├── CLAUDE.md                       (this file — Claude-facing notes)
├── README.md                       (human-facing overview + navigation)
├── pyproject.toml                  pinned deps (uv-managed; albumentations==1.4.6, ...)
├── .gitignore                      excludes runs/ and .venv/ everywhere beneath
├── data_loader.py                  shared — HF cppe-5 ↔ YOLO bridge + --dump-cppe5 CLI
│
├── reference_rtdetr_v2/            qubvel's RT-DETRv2 reference
│   ├── finetune.py                 RT-DETRv2-R50 on CPPE-5 (CLI: --seed, --tag, --aug)
│   ├── inference.py                single-image inference
│   ├── RT_DETR_v2_finetune_on_a_custom_dataset.ipynb   (upstream original)
│   ├── RT_DETR_v2_inference.ipynb                       (upstream original)
│   ├── README.md
│   └── runs/                       (gitignored) training outputs
│
├── reference_dfine/                qubvel's D-FINE reference
│   ├── finetune.py                 D-FINE-large on CPPE-5 (lr=2e-5 fix applied)
│   ├── inference.py
│   ├── DFine_finetune_on_a_custom_dataset.ipynb        (upstream original)
│   ├── DFine_inference.ipynb
│   ├── README.md
│   └── runs/                       (gitignored)
│
├── our_rtdetr_v2_albumentations/   OUR pipeline, RT-DETRv2, Albumentations aug — DONE
│   ├── 05_data.yaml / 06_training.yaml / README.md
│   └── runs/                       (gitignored)
│
├── our_rtdetr_v2_torchvision/      OUR pipeline, RT-DETRv2, torchvision v2 aug — PLACEHOLDER
├── our_dfine/                      OUR pipeline, D-FINE, HF backend — PLACEHOLDER
└── our_yolox/                      OUR pipeline, YOLOX-M, pytorch backend — PLACEHOLDER
```

Two-kind folder convention:
- `reference_<arch>/` holds the **upstream baseline** — qubvel's notebook
  as runnable Python, next to its original `.ipynb`, with training outputs
  under the folder's own `runs/`.
- `our_<arch>[_<aug>]/` holds the **same experiment run through our
  pipeline** (`core/p06_training/train.py --backend hf`) for
  apples-to-apples comparison. Each is self-contained: config YAMLs,
  README with setup + expected numbers, and its own `runs/`.

## Venv

Always use `.venv-notebook/` (not the main `.venv/`). Pinned separately via
`notebooks/detr_finetune_reference/pyproject.toml` + `scripts/setup-notebook-venv.sh`
because:
- `albumentations==1.4.6` is qubvel's pin (newer versions deprecate
  `A.BboxParams(min_area=N)` semantics that the notebook relies on).
- HF transformers from git — same as main venv, but decoupled so notebook env
  can be bumped without touching production training.
- `jupyterlab + ipykernel` registered as kernel name `detr-reference`.

Setup: `bash scripts/setup-notebook-venv.sh` runs
`uv sync --project notebooks/detr_finetune_reference` with
`UV_PROJECT_ENVIRONMENT=$REPO_ROOT/.venv-notebook` so the venv lands at repo
root (not inside the project dir), then installs `-e <repo>` with `--no-deps`
so `utils/` imports work from `data_loader.py`.

Invocation pattern (from repo root):
```bash
# Reference (qubvel's pipeline)
.venv-notebook/bin/python notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py --seed 42

# Our in-repo pipeline (same recipe, our trainer)
CUDA_VISIBLE_DEVICES=1 uv run core/p06_training/train.py \
  --config notebooks/detr_finetune_reference/our_rtdetr_v2_albumentations/06_training.yaml
```
Do NOT `uv run` from the repo root — that uses the main `.venv/` and will
fail on the albumentations pin. If running `uv` against the reference project
directly, always set `UV_PROJECT_ENVIRONMENT=$REPO_ROOT/.venv-notebook` so uv
targets the notebook venv and not `notebooks/detr_finetune_reference/.venv`.

## Conversion gotchas (applied to the `.py` files)

All `.py` files are `jupyter nbconvert --to script` output + 3 mechanical fixes.
If the user re-converts from `.ipynb` in the future, re-apply:

1. **Remove shell installs** — `get_ipython().system('pip install …')` and
   `!pip …` lines. Deps come from `requirements.txt`. Keeping these in the
   `.py` would re-install in whichever Python runs the script and break the
   pinned env.

2. **Comment out `display(...)`** — Jupyter-only function. Un-commented will
   raise `NameError` in plain Python. Training/eval behaviour unchanged; these
   are visualization cells only.

3. **`datasets` 4.x access-pattern fix** — qubvel's notebook uses
   ```python
   ds.features["objects"].feature["category"].names     # datasets 2.x
   ```
   which breaks on `datasets>=4.0` (`features["objects"]` is now a plain dict,
   not a `Sequence` with `.feature`). Rewrite to:
   ```python
   ds.features["objects"]["category"].feature.names     # datasets 4.x
   ```
   **Verified**: running `load_dataset("cppe-5")` under datasets 4.8 shows
   `features["objects"]` as a dict → the notebook's original access would fail
   even without our edits. This is an upstream notebook bug, not a
   local-pipeline issue.

## Phase 1: RT-DETRv2 reproduction on CPPE-5 — **DONE**

Qubvel's published test mAP = **0.5789** (single run, no seed variance
reported). Both our reference port (`reference_rtdetr_v2/`) and our
in-repo pipeline (`our_rtdetr_v2_albumentations/`) land within noise of
that under Bundle B hyperparameters.

### Head-to-head, seed=42, Bundle B (2026-04-20, parallel GPUs)

Both runs launched simultaneously on different GPUs of the same box; same
`train_test_split(0.15, seed=1337)` CPPE-5 split; same Bundle B recipe
(bs=16, lr=1e-4, cosine, WD=1e-4, bf16, 40 ep); same determinism setup.
The only difference is the training loop: qubvel's notebook Trainer call
vs `core/p06_training/train.py --backend hf`.

| Axis | qubvel published¹ | `reference_rtdetr_v2/` | `our_rtdetr_v2_albumentations/` | Δ (ours − ref) |
|---|---|---|---|---|
| Pipeline | upstream notebook | our `.py` port of same | `core/p06_training/hf_trainer.py` | |
| GPU | — | 1 | 0 | |
| `train_runtime` | — (not reported) | 617.1 s | 615.1 s | −2.0 s |
| Best val mAP @ ep | — | 0.3655 @ 13 | 0.3510 @ 18 | −0.0145 |
| **Test mAP** | **0.5789** | 0.5464 | **0.5577** | **+0.0113** |
| Test mAP₅₀ | 0.8674 | 0.8043 | 0.8285 | +0.0242 |
| Test mAP₇₅ | 0.6689 | 0.6316 | 0.5847 | −0.0469 |
| Coverall | 0.6130 | 0.6146 | 0.5470 | −0.068 |
| Face_Shield | 0.7165 | 0.6652 | 0.6256 | −0.040 |
| Gloves | 0.5180 | 0.4645 | **0.5346** | **+0.070** |
| Goggles | 0.5202 | 0.4125 | **0.5343** | **+0.122** |
| Mask | 0.5269 | 0.5751 | 0.5471 | −0.028 |

¹ Numbers transcribed from `reference_rtdetr_v2/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb`
(the output cell of the final `pprint(metrics)` after
`trainer.evaluate(test_dataset)`). Qubvel published a single run with no
seed variance info; his recipe was 40 epochs, bs=16, lr=1e-4, cosine,
WD=1e-4, bf16 — i.e. Bundle B minus explicit determinism.

**Bottom line**: the in-repo pipeline matches reference wall clock within
2s (0.3% faster), edges ahead on test mAP by +0.011, and is more balanced
per class — weakest class is 0.53 here vs 0.41 for the reference run.
Goggles (historically our worst class) jumps +0.12, likely a seed-specific
effect from our code-path's slightly different backward-pass numerics
under `use_deterministic_algorithms(warn_only=True)`.

Visual comparison — same `annotate_gt_pred` helper on both sides so
boxes/colours/legend match byte-for-byte:
- Reference grid pair: `reference_rtdetr_v2/runs/rtdetr_v2_r50_cppe5_seed42_bs16_lr1e4_cosine_wd_bf16/{val,test}_predictions/final.png` (from `reference_rtdetr_v2/inference.py`)
- In-repo per-epoch val grids: `our_rtdetr_v2_albumentations/runs/seed42/val_predictions/epoch_NN.png` (from `HFValPredictionCallback` at train time)

### Historical progression (how we arrived at Bundle B)

Qubvel's single-run number is 0.5789. Early in-repo runs landed much
lower because of seed-lottery effects in the class-head reinit. Path:

| Step | Config | test mAP | Gap vs qubvel |
|---|---|---|---|
| naïve port | qubvel's recipe, OS-entropy seeds | 0.5054 | −0.073 |
| +determinism | `set_seed(42)` before `from_pretrained` | 0.5325 | −0.046 |
| +Bundle A | cosine + WD 1e-4 + bf16 | 0.5348 | −0.044 |
| +Bundle B | bs=16, lr=1e-4 (linear scale) | 0.5585 | −0.020 |
| 3-seed mean ± std | seeds 42/0/2024 | 0.5287 ± 0.030 | −0.050 (+1.65σ) |
| `--aug strong` | + BBoxSafeRandomCrop / CLAHE | 0.5420 | **worse** than basic |

Single-seed test-mAP variance is ±0.03 for 29-image test. Qubvel's 0.5789
sits at ~+1.5σ above our mean — reachable as a lucky seed, statistically
not distinguishable from a library regression with this sample size.

### The deterministic recipe — why it took so many tries

Qubvel's notebook runs on default HF Trainer seed=42 which is set **inside**
`Trainer.__init__`. That's *after* `from_pretrained` has already re-init'd
the six decoder class-embed heads + enc_score_head + denoising_class_embed
using whatever OS-entropy torch booted with. So even "same code, same
weights" = different class-head init every fresh process. Our -0.073 gap
on first run was ~60% seed lottery, confirmed by +0.027 mAP just from
calling `set_seed(42)` early.

### Bundle B as the default recipe

These go beyond qubvel's notebook but are standard DETR-family tuning and
apply cleanly to Phase 2 features:

```python
per_device_train_batch_size = 16     # was 8
learning_rate               = 1e-4   # was 5e-5 (linear scaling for 2x batch)
lr_scheduler_type           = "cosine"
weight_decay                = 1e-4
bf16                        = True   # RTX 5090 tensor cores (no quality loss)
seed = data_seed            = 42     # passed into TrainingArguments

# And *before* any from_pretrained():
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Also: Bundle B **finishes in 9m 23s on RTX 5090, 11% faster than qubvel's
bs=8 recipe**.

### D-FINE status — **NOT reproduced**

`reference_dfine/finetune.py` applies the lr=2e-5 fix over qubvel's recipe; val
mAP plateaus at ep3 ≈ 0.20 and never climbs — **the same LR that works for
rtdetr_v2_r50 is too hot for dfine-large's ~3x-larger backbone**. Test
mAP = 0.2617 vs qubvel's 0.4485. Fix (not yet applied):

```python
learning_rate = 2e-5   # was 5e-5, halved-ish for large backbone
warmup_steps  = 500    # was 300, gentler rampup
# + Bundle B's other changes (cosine, WD, bf16, seed)
```

Before using D-FINE as a Phase-2 reference, apply the above and re-run.

### `--aug strong` is opt-in, default stays qubvel-identical

Adding `BBoxSafeRandomCrop(p=0.3) + HueSaturationValue(15,25,15, p=0.3) +
CLAHE(p=0.2)` on top of Bundle B yielded 0.5420 vs basic aug's 0.5585 —
net **negative** on overall mAP, despite +0.005 on Goggles and +0.05 on
Mask. It rebalances classes rather than improving the frontier. Kept
available via `--aug strong` but **not default**.

### Known remaining weak spot: Goggles class

Across all 3 seeds × both aug settings, Goggles AP sits at 0.34–0.42
(std 0.038 — the highest of any class). It's the rarest class in CPPE-5
and the dominant source of our residual gap. If reaching 0.58 test mAP is
a hard requirement, targeted Goggles oversampling or class-weighted loss
would help more than stronger augmentation — but that's scope creep we
chose not to pursue.

## Phase 2: swap CPPE-5 for our features

Only run after Phase 1 succeeds. `data_loader.py::load_feature_dataset(name, subset)`
returns a dict keyed `{"train", "validation", "test"}`, each an HF `Dataset`
with schema byte-compatible with CPPE-5 (COCO bbox format
`[x, y, w, h]`, `ClassLabel` category feature reading class names from the
feature's `05_data.yaml::names`).

One-line swap in each `*_finetune_cppe5.py`:
```python
# Replace:
# from datasets import load_dataset
# dataset = load_dataset("cppe-5")
# if "validation" not in dataset: ...

# With:
from data_loader import load_feature_dataset
dataset = load_feature_dataset("fire_detection", subset=0.05)
```

All downstream code (Albumentations pipeline, `CPPE5Dataset` class,
`AutoImageProcessor`, `AutoModelForObjectDetection.from_pretrained`, collator,
`TrainingArguments`, `MAPEvaluator`) stays verbatim.

### `data_loader.py` contract

- Input: `dataset_store/training_ready/<dataset_name>/{train,val,test}/{images,labels}/`
  in YOLO `.txt` format (one line per box: `<cls> <cx_norm> <cy_norm> <w_norm> <h_norm>`).
- Output HF `Dataset` columns match CPPE-5 exactly:
  - `image_id: int`
  - `image: PIL.Image` (via `datasets.Image()`)
  - `width, height: int`
  - `objects: Sequence({id, area, bbox, category})` — **COCO pixel format**
    `[x_top_left, y_top_left, width, height]`, matches
    `A.BboxParams(format="coco", ...)` used in the reference scripts.
  - `category` is a `ClassLabel` feature so notebook schema introspection works.
- Class names auto-loaded from `features/<feature-dir>/configs/05_data.yaml::names`.
  Feature-dir lookup uses the map in `load_class_names()`; add new features
  there when extending beyond fire_detection / ppe-helmet / etc.

## CLI for `reference_rtdetr_v2/finetune.py`

The script now takes three flags (all optional, override via env var too):

```bash
.venv-notebook/bin/python notebooks/detr_finetune_reference/reference_rtdetr_v2/finetune.py \
    --seed 42                          # SEED env var; default 42
    --tag bs16_lr1e4_cosine_wd_bf16    # RUN_TAG env var; default empty
    --aug basic|strong                 # AUG env var; default "basic" (qubvel)
```

Output dir is derived: `runs/rtdetr_v2_r50_cppe5_seed{SEED}{_TAG}/`. Always
include a `--tag` when running a new config so it doesn't overwrite the
deterministic baseline.

## Per-seed variance is large — compare like-for-like

Our 3-seed Bundle B sweep (seeds 42, 0, 2024) spans **0.4857 – 0.5585** on
test mAP — a 0.073 range on a 29-image test set. **Single-run comparisons
between this reference and our in-repo trainer are not meaningful** unless
both have used the same seed *and* the same determinism setup. When
debugging, either:
- Run both with `seed=42` and verify run-over-run reproducibility on each
  side first (`use_deterministic_algorithms(True, warn_only=True)` +
  `set_seed(42)` early), OR
- Run 3+ seeds on both sides and compare means, not singles.

## Invariants for reference-vs-ours comparison

When running the scripts to debug our pipeline, these MUST stay constant
between reference and our in-repo run for an apples-to-apples comparison:
- Same train/val/test split (same dataset, same subset fraction, same seed).
- Same model checkpoint (qubvel uses `r50vd`; our in-repo typically tests `r18vd` —
  switch one so both match before comparing mAP).
- Same input image size (qubvel: 480; our in-repo: 640 — pick one).
- Same batch size, learning rate, epoch count, warmup steps.
- **Same RNG hygiene**: `set_seed(N)` before any `from_pretrained`, same cuDNN
  deterministic flags. Without this the class-head reinit alone swings mAP
  by 0.03–0.07.

When swapping `checkpoint` / `image_size` in these scripts, update both
`AutoImageProcessor.from_pretrained` and `AutoModelForObjectDetection.from_pretrained`
calls — they live in separate cells/sections.

## Known gotchas

- **Do not run these scripts from the main venv** — albumentations 1.4.6 is
  the pin; the main `.venv/` has a newer version with different box-clip
  semantics. Always invoke via `.venv-notebook/bin/python`.
- **`metric_for_best_model="eval_map"`** in `TrainingArguments` — requires
  the `MAPEvaluator` output to contain a key literally named `"map"`. If that
  key is renamed in torchmetrics, the HF Trainer will silently fail to
  checkpoint the best model. Check `torchmetrics.detection.MeanAveragePrecision`
  output keys match what `MAPEvaluator.__call__` returns under the version in
  use.
- **D-FINE notebook uses `ustc-community/dfine-large-coco`** (large, slower).
  If iterating locally, drop to `dfine-small-coco` or `dfine-medium-obj365`
  to speed up smoke tests. **And** lower the LR — see "D-FINE status" above.
- **Trailing inference block would crash before the fix** — qubvel's notebook
  ends with `model_repo = "<your-name-on-hf>/...` (a placeholder). After
  `trainer.evaluate(test_dataset)` the scripts now call
  `trainer.save_model(_RUN_DIR / "best")` and point `model_repo` at that
  local path, so the inference cells run end-to-end without HF Hub creds.
- **`torch.use_deterministic_algorithms(True)` strict mode crashes RT-DETRv2**:
  the multi-scale deformable attention backward (`grid_sampler_2d_backward_cuda`)
  and memory-efficient attention backward don't have deterministic kernels.
  We use `warn_only=True` — last-digit of mAP still varies run-to-run from
  these ~2 ops, but the rest of the graph is locked.
- **`data_loader.py` stores image paths, not bytes** — the HF Dataset holds
  string paths; the `datasets.Image()` feature lazy-loads via PIL. Safe for
  local training but breaks if the dataset is moved off-machine. For upload
  to HF Hub, `ds = ds.cast_column("image", HFImage(decode=True))` to
  materialize bytes first.
