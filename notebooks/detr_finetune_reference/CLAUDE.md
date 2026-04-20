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
├── README.md                       (human-facing setup + usage)
├── pyproject.toml                  pinned deps (uv-managed; albumentations==1.4.6, ...)
├── rtdetr_v2_finetune_cppe5.py     RT-DETRv2 fine-tune on CPPE-5 (runnable)
├── dfine_finetune_cppe5.py         D-FINE fine-tune on CPPE-5 (runnable)
├── rtdetr_v2_inference.py          RT-DETRv2 single-image inference
├── dfine_inference.py              D-FINE single-image inference
├── data_loader.py                  YOLO → HF Dataset bridge (for Phase 2 —
│                                   swapping CPPE-5 for our features)
└── reference/                      Untouched original .ipynb files
```

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
.venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py
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

## Phase 1: replicate on CPPE-5 (baseline)

Goal: confirm the reference training code works end-to-end in our
`.venv-notebook`. Run on CPPE-5 (~1k images, 5 classes), compare final mAP
against qubvel's published numbers:

| Script | Epochs | Expected val mAP50 |
|---|---|---|
| `rtdetr_v2_finetune_cppe5.py` | 40 | ≈ 0.34 |
| `dfine_finetune_cppe5.py` | 30 | ≈ 0.33 |

If mAP matches (± 0.05) → reference env is healthy, unblock Phase 2.
If mAP is way off → fix env before attributing anything to our in-repo code.

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

## Invariants for reference-vs-ours comparison

When running the scripts to debug our pipeline, these MUST stay constant
between reference and our in-repo run for an apples-to-apples comparison:
- Same train/val/test split (same dataset, same subset fraction, same seed).
- Same model checkpoint (qubvel uses `r50vd`; our in-repo typically tests `r18vd` —
  switch one so both match before comparing mAP).
- Same input image size (qubvel: 480; our in-repo: 640 — pick one).
- Same batch size, learning rate, epoch count, warmup steps.

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
  to speed up smoke tests.
- **`data_loader.py` stores image paths, not bytes** — the HF Dataset holds
  string paths; the `datasets.Image()` feature lazy-loads via PIL. Safe for
  local training but breaks if the dataset is moved off-machine. For upload
  to HF Hub, `ds = ds.cast_column("image", HFImage(decode=True))` to
  materialize bytes first.
