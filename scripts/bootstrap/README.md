# Bootstrap — fetch public pretrained weights + open datasets

## Purpose

On a freshly-provisioned workstation (e.g. `ssh-sg4`) these two scripts
re-materialise everything under `ai/pretrained/` and `ai/dataset_store/`
that is publicly available from its original source (HF Hub, GitHub,
Kaggle, Roboflow, Google Cloud Storage, etc.), so the operator does not
have to push ~135 GB through our Cloudflare tunnel. Only the internal,
non-reproducible artefacts (Nitto-Denko footage, sibling-project
checkpoints) still need an out-of-band `rsync`.

## Prereqs

Fresh Ubuntu 24.04 only needs `bash`, `curl`, `python3`, `unzip`.
Install the rest per-user:

```bash
# Hugging Face CLI (new `hf` replaces deprecated `huggingface-cli`)
pip install --user -U "huggingface_hub[cli]"

# Kaggle CLI (datasets manifest uses it)
pip install --user kaggle

# gdown (only needed if you pull a Google Drive entry; scripts install it on demand)
pip install --user gdown
```

Create `$PROJECT_ROOT/.env` (never committed, never echoed):

```bash
HF_TOKEN=hf_********
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
ROBOFLOW_API_KEY=rf_********
```

Alternatively Kaggle accepts `~/.kaggle/kaggle.json` with `{"username": "...", "key": "..."}` (chmod 600).

## Gated HF repos (one-time Agree on model card)

Source: `ai/docs/technical_study/gated-retry-download-log.md`. Using the
`HF_TOKEN` in `.env`, visit each URL with the matching account and click
"Agree and access repository":

- https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- https://huggingface.co/facebook/sapiens-pose-0.3b
- https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript
- https://huggingface.co/Advantech-EIOT/qualcomm-ultralytics-ppe_detection
- https://huggingface.co/idiap/EdgeFace

Afterwards the scripts download them automatically.

## Usage

```bash
# One-time: place the .env at project root.
cp /secure/path/edge_ai.env "$PROJECT_ROOT/.env"

# All pretrained weights (~29 GB public; skips anything already on disk)
bash ai/scripts/bootstrap/download_pretrained.sh

# Subset by feature folder name
bash ai/scripts/bootstrap/download_pretrained.sh --only safety-fire_detection

# Dry run (plan only)
bash ai/scripts/bootstrap/download_pretrained.sh --dry-run

# Datasets (~106 GB public)
bash ai/scripts/bootstrap/download_datasets.sh --only safety-fire_detection
```

Both scripts are idempotent — re-running is cheap when everything is
already there. Per-row status goes to
`ai/scripts/bootstrap/bootstrap.log` (tab-separated).

## Estimated disk + time

| Bucket                       | Size   | Notes |
|------------------------------|-------:|-------|
| Pretrained weights (public)  | ~29 GB | +~4 GB if all gated Sapiens/DINOv3 accepted |
| Open datasets (public)       | ~106 GB | dominated by FASDD_CV (~30 GB) + COCO-2017-keypoints (~18 GB) |
| Internal (rsync_only)        | ~? GB | sibling-project symlinks + site-collected + training_ready |

Wall time: plan on 1.5–3 h over a 100 Mbps link; most of it is the
Kaggle datasets.

## Non-reproducible entries (rsync_only)

Listed at the bottom of each manifest under `# non-reproducible` /
`# Internal / non-reproducible` headers. They are **not** downloaded.
After the public pulls succeed, rsync them out-of-band, e.g.:

```bash
# From the workstation that already has them (typical: the dev laptop)
rsync -aHP --info=progress2 \
  ~/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/dataset_store/site_collected/ \
  ssh-sg4:/home/you/edge_ai/ai/dataset_store/site_collected/

rsync -aHP --info=progress2 \
  ~/Documents/05_Team/02_Vietsol/01_Projects/visual_core/01_code/checkpoints/ \
  ssh-sg4:/home/you/visual_core/01_code/checkpoints/

rsync -aHP --info=progress2 \
  ~/Documents/05_Team/02_Vietsol/01_Projects/dms_oms/pretrained/ \
  ssh-sg4:/home/you/dms_oms/pretrained/
```

The two sibling projects live *outside* `edge_ai/`; the symlinks under
`ai/pretrained/access-*` and `ai/pretrained/safety-fall_classification/`
point into those trees.

## Troubleshooting

- **Google Drive virus-scan gate.** `gdown --fuzzy` fails on large
  assets if Google shows a confirmation interstitial. Open the URL in a
  browser, accept, then re-run — `gdown` picks up the cookie.
- **HF gated terms (401).** Accept on the model card (see Gated list
  above); re-run. Verify `HF_TOKEN` is set: `hf whoami`.
- **Kaggle CLI token.** Either set `KAGGLE_USERNAME`/`KAGGLE_KEY` in
  `.env` or place `~/.kaggle/kaggle.json` (chmod 600).
- **Roboflow URL format.** The scripts append
  `?key=$ROBOFLOW_API_KEY&format=yolov8` (or `&...` if the URL already
  has a query). For a Roboflow Universe project, use the *Export* URL
  under `Versions → Download → raw URL`; it looks like
  `https://universe.roboflow.com/ds/<slug>`. If the project requires
  `format=coco`, edit the manifest row's URL accordingly.
- **SHA mismatch.** Scripts retry once then log FAIL — inspect
  `bootstrap.log` tab-separated; the `dest` column tells you which
  file to inspect.
- **Ctrl-C mid-download.** Files land with a `.partial` suffix (for
  curl paths); re-running resumes via `curl -C -`.
