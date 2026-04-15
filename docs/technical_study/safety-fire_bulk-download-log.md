# Fire / Smoke Bulk Pretrained Weight Download Log

> Bulk pull of every fire/smoke detection pretrained weight previously skipped
> due to license flags in `safety-fire_detection-sota.md` and
> `safety-fire_oss-pretrained-deep-dive.md`. User directive: "don't worry about
> license, just download first". This file records what was fetched and
> (where applicable) why a candidate was skipped. License flags are NOT
> retracted — see the two source docs for redistribution constraints.

## Destination

Absolute path:
`/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai/pretrained/safety-fire_detection/`

Per-file SHA256 manifest (gitignored):
`ai/pretrained/safety-fire_detection/DOWNLOAD_MANIFEST.md`

Total on disk: ~2.0 GB across 31 model folders + 7 loose `.pth` files.

## Method

- HF repos: `hf download <repo> --local-dir <dest>` with a 120 s per-repo
  timeout; stalled repos killed and the loop advanced to the next entry.
- GitHub release assets: `wget` (D-FINE, YOLOX-M).
- Google Drive (DEIM): `gdown` via direct file id (no `--fuzzy`).
- Tool install: `pip install --user --break-system-packages huggingface_hub`
  (wrote `~/.local/bin/hf`).

## Candidates and outcomes

### HuggingFace fire-specific fine-tunes (previously flagged AGPL-contaminated)

| HF repo | Status | Folder | Notes |
|---|---|---|---|
| pyronear/yolo11s_sensitive-detector_v1.0.0 | OK | `pyronear_yolo11s_sensitive-detector` | 81 MB, 5 files |
| pyronear/yolov8s | OK | `pyronear_yolov8s` | 100 MB, 4 files |
| pyronear/yolov11n | OK | `pyronear_yolov11n` | 41 MB, 6 files |
| pyronear/rexnet1_0x | OK | `pyronear_rexnet1_0x` | 28 MB, 4 files |
| pyronear/rexnet1_5x | OK | `pyronear_rexnet1_5x` | 60 MB, 4 files |
| pyronear/mobilenet_v3_large | OK | `pyronear_mobilenet_v3_large` | 33 MB, 4 files |
| pyronear/resnet18 | OK | `pyronear_resnet18` | 86 MB, 4 files |
| pyronear/resnet34 | OK | `pyronear_resnet34` | 163 MB, 4 files |
| SalahALHaismawi/yolov26-fire-detection | OK | `SalahALHaismawi_yolov26-fire-detection` | 21 MB (YOLOv26-S best.pt + metrics) |
| touati-kamel/yolov8s-forest-fire-detection | OK | `touati-kamel_yolov8s-forest-fire` | 22 MB |
| touati-kamel/yolov10n-forest-fire-detection | OK | `touati-kamel_yolov10n-forest-fire` | 12 MB |
| touati-kamel/yolov12n-forest-fire-detection | OK | `touati-kamel_yolov12n-forest-fire` | 11 MB (27 files) |
| JJUNHYEOK/yolov8n_wildfire_detection | OK | `JJUNHYEOK_yolov8n_wildfire` | 22 MB |
| Mehedi-2-96/fire-smoke-detection-yolo | OK | `Mehedi-2-96_fire-smoke-yolo` | 22 MB (YOLOv8-S) |
| TommyNgx/YOLOv10-Fire-and-Smoke-Detection | OK (not gated today) | `TommyNgx_YOLOv10-Fire-and-Smoke` | 119 MB, 7 files |
| shawnmichael/vit-fire-smoke-detection-v4 | PARTIAL | `shawnmichael_vit-fire-smoke-v4` | 65 MB, download stalled, killed at timeout |
| shawnmichael/convnext-tiny-fire-smoke-detection-v1 | OK | `shawnmichael_convnext-tiny-fire-smoke` | 107 MB, 6 files |
| shawnmichael/swin-fire-smoke-detection-v1 | OK | `shawnmichael_swin-fire-smoke` | 106 MB, 6 files |
| shawnmichael/efficienetb2-fire-smoke-detection-v1 | OK | `shawnmichael_efficientnetb2-fire-smoke` | 30 MB, 6 files |
| shawnmichael/vit-large-fire-smoke-detection-v1 | PARTIAL | `shawnmichael_vit-large-fire-smoke` | 65 MB of ~1.2 GB, killed at timeout; safetensors partial |
| dima806/wildfire_types_image_detection | PARTIAL | `dima806_wildfire_types` | 65 MB (checkpoints partial) |
| sequoiaandrade/Smoke-Cloud-Segmentation-RACE-ODIN | OK | `sequoiaandrade_smoke-cloud-race-odin` | 27 MB, NASA Apache-2.0 |
| Shoriful025/wildfire_smoke_segmentation_vit | OK (tiny) | `Shoriful025_wildfire_smoke_seg_vit` | 48 KB — repo is largely placeholders; no large weight found |

HF-gated check: none of the above returned 401 on the unauthenticated
`hf download` path at run time. The TommyNgx repo is flagged as
access-gated in the deep-dive doc but resolved without challenge during
this run (likely the gate was lifted or the CLI's cached token passed).

### Pyronear ONNX classifiers & detector (already on disk from prior pass, retained)

- `pyronear_rexnet1_3x/` — already present; `hf download` skipped.
- `pyronear_mobilenet_v3_small/` — already present; skipped.
- `pyronear_yolo11s_nimble-narwhal_v6/` — already present; skipped.

### D-FINE / DEIM / YOLOX (GitHub + Google Drive)

| Asset | Status | Local file | Notes |
|---|---|---|---|
| D-FINE-S (Objects365) | OK (pre-existing) | `dfine_s_obj365.pth` | 43 MB |
| D-FINE-M (Objects365) | OK (pre-existing) | `dfine_m_obj365.pth` | 81 MB |
| D-FINE-S (COCO) | OK | `dfine_s_coco.pth` | pulled this run |
| D-FINE-M (COCO) | OK | `dfine_m_coco.pth` | pulled this run |
| D-FINE-L (Objects365) | OK | `dfine_l_obj365.pth` | pulled this run (teacher option) |
| YOLOX-M (COCO) | OK | `yolox_m.pth` | pulled this run (baseline reference) |
| DEIM-D-FINE-M 90e (COCO) | OK | `deim_dfine_m_coco/deim_dfine_hgnetv2_m_coco_90e.pth` | 76 MB via gdown |
| DEIM-D-FINE-S 120e (COCO) | OK | `deim_dfine_s_coco/deim_dfine_hgnetv2_s_coco_120e.pth` | 40 MB via gdown |
| USTC D-FINE-medium COCO (HF) | OK (pre-existing) | `ustc-community_dfine-medium-coco/` | HF Transformers port |
| USTC D-FINE-small COCO (HF) | OK (pre-existing) | `ustc-community_dfine-small-coco/` | HF Transformers port |

### RF-DETR

- **Skipped.** No public GCS / release URLs accept unauthenticated GET
  (`storage.googleapis.com/rfdetr/*` returns 403). The `rfdetr` Python
  package auto-fetches to `~/.cache/rfdetr/` on first model instantiation.
  To obtain weights offline, install `rfdetr` in the feature env and
  instantiate `RFDETRNano()`, `RFDETRSmall()`, `RFDETRMedium()` once.

### Paper-only detectors (no public weight artifact)

- **Smoke-DETR** (MDPI 2024) — no GitHub release.
- **RT-DETR-Smoke** (MDPI 2025) — no public weights.
- **FSH-DETR** (MDPI 2024) — no release found at run time.
- **MS-FSDB benchmark checkpoints** — repo ships configs + dataset links only;
  weights require author request.
- **FASDD-trained Swin / Cascade / Mask R-CNN** (openrsgis) — dataset and
  training code published, no distributed weight artefacts.
- **FireRisk MAE**, **SmokeyNet**, **UAV real-time smoke segmenter**,
  **Hyperspectral Smoke MoP** — paper-only, deferred.

### Datasets mentioned in docs (not in scope)

FASDD / D-Fire / FIRESENSE / FLAME / FireNet are datasets, not pretrained
weights. Not part of this download pass.

## Summary

- HF repos attempted: **23**
- HF repos fully downloaded: **19** (all folders non-empty and containing
  the primary `best.pt` / `pytorch_model.bin` / `model.onnx` / `model.safetensors`).
- HF repos partial (killed at 120 s timeout): **3**
  (`shawnmichael/vit-fire-smoke-detection-v4`,
   `shawnmichael/vit-large-fire-smoke-detection-v1`,
   `dima806/wildfire_types_image_detection`).
- HF gated 401: **0**.
- HF repos pre-existing (skipped): **3** pyronear + **2** ustc-community.
- GitHub release assets pulled: **4** (`dfine_s_coco.pth`, `dfine_m_coco.pth`,
  `dfine_l_obj365.pth`, `yolox_m.pth`).
- Google Drive assets (DEIM): **2** (S + M).
- Paper-only / RF-DETR GCS / deferred: **~7 families** (see table above).

Total new folders/files on disk: **31 folders + 7 loose `.pth`**, ~**2.0 GB**.

## Re-download pointers

Partial repos can be completed by re-running, for example:

```bash
hf download shawnmichael/vit-large-fire-smoke-detection-v1 \
  --local-dir ai/pretrained/safety-fire_detection/shawnmichael_vit-large-fire-smoke
```

The 120 s timeout was the only reason they aborted; files larger than the
budget simply need a longer fetch window.

## License reminder (not revisited here)

Every Ultralytics-based checkpoint in this pull (YOLOv5 / v8 / v10 / v11 /
v12 / v26, including all pyronear YOLO repos) remains **AGPL-3.0 derivative
work** regardless of the `license:` tag on the HF card. See
`safety-fire_oss-pretrained-deep-dive.md` §7 for the full commentary.
These weights are downloaded to accelerate offline evaluation and
distillation experiments only. **Do not ship them in product binaries
without Ultralytics Enterprise licensing.**
