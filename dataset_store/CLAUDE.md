# Dataset Store

Root: `ai/dataset_store/raw/`

## Download Status (as of 2026-04-15)

| Dataset | Size | Images | Labels | Status |
|---|---|---|---|---|
| fall_detection | ~11 GB | 74,826 | 15,886 | Partial |
| fire_detection | ~20 GB | 132,409 | 35,196 | Partial |
| helmet_detection | ~42 GB | 101,627 | 154,802 | Partial |
| phone_detection | ~4.3 GB | 66,987 | 48,979 | Partial |
| shoes_detection | ~4.4 GB | 13,549 | 1,578 | Partial |
| _coco_val | — | 0 | 0 | **Empty** |
| zone_intrusion | — | 0 | 0 | **Empty** |

---

## Per-Source Breakdown

### fall_detection (~11 GB)

| Source | Size | Images | Labels | Notes |
|---|---|---|---|---|
| cctv_fall | 165 MB | 112 | 111 | OK |
| coco_keypoints | 9.3 GB | 58,945 | 2 | Needs annotation conversion |
| fall_detection_imgs | 54 MB | 485 | 485 | OK |
| roboflow_fall | 575 MB | 10,787 | 10,789 | OK |
| roboflow_fall_alt | 165 MB | 4,497 | 4,499 | OK |
| ur_fall | — | 0 | 0 | **Empty — download failed** |

### fire_detection (~20 GB)

| Source | Size | Images | Labels | Notes |
|---|---|---|---|---|
| d_fire | 3.0 GB | 21,528 | 21,527 | OK |
| fasdd_cv | 12 GB | 95,317 | 3 | Needs annotation conversion |
| fire_cctv | 355 MB | 1,728 | 0 | **No labels** |
| fire_smoke_datacluster | 90 MB | 100 | 100 | OK |
| huggingface_d_fire | 4.1 GB | 7,393 | 7,221 | OK |
| industrial_hazards | 83 MB | 1,343 | 1,345 | OK |
| roboflow_fire | — | 0 | 0 | **Empty — download failed** |
| zenodo_indoor_fire | 409 MB | 5,000 | 5,000 | OK |

### helmet_detection (~42 GB)

| Source | Size | Images | Labels | Notes |
|---|---|---|---|---|
| construction_safety | 233 MB | 2,825 | 2,803 | OK |
| hardhat_vest_v3 | 4.4 GB | 22,141 | 22,142 | OK |
| hard_hat_workers | 1.3 GB | 5,000 | 5,000 | OK |
| huggingface_safety_helmet | 329 MB | 7,035 | 7,035 | OK |
| ppe_yolov8 | 2.6 GB | 44,002 | 44,004 | OK |
| roboflow_hardhat | — | 0 | 0 | **Empty — download failed** |
| sfchd | 4.1 GB | 12,504 | 49,519 | OK (multiple labels per image) |
| sh17_ppe | 29 GB | 8,099 | 24,299 | OK (multiple labels per image) |
| shwd | 3.6 MB | 21 | 0 | **No labels; nearly empty** |

### phone_detection (~4.3 GB)

| Source | Size | Images | Labels | Notes |
|---|---|---|---|---|
| action_recognition | 349 MB | 18,011 | 0 | **No labels** |
| fpi_det | 3.6 GB | 45,761 | 45,762 | OK |
| mobile_phone | 292 MB | 100 | 100 | OK |
| phone_call_usage | 129 MB | 3,115 | 3,117 | OK |

### shoes_detection (~4.4 GB)

| Source | Size | Images | Labels | Notes |
|---|---|---|---|---|
| additional_downloads | — | 0 | 0 | **Empty** |
| keremberke_ppe | 4.3 GB | 11,979 | 6 | Needs annotation conversion |
| shoe_ppe | 108 MB | 1,570 | 1,572 | OK |

### _coco_val

Empty — not yet populated.

### zone_intrusion

Empty — not yet populated.

---

## Download Scripts

Canonical download system: `ai/scripts/bootstrap/`

```bash
# Put credentials in ai/../.env (loaded automatically)
# KAGGLE_USERNAME=... KAGGLE_KEY=... ROBOFLOW_API_KEY=... HF_TOKEN=...

bash ai/scripts/bootstrap/download_datasets.sh                    # all datasets
bash ai/scripts/bootstrap/download_datasets.sh --only fire        # single feature
bash ai/scripts/bootstrap/download_datasets.sh --dry-run          # preview only
bash ai/scripts/bootstrap/download_datasets_missing_only.sh       # re-download empties
```

Manifest: `ai/scripts/bootstrap/manifests/datasets.tsv`

**Manual / rsync-only sources** (no public CLI):
- `fire_cctv`, `fire_smoke_datacluster`, `industrial_hazards` — internal/Datacluster, rsync from `ssh-sg4`
- `fall_detection_imgs`, `action_recognition`, `mobile_phone` — internal, rsync from `ssh-sg4`
- `construction_safety`, `roboflow_fall_alt` — internal-derived, rsync from `ssh-sg4`
- `sfchd` — Baidu fallback: https://pan.baidu.com/s/1k2pWg8r-G3KSI2Q3Tdt6kg (code: v4ao)
- `fpi_det` — Baidu fallback: https://pan.baidu.com/s/1_xjDuK9FvhguqoMwjAIlIA (code: Mofo)

---

## Action Items

- **Re-download:** `ur_fall`, `roboflow_fire`, `roboflow_hardhat`
- **Add labels / convert annotations:** `coco_keypoints`, `fasdd_cv`, `keremberke_ppe`, `fire_cctv`, `action_recognition`, `shwd`
- **Populate:** `_coco_val`, `zone_intrusion`, `additional_downloads`
