# Dataset Store

Root: `ai/dataset_store/`

## Layout

```
dataset_store/
├── raw/                          # public datasets, downloaded via MCP / curl / git
│   ├── _coco_val/                # (empty — COCO val2017 placeholder)
│   ├── apron_detection/
│   ├── ear_protection/
│   ├── fall_detection/
│   ├── fire_detection/
│   ├── glove_detection/
│   ├── harness_detection/
│   ├── helmet_detection/
│   ├── mask_detection/
│   ├── phone_detection/
│   ├── shoes_detection/
│   ├── smoking_detection/
│   └── zone_intrusion/
├── site_collected/               # internal Nitto Denko site footage (empty placeholders)
│   ├── fall_detection/
│   ├── fire_detection/
│   ├── helmet_detection/
│   ├── phone_detection/
│   ├── shoes_detection/
│   └── test_video/
└── training_ready/               # derived splits built by core/p00_data_prep/ (currently empty)
```

Pretrained weights live **outside** this tree at `ai/pretrained/` (YOLOX, SCRFD, etc.).

---

## Top-level summary

> **Disk-size snapshot last verified: 2026-04-21.** Image counts are from Phase A data-prep logs (2026-04-17) and are not re-counted here — for current per-source counts see each feature's `DATASET_REPORT.md`. To refresh disk sizes: `du -sh dataset_store/raw/*/ dataset_store/training_ready/`.

| Category | Size | Imgs | Notes |
|---|---|---|---|
| `raw/fire_detection` | 16 GB | 131,387 | d_fire + zenodo + industrial + fasdd_cv + fire_seg |
| `raw/helmet_detection` | 25 GB | 114,929 | sh17, shlokraval, hardhat_vest_v3, roboflow_hardhat + 7 more (sfchd/shwd Baidu-only) |
| `raw/fall_detection` | 10 GB | 71,501 | coco_keypoints (pose-only), roboflow_person_fall_8k, roboflow_fall, cctv_fall |
| `raw/shoes_detection` | 8.4 GB | 34,662+ | grew post-v1: keremberke, rf_feet_segmentation, rf_safety_shoes_11k, rf_footwear_*, rf_sneakers, rf_barefoot, shoe_ppe |
| `raw/phone_detection` | 4.6 GB | 63,419 | fpi_det + 7 roboflow sources (person-using-phone coverage) |
| `raw/smoking_detection` | 676 MB | 10,291 | rf_cigarette_5k |
| `raw/harness_detection` | 547 MB | 3,303 | rf_body_harness + rf_safety_harness_v2 |
| `raw/mask_detection` | 465 MB | 12,226 | rf_n95, rf_mask_3class, rf_mask_3k, mask_type_det |
| `raw/glove_detection` | 473 MB | 1,181 | datacluster_gloves (VOC), rf_hand_gloves |
| `raw/zone_intrusion` | 368 MB | 6,507 | rf_intrusion_3k + rf_intrusion_face_3k |
| `raw/apron_detection` | 76 MB | 1,379 | rf_apron |
| `raw/ear_protection` | 37 MB | 780 | rf_earplug |
| `raw/_coco_val` | — | 0 | empty (COCO val2017 placeholder) |
| `site_collected/` | — | 0 | placeholders only (populate on-site) |
| `training_ready/` | 11 GB | 112,099 | Phase A complete (2026-04-17) — 5 features built (fire, helmet, shoes, fall, phone-usage) |

**Total raw: ~67 GB across 12 categories, ~450K images.**

---

## Per-source inventory (disk truth)

Legend: ✅ usable · ⚠️ caveat · ❌ stub/empty

### raw/fall_detection (10 GB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `coco_keypoints` | 9.3 GB | 58,945 | ⚠️ Keypoints JSON only, no bbox — reserve for pose model |
| `roboflow_person_fall_8k` | 338 MB | 7,947 | ✅ person-fall / person-nofall |
| `roboflow_fall` | 165 MB | 4,497 | ✅ Fall-Detected |
| `cctv_fall` | 165 MB | 112 | ✅ laying / standing (pose kpts present, YOLO parser ignores) |

### raw/fire_detection (16 GB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `fasdd_cv` | 12 GB | 95,314 | ⚠️ COCO JSON (95K imgs, needs conversion) — v2 |
| `d_fire` | 3.0 GB | 21,527 | ✅ fire / smoke — paper-authored |
| `zenodo_indoor_fire` | 258 MB | 10,000 | ✅ 0→fire, 1→smoke |
| `roboflow_fire_seg` | 262 MB | 3,203 | ✅ seg masks — reserve for p03 gen aug |
| `industrial_hazards` | 83 MB | 1,343 | ✅ fire/smoke (+ chem/no-helmet/water-leak dropped) |

### raw/helmet_detection (25 GB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `sh17_ppe` | 14 GB | 8,099 | ✅ 17-class Pexels-sourced, VOC + YOLO + JSON |
| `hardhat_vest_v3` | 4.4 GB | 22,141 | ⚠️ Community-scraped near-dupes — v2 |
| `shlokraval_ppe_yolov8` | 2.6 GB | 44,002 | ✅ 14-class incl. NO-Hardhat/Gloves/Goggles/Mask/Vest |
| `roboflow_hardhat` | 1.6 GB | 18,291 | ⚠️ Single-source, redundant w/ sh17 — v2 |
| `hard_hat_workers` | 1.3 GB | 5,000 | ⚠️ VOC XML — needs conversion, Western-sites — v2 |
| `roboflow_vest_helmet_9k` | 529 MB | 9,526 | ⚠️ 2-class only, flagged "spot-check labels" |
| `roboflow_ppe_6class` | 183 MB | 3,481 | ✅ MIT, Boots/Gloves/Goggles/Helmet/Person/Vest |
| `construction_safety` | 163 MB | 2,777 | ✅ 10-class chemical PPE |
| `rf_siabar_ppe` | 108 MB | 1,613 | ✅ 8-class incl. Boots + Ear-protection |
| `rf_chemical_ppe_11class` | 105 MB | 1,180 | ✅ 11-class chemical context + NO-Shoes |
| `shwd` | 3.6 MB | 21 | ❌ Git sample only; full 7,581 set Baidu-only |
| `sfchd` | 4 KB | 0 | ❌ Baidu-only (12K imgs, lijfrank/SFCHD-SCALE) |

### raw/phone_detection (4.6 GB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `fpi_det` | 3.6 GB | 45,761 | ✅ phone+face YOLO, 22,879 unique (copy duplicates size). License **AGPL-3.0** |
| `rf_phone_using_fyp` | 264 MB | 4,216 | ✅ Phone (person-context) |
| `rf_phone_texting_4class` | 316 MB | 3,193 | ✅ "using phone" + opaque v9 (dropped) |
| `rf_using_phone_unimonitor` | 143 MB | 3,748 | ✅ Using Phone + mobile |
| `phone_call_usage` | 128 MB | 3,115 | ⚠️ Opaque class name `phone-person - v1...` |
| `roboflow_mobile_phone_ineuron` | 66 MB | 1,522 | ⚠️ cell phone object only (not usage) |
| `roboflow_phone_usage_dms` | 30 MB | 997 | ✅ phone usage |
| `rf_texting_calling_drinking` | 87 MB | 867 | ✅ Texting/Calling/(Drinking dropped) |
| `state_farm_distracted` | 4 KB | 0 | ❌ Kaggle competition, requires rules acceptance |

### raw/shoes_detection (3.6 GB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `keremberke_ppe` | 2.2 GB | 11,979 | ⚠️ COCO JSON; only ~1,361 shoe anns (mostly glove/goggle) |
| `rf_feet_segmentation` | 835 MB | 17,888 | ⚠️ foot seg masks — reserve for two-stage shoe_region |
| `rf_barefoot` | 378 MB | 1,258 | ✅ no_shoes (barefoot negatives) |
| `rf_footwear_sandal_shoe` | 119 MB | 1,967 | ✅ Sandal / Shoe / Unidentified |
| `shoe_ppe` | 108 MB | 1,570 | ❌ Opaque class `"14"` — unusable |

### raw/harness_detection (547 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_body_harness` | 241 MB | 2,069 | ✅ safety_harness / worker |
| `rf_safety_harness_v2` | 306 MB | 1,234 | ✅ anchored / non_anchored / no_harness (hook detection) |
| `rf_harness_64class` | 4 KB | 0 | ❌ Empty stub |

### raw/mask_detection (465 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_mask_3k` | 224 MB | 7,182 | ✅ with_mask / without_mask |
| `rf_mask_3class` | 117 MB | 2,307 | ✅ mask_weared_incorrect + with/without |
| `rf_n95` | 107 MB | 2,461 | ✅ N95 / ear / earplug |
| `mask_type_det` | 18 MB | 276 | ✅ VOC mask subtype classification |

### raw/glove_detection (473 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `datacluster_gloves` | 443 MB | 236 | ✅ VOC annotations |
| `rf_hand_gloves` | 31 MB | 945 | ✅ HAND-GLOVES |

### raw/zone_intrusion (368 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_intrusion_3k` | 280 MB | 3,141 | ✅ intrusion_detection |
| `rf_intrusion_face_3k` | 88 MB | 3,366 | ✅ face / not_face / person |

### raw/apron_detection (76 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_apron` | 76 MB | 1,379 | ✅ Wearing-Apron (chem/split subtypes need site collection) |

### raw/smoking_detection (676 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_cigarette_5k` | 676 MB | 10,291 | ✅ cigarette / face |

### raw/ear_protection (37 MB)
| Source | Size | Imgs | Status |
|---|---|---|---|
| `rf_earplug` | 37 MB | 780 | ✅ 2-class earplug |

### raw/_coco_val
Empty placeholder for COCO val2017 (reserve for zone-intrusion pretrained eval).

---

## site_collected/ (internal Nitto Denko footage)

All empty placeholders — populate on-site. No public source. Layout:
- `site_collected/fall_detection/`
- `site_collected/fire_detection/`
- `site_collected/helmet_detection/` (also the only source for `head_with_nitto_hat` class)
- `site_collected/phone_detection/`
- `site_collected/shoes_detection/`
- `site_collected/test_video/` (demo videos)

---

## training_ready/ (derived)

Phase A complete (2026-04-17) — 5 features built. Build remaining features with:
```bash
uv run core/p00_data_prep/run.py --config features/<feature>/configs/00_data_preparation.yaml
```

### Built datasets (Phase A)

| Feature | dataset_name | Images | Splits |
|---|---|---|---|
| safety-fire_detection | `fire_detection` | 17,373 | train/val/test |
| ppe-helmet_detection | `helmet_detection` | 22,323 | train/val/test |
| ppe-shoes_detection | `shoes_detection` | 37,026 | train/val/test |
| safety-fall-detection | `fall_detection` | 12,402 | train/val/test |
| safety-poketenashi_phone_usage | `safety_poketenashi_phone_usage` | 22,975 | train/val/test |

### Phase 1 v1 — planned distribution (dry-run verified)

All 5 configs wired in `features/<feature>/configs/00_data_preparation.yaml` and pass `--dry-run`.

**Grand total: 112,099 samples across 5 Phase 1 models.**

#### Model A — Fire (`safety-fire_detection`)
**17,373 samples · fire 46% / smoke 54%**
| Source | Imgs contributed | Why |
|---|---|---|
| `fire_detection/d_fire` | ~15,600 (all splits) | Paper-authored, clean YOLO, primary |
| `fire_detection/zenodo_indoor_fire` | ~500 | Indoor diversity |
| `fire_detection/industrial_hazards` | ~1,273 | Factory context |

#### Model G — Fall (`safety-fall-detection`)
**12,402 samples · person 62% / fallen_person 38%**
| Source | Imgs contributed | Why |
|---|---|---|
| `fall_detection/roboflow_person_fall_8k` | 7,947 | Clean 2-class fall/nofall (primary) |
| `fall_detection/roboflow_fall` | 4,497 | Fall angles/scenes |
| `fall_detection/cctv_fall` | 112 | CCTV angle (deployment match) |

#### Model B — Helmet (`ppe-helmet_detection`)
**22,323 samples · head_with_helmet 74% / head_without_helmet 21% / head_with_nitto_hat 1.5% / person 3% ⚠️**
| Source | Imgs contributed | Why |
|---|---|---|
| `helmet_detection/shlokraval_ppe_yolov8` | ~14,500 | Only large NO-Hardhat negatives source |
| `helmet_detection/sh17_ppe` (VOC) | ~7,000 | Pexels-sourced diversity (backbone) |
| `helmet_detection/rf_bump_cap` | 664 | **Bump-cap / No-BumpCap — fills `head_with_nitto_hat` class** |
| `helmet_detection/roboflow_ppe_6class` | ~200 | Cross-PPE Person context |

Canonical class `head_with_nitto_hat` — now has 926 instances from bump-cap dataset. Site-collected Nitto footage will strengthen further.

#### Model F — Shoes (`ppe-shoes_detection`)
**37,026 samples · foot_with_safety_shoes 68% / foot_without_safety_shoes 29% / person 2.4% ⚠️**
| Source | Imgs contributed | Why |
|---|---|---|
| `shoes_detection/rf_safety_shoes_11k` | 19,976 | **Big safety-shoes backbone** (single class, all positive) |
| `shoes_detection/rf_footwear_3class_mohamed` | 7,982 | 3-class shoes/no_shoes/slippers (perfect taxonomy match) |
| `shoes_detection/rf_footwear_9class` | 4,695 | 9-class Boots vs Sneakers/Sandals/Heels etc. |
| `shoes_detection/rf_safety_shoes_flipflop` | 3,405 | safety_shoes vs FlipFlops |
| `shoes_detection/rf_sneakers` | 1,630 | Pure sneaker negatives |
| `shoes_detection/rf_barefoot` | 1,258 | Barefoot negatives |
| `shoes_detection/rf_footwear_sandal_shoe` | 1,967 | Sandal negatives |
| `helmet_detection/roboflow_ppe_6class` | ~1,000 | Boots + Person |
| `helmet_detection/rf_chemical_ppe_11class` | ~200 | Shoes/NO-Shoes + Person |
| `shoes_detection/keremberke_ppe` | ~200 | 1,361 shoe anns |

#### Model H — Phone Usage (`safety-poketenashi_phone_usage`)
**22,975 samples · phone_usage 91% (25,631 anns) / person 9% (1,449 anns) ⚠️**
| Source | Imgs contributed | Why |
|---|---|---|
| `phone_detection/fpi_det` | 18,800 (train) | Face+phone YOLO, every frame is person-using-phone (manually fetched from Baidu) |
| `phone_detection/rf_phone_using_fyp` | ~4,000 | Phone in person context |
| `phone_detection/rf_using_phone_unimonitor` | ~3,500 | Using Phone + mobile |
| `phone_detection/rf_phone_texting_4class` | ~3,000 | using phone |
| `phone_detection/roboflow_phone_usage_dms` | ~1,000 | phone usage |
| `phone_detection/rf_texting_calling_drinking` | ~800 | Texting/Calling action labels |
| `helmet_detection/shlokraval_ppe_yolov8` | subset | Person class only |

### Class-balance caveats (v1)

- **Helmet `person` 3%**: primary task is head classes; supplementary. If mAP@person is too low, pull `roboflow_hardhat` into v2 (18K imgs with head/helmet/person).
- **Phone-usage `person` 9%**: `phone_usage` is the alert class; tracking person is done separately. If needed, add a dedicated person-rich source (COCO val, roboflow_hardhat).
- **Shoes 2.7K**: smaller than other models; `rf_feet_segmentation` (17K) held back for two-stage pipeline in v2.

### V1 source URLs (click to inspect label quality manually)

All links open the source page — browse images, preview labels, check for noise before accepting into training.

#### Fire (Model A)
- [`d_fire`](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) — Kaggle, sayedgamal99/smoke-fire-detection-yolo
- [`zenodo_indoor_fire`](https://zenodo.org/records/15826133) — Zenodo record 15826133
- [`industrial_hazards`](https://www.kaggle.com/datasets/vigneshnachu/industrial-hazards-detection) — Kaggle, vigneshnachu/industrial-hazards-detection

#### Fall (Model G)
- [`roboflow_person_fall_8k`](https://universe.roboflow.com/xzjs/person-fall-ewpyk) — Roboflow Universe, xzjs/person-fall-ewpyk
- [`roboflow_fall`](https://universe.roboflow.com/roboflow-universe-projects/fall-detection-ca3o8) — Roboflow Universe, roboflow-universe-projects/fall-detection-ca3o8 v1
- [`cctv_fall`](https://www.kaggle.com/datasets/simuletic/cctv-incident-dataset-fall-and-lying-down-detection) — Kaggle, simuletic/cctv-incident-dataset-fall-and-lying-down-detection

#### Helmet (Model B)
- [`sh17_ppe`](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection) — Kaggle, mugheesahmad/sh17-dataset-for-ppe-detection (17-class, Pexels-sourced, arXiv paper [2407.04590](https://arxiv.org/abs/2407.04590))
- [`shlokraval_ppe_yolov8`](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8) — Kaggle, shlokraval/ppe-dataset-yolov8 (14-class incl. NO-* negatives; mirror of [Roboflow PPE combined v4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model))
- [`roboflow_ppe_6class`](https://universe.roboflow.com/dataset-ppe/ppe-vest-helmet) — Roboflow Universe, dataset-ppe/ppe-vest-helmet v1
- [`rf_bump_cap`](https://universe.roboflow.com/n-wto9b/bump-cap) — Roboflow Universe, n-wto9b/bump-cap v4 (**Nitto soft hat source**)

#### Shoes (Model F)
- [`rf_safety_shoes_11k`](https://universe.roboflow.com/bns218/safety-shoes-yawyh) — Roboflow Universe, bns218/safety-shoes-yawyh v1 (biggest safety-shoes set)
- [`rf_footwear_3class_mohamed`](https://universe.roboflow.com/mohamed-nihal/footwear-hh4hz) — Roboflow Universe, mohamed-nihal/footwear-hh4hz v1 (shoes/no_shoes/slippers)
- [`rf_footwear_9class`](https://universe.roboflow.com/workspace4-hdyta/footwear-p5juo) — Roboflow Universe, workspace4-hdyta/footwear-p5juo v7 (Boots/Sneakers/Sandals/etc.)
- [`rf_safety_shoes_flipflop`](https://universe.roboflow.com/bagas-nfbz0/safety-shoes-adtln) — Roboflow Universe, bagas-nfbz0/safety-shoes-adtln v1
- [`rf_sneakers`](https://universe.roboflow.com/ds06/sneakers-k7svc) — Roboflow Universe, ds06/sneakers-k7svc v5
- [`rf_barefoot`](https://universe.roboflow.com/mohamed-nihal/barefoot) — Roboflow Universe, mohamed-nihal/barefoot v1
- [`rf_footwear_sandal_shoe`](https://universe.roboflow.com/ariel-hzk0a/footwear-2ws0y) — Roboflow Universe, ariel-hzk0a/footwear-2ws0y v1
- [`keremberke_ppe`](https://huggingface.co/datasets/keremberke/protective-equipment-detection) — HF, keremberke/protective-equipment-detection
- cross-use from helmet: [`roboflow_ppe_6class`](https://universe.roboflow.com/dataset-ppe/ppe-vest-helmet) · [`rf_chemical_ppe_11class`](https://universe.roboflow.com/data2-4uqki/chemical-8oaqi)

#### Phone-usage (Model H)
- [`fpi_det`](https://github.com/KvCgRv/FPI-Det) — GitHub, KvCgRv/FPI-Det (**manually fetched via Baidu Netdisk, code `Mofo`**) — [arXiv paper 2509.09111](https://arxiv.org/abs/2509.09111). License: AGPL-3.0.
- [`rf_using_phone_unimonitor`](https://universe.roboflow.com/unimonitor-03qbp/using-phone-pqaut) — Roboflow Universe, unimonitor-03qbp/using-phone-pqaut v1
- [`rf_phone_using_fyp`](https://universe.roboflow.com/fyp-odgmu/phone-using-dhjqe) — Roboflow Universe, fyp-odgmu/phone-using-dhjqe v3
- [`rf_phone_texting_4class`](https://universe.roboflow.com/cv2-kyb30/mobile-phone-texting) — Roboflow Universe, cv2-kyb30/mobile-phone-texting v4
- [`rf_texting_calling_drinking`](https://universe.roboflow.com/hand-rules/texting-calling-drinking) — Roboflow Universe, hand-rules/texting-calling-drinking v1
- [`roboflow_phone_usage_dms`](https://universe.roboflow.com/dms-ejczp/phone-usage-detection-a6oou) — Roboflow Universe, dms-ejczp/phone-usage-detection-a6oou v1
- cross-use from helmet: [`shlokraval_ppe_yolov8`](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8) (Person class only)

### v2 candidates (held back from v1)

| Category | Dataset held | Reason held | When to add |
|---|---|---|---|
| Fire | `fasdd_cv` (95K) | COCO→YOLO conversion overhead + near-duplicates | If domain-gap at test time |
| Fire | `roboflow_fire_seg` | Seg masks, reserve for p03 generative aug | When p03 gen aug pipeline runs |
| Helmet | `hardhat_vest_v3` (22K) | Community-scraped near-dupes w/ sh17 | If recall <target on construction scenes |
| Helmet | `roboflow_hardhat` (18K) | Single-source, head/helmet/person | If person class weak |
| Helmet | `hard_hat_workers` (5K) | VOC, Western-sites, already covered | Diversity boost |
| Helmet | `rf_chemical_ppe_11class`, `rf_siabar_ppe` | Small, chem/ear context | Once chemical-area deployment starts |
| Shoes | `rf_feet_segmentation` (17K) | Seg masks for shoe_region pipeline | When two-stage detection enabled |
| Shoes | `sh17_ppe` boots, `rf_siabar_ppe` Boots | Cross-use from helmet datasets | More Boots diversity |
| Fall | `coco_keypoints` (58K) | Pose-only, no bbox | When pose model (Model G-pose) starts |
| Phone | `roboflow_mobile_phone_ineuron` (1.5K) | Object-only (not usage) | If cell-phone object detection split out |
| Zone intrusion | `rf_intrusion_3k`, `rf_intrusion_face_3k` | Model I uses pretrained — no training needed in Phase 1 | If Phase 2 retraining |

---

## Downloading datasets (via MCP, no scripts)

- **Roboflow Universe**: `mcp__roboflow__download_universe_dataset` (workspace/project/version/location)
- **Kaggle**: `mcp__kaggle__download_dataset` returns signed URL → `curl -L` + `unzip`
- **Hugging Face**: `hf download <repo> --repo-type dataset --local-dir <path>`
- **Direct (GitHub/Zenodo)**: `curl -L -o` + `unzip`

Before re-downloading a stub: `rm -rf raw/<cat>/<source>` then re-fetch.

---

## Source registry (public, reproducible)

All entries below are CLI-downloadable from scratch. For Baidu-only entries see "Known gaps" below.

### Fire / smoke
| dest | source | license |
|---|---|---|
| `raw/fire_detection/d_fire` | kaggle `sayedgamal99/smoke-fire-detection-yolo` | CC-BY-4.0 |
| `raw/fire_detection/fasdd_cv` | kaggle `yuulind/fasdd-cv-coco` | CC-BY-4.0 |
| `raw/fire_detection/zenodo_indoor_fire` | url `https://zenodo.org/api/records/15826133/files/Indoor%20Fire%20Smoke.zip/content` | CC-BY-4.0 |
| `raw/fire_detection/roboflow_fire_seg` | roboflow `fire-nxmk4/fire-smoke-detection-elt66` v1 | MIT |
| `raw/fire_detection/industrial_hazards` | kaggle `vigneshnachu/industrial-hazards-detection` | CC-BY-4.0 |

### Helmet / PPE
| dest | source | license |
|---|---|---|
| `raw/helmet_detection/sh17_ppe` | kaggle `mugheesahmad/sh17-dataset-for-ppe-detection` | CC-BY-NC-SA-4.0 |
| `raw/helmet_detection/shlokraval_ppe_yolov8` | kaggle `shlokraval/ppe-dataset-yolov8` | Apache-2.0 |
| `raw/helmet_detection/hardhat_vest_v3` | kaggle `muhammetzahitaydn/hardhat-vest-dataset-v3` | CC-BY-4.0 |
| `raw/helmet_detection/hard_hat_workers` | kaggle `andrewmvd/hard-hat-detection` | PDDL |
| `raw/helmet_detection/construction_safety` | roboflow `custom-model/construction-safety-nvqbd` v2 | CC-BY-4.0 |
| `raw/helmet_detection/roboflow_hardhat` | roboflow `yus-space/construction-site-safety-helmet` v1 | CC-BY-4.0 |
| `raw/helmet_detection/roboflow_vest_helmet_9k` | roboflow `uhhh/vest-helmet-wlbch` v1 | CC-BY-4.0 |
| `raw/helmet_detection/roboflow_ppe_6class` | roboflow `dataset-ppe/ppe-vest-helmet` v1 | MIT |
| `raw/helmet_detection/rf_siabar_ppe` | roboflow `siabar/ppe-dataset-for-workplace-safety` v2 | CC-BY-4.0 |
| `raw/helmet_detection/rf_chemical_ppe_11class` | roboflow `data2-4uqki/chemical-8oaqi` v3 | CC-BY-4.0 |
| `raw/helmet_detection/shwd` | github `https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset` (git-clone only 21 sample imgs; full set Baidu) | MIT |

### Harness
| dest | source | license |
|---|---|---|
| `raw/harness_detection/rf_body_harness` | roboflow `construction-images/body_harness` v2 | CC-BY-4.0 |
| `raw/harness_detection/rf_safety_harness_v2` | roboflow `techling/safety_harness_v2` v1 | CC-BY-4.0 |

### Apron
| dest | source | license |
|---|---|---|
| `raw/apron_detection/rf_apron` | roboflow `knowledgeflex/apron-detection` v6 | CC-BY-4.0 |

### Mask
| dest | source | license |
|---|---|---|
| `raw/mask_detection/rf_n95` | roboflow `yolo-uwp4e/n95-sdn3f` v2 | CC-BY-4.0 |
| `raw/mask_detection/rf_mask_3class` | roboflow `data-labeling/face-mask-detection-euti3` v2 | CC-BY-4.0 |
| `raw/mask_detection/rf_mask_3k` | roboflow `cat-vs-dog-classification/face-mask-detection-j3gwp` v1 | CC-BY-4.0 |
| `raw/mask_detection/mask_type_det` | kaggle `takiaibnath19/mask-type-detection-dataset` | CC0 |

### Glove
| dest | source | license |
|---|---|---|
| `raw/glove_detection/datacluster_gloves` | kaggle `dataclusterlabs/gloves-dataset-covid-safety-wear` | CC0 |
| `raw/glove_detection/rf_hand_gloves` | roboflow `terry-kanana-21-gmail-com/hand-gloves-ik2gg` v1 | CC-BY-4.0 |

### Shoes
| dest | source | license |
|---|---|---|
| `raw/shoes_detection/keremberke_ppe` | hf `keremberke/protective-equipment-detection` | AGPL |
| `raw/shoes_detection/shoe_ppe` | roboflow `harnessafesite/shoe_ppe` v1 | CC-BY-4.0 |
| `raw/shoes_detection/rf_barefoot` | roboflow `mohamed-nihal/barefoot` v1 | CC-BY-4.0 |
| `raw/shoes_detection/rf_footwear_sandal_shoe` | roboflow `ariel-hzk0a/footwear-2ws0y` v1 | CC-BY-4.0 |
| `raw/shoes_detection/rf_feet_segmentation` | roboflow `unlimited-robotics/feet-segmentation-q30ii` v5 | CC-BY-4.0 |

### Fall
| dest | source | license |
|---|---|---|
| `raw/fall_detection/coco_keypoints` | kaggle `asad11914/coco-2017-keypoints` | CC-BY-4.0 |
| `raw/fall_detection/roboflow_fall` | roboflow `roboflow-universe-projects/fall-detection-ca3o8` v1 | CC-BY-4.0 |
| `raw/fall_detection/cctv_fall` | kaggle `simuletic/cctv-incident-dataset-fall-and-lying-down-detection` | CC-BY-4.0 |
| `raw/fall_detection/roboflow_person_fall_8k` | roboflow `xzjs/person-fall-ewpyk` | CC-BY-4.0 |

### Phone
| dest | source | license |
|---|---|---|
| `raw/phone_detection/fpi_det` | github `KvCgRv/FPI-Det` (Baidu Netdisk, code `Mofo`) — **manually fetched** | AGPL-3.0 |
| `raw/phone_detection/phone_call_usage` | roboflow `phoneusagedetection/phone-call-usage` | CC-BY-4.0 |
| `raw/phone_detection/roboflow_mobile_phone_ineuron` | roboflow `ineuron-8bdse/mobile-phone-b83c7` v1 | MIT |
| `raw/phone_detection/roboflow_phone_usage_dms` | roboflow `dms-ejczp/phone-usage-detection-a6oou` v1 | CC-BY-4.0 |
| `raw/phone_detection/rf_phone_using_fyp` | roboflow `fyp-odgmu/phone-using-dhjqe` v3 | CC-BY-4.0 |
| `raw/phone_detection/rf_using_phone_unimonitor` | roboflow `unimonitor-03qbp/using-phone-pqaut` v1 | CC-BY-4.0 |
| `raw/phone_detection/rf_phone_texting_4class` | roboflow `cv2-kyb30/mobile-phone-texting` v4 | CC-BY-4.0 |
| `raw/phone_detection/rf_texting_calling_drinking` | roboflow `hand-rules/texting-calling-drinking` v1 | CC-BY-4.0 |
| `raw/phone_detection/state_farm_distracted` | kaggle competition `state-farm-distracted-driver-detection` | competition-use |

### Zone intrusion
| dest | source | license |
|---|---|---|
| `raw/zone_intrusion/rf_intrusion_3k` | roboflow `ardhas-workspace/intrusion-vstvg` v2 | CC-BY-4.0 |
| `raw/zone_intrusion/rf_intrusion_face_3k` | roboflow `none-n1imd/intrusion-7t7sz` v2 | CC-BY-4.0 |

### Smoking / Ear / COCO
| dest | source | license |
|---|---|---|
| `raw/smoking_detection/rf_cigarette_5k` | roboflow `cigarette-c6554/cigarette-ghnlk` v7 | CC-BY-4.0 |
| `raw/ear_protection/rf_earplug` | roboflow `garyspace/earplug-jocg0` v1 | Public Domain |
| `raw/_coco_val` | hf dataset `lmms-lab/COCO` (val2017) | CC-BY-4.0 |

---

## Known gaps

- **`sfchd`** (12K chemical-plant helmet imgs): Baidu-only, `lijfrank/SFCHD-SCALE` repo has the extraction code.
- **`shwd`** full set (7,581 helmet imgs): Baidu-only, `njvisionpower/Safety-Helmet-Wearing-Dataset` repo — only 21 sample imgs ship via git.
- **`rf_harness_64class`**: listed on Universe but no versioned release → untrainable via MCP.
- **Chemical/split apron subtypes, N95 vs respirator vs face_shield detection-level labels, chem/cut/insulated glove subtypes**: no public datasets — need site_collected + SAM3 auto-labeling.
- **`state_farm_distracted`**: Kaggle competition, requires rules acceptance before download.
- **`ur_fall`**: old academic mirror, URL dead — new mirror unknown.

## Annotation conversions pending

- `coco_keypoints` (keypoint JSON → pose-specific loader)
- `fasdd_cv` (COCO → YOLO)
- `keremberke_ppe` (COCO — already supported by `core/p00_data_prep/parsers/coco.py`)
- `hard_hat_workers` (VOC XML — supported by `parsers/voc.py`)
- `sh17_ppe` (VOC XML — supported; YOLO class-index order unclear, prefer VOC labels)

## Disk cleanup opportunities

- `raw/phone_detection/fpi_det/fpi_det_images/reorganized_dataset/` — exact duplicate of top-level `reorganized_dataset/` (1.3 GB redundant).
