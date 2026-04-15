# Dataset Store

Root: `ai/dataset_store/`

Layout:
- `raw/<category>/<source>/` — public datasets, download via MCP / curl
- `site_collected/<category>/` — internal site footage (Nitto Denko) — no public source, collected on-site
- `training_ready/` — derived splits built by `core/p00_data_prep/` from `raw/` + `site_collected/`

Pretrained weights live **outside** this store at `ai/pretrained/` (YOLOX, SCRFD, etc.).

---

## Download Status (as of 2026-04-15, verified from disk)

| Category | On-disk | Images | Labels | Status |
|---|---|---|---|---|
| `raw/fall_detection` | 10 GB | 71,613 | 12,559 txt + 2 json | ✅ (ur_fall only academic gap) |
| `raw/fire_detection` | 15.6 GB | 131,387 | 25,075 txt + 3 json | ✅ (all auto-downloadable sources present) |
| `raw/helmet_detection` | 25 GB | 114,950 | 123,644 txt + 13,099 xml + 8,099 json | **Strong coverage** (sfchd Baidu-only) |
| `raw/harness_detection` | 547 MB | 3,303 | 3,307 | ✅ Complete |
| `raw/apron_detection` | 76 MB | 1,379 | 1,381 | ✅ Complete (1-class only) |
| `raw/mask_detection` | 466 MB | 12,226 | 12,750 | ✅ Strong (incl. N95, mask_weared_incorrect) |
| `raw/glove_detection` | 474 MB | 1,181 | ~1,420 | ✅ (classification + hand/gloves detection) |
| `raw/smoking_detection` | 676 MB | 10,291 | ~10,293 | **NEW** (2026-04-15, cigarette/face) |
| `raw/ear_protection` | 37 MB | 780 | 782 | **NEW** (2026-04-15, earplug) |
| `raw/zone_intrusion` | 368 MB | 6,507 | ~6,511 | **NEW** (2026-04-15, person/intrusion/face) |
| `raw/phone_detection` | 1.0 GB | 17,658 | ~17,662 | ✅ Strong (incl. action-level texting/calling) |
| `raw/shoes_detection` | 1.6 GB | 34,663 | Parquet + 23K txt | ✅ Strong (incl. barefoot + sandal negatives) |
| `site_collected/` | — | — | — | Placeholder (populate on-site) |
| `training_ready/` | — | — | — | Placeholder (build via p00_data_prep) |

---

## Per-Source Inventory (ground truth from disk)

Rows marked **STUB** have only a README-only placeholder — re-download via the source in the registry below.

### raw/fall_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `cctv_fall` | 165 MB | 112 | 111 txt | ✅ |
| `coco_keypoints` | 9.3 GB | 58,945 | 2 json | ✅ (keypoint JSON, needs conversion for detection) |
| `roboflow_fall` | 165 MB | 4,497 | 4,499 txt | ✅ (2026-04-15, via MCP) |
| `roboflow_person_fall_8k` | 338 MB | 7,947 | 7,949 txt | ✅ |

### raw/fire_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `d_fire` | 3.0 GB | 21,527 | 21,529 txt | ✅ (2026-04-15, Kaggle sayedgamal99) |
| `fasdd_cv` | 12 GB | 95,314 | 3 json | ✅ (COCO JSON, needs conversion) |
| `industrial_hazards` | 83 MB | 1,343 | 1,345 txt | ✅ (2026-04-15, Kaggle vigneshnachu — replaces old internal copy) |
| `roboflow_fire_seg` | 262 MB | 3,203 | 3,205 txt | ✅ (seg masks for p03 aug) |
| `zenodo_indoor_fire` | 258 MB | 10,000 | 3,000 txt | ✅ (2026-04-15, Zenodo 15826133) |

### raw/helmet_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `construction_safety` | 163 MB | 2,777 | 2,779 | ✅ (10-class, via MCP 2026-04-15) |
| `hardhat_vest_v3` | 4.4 GB | 22,141 | 22,142 | ✅ |
| `hard_hat_workers` | 1.3 GB | 5,000 | 5,000 xml | ✅ (VOC — convert) |
| `rf_siabar_ppe` | 108 MB | 1,613 | 1,615 | ✅ (8 classes incl. Boots + Ear-protection) |
| `rf_chemical_ppe_11class` | 105 MB | 1,180 | 1,182 | ✅ (2026-04-15, **11-class chemical-context PPE** incl. Faceshield/NO-Shoes) |
| `roboflow_hardhat` | 1.6 GB | 18,291 | 18,293 | ✅ |
| `roboflow_ppe_6class` | 183 MB | 3,481 | 3,483 | ✅ (6 classes incl. Boots/Gloves/Goggles) |
| `roboflow_vest_helmet_9k` | 529 MB | 9,526 | 9,528 | ✅ (2-class — spot-check) |
| `sfchd` | 16 KB | 0 | — | **STUB — Baidu only** (repo lijfrank/SFCHD-SCALE; extraction code in repo) |
| `sh17_ppe` | 14 GB | 8,099 | 8,101 txt + 8,099 xml + 8,099 json | ✅ (17 classes, Pexels sourced) |
| `shlokraval_ppe_yolov8` | 2.6 GB | 44,002 | 44,004 | ✅ (**14 classes incl. NO-Hardhat/Gloves/Goggles/Mask/Vest negatives**) |
| `shwd` | 3.6 MB | 21 | — | ⚠️ git repo only (21 sample imgs); full 7,581-img set is Baidu-only from upstream |

### raw/harness_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `rf_body_harness` | 241 MB | 2,069 | 2,071 | ✅ (safety_harness / worker) |
| `rf_safety_harness_v2` | 306 MB | 1,234 | 1,236 | ✅ (**anchored / non_anchored / no_harness** — Phase 2 hook detection) |

### raw/apron_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `rf_apron` | 76 MB | 1,379 | 1,381 | ✅ (1-class: Wearing-Apron) |

### raw/mask_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `mask_type_det` | 18 MB | 276 | 520 xml | ✅ (VOC, mask subtypes classification) |
| `rf_n95` | 107 MB | 2,461 | 2,463 | ✅ (2026-04-15, **N95/ear/earplug** subtype labels) |
| `rf_mask_3class` | 117 MB | 2,307 | 2,309 | ✅ (2026-04-15, **mask_weared_incorrect/with_mask/without_mask**) |
| `rf_mask_3k` | 224 MB | 7,182 | 7,184 | ✅ (2026-04-15, with_mask/without_mask) |

### raw/glove_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `datacluster_gloves` | 443 MB | 236 | 236 xml | ✅ (VOC annotations) |
| `rf_hand_gloves` | 31 MB | 945 | 947 | ✅ (2026-04-15, HAND-GLOVES) |

### raw/smoking_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `rf_cigarette_5k` | 676 MB | 10,291 | 10,293 | ✅ (2026-04-15, cigarette/face; most-downloaded smoking set) |

### raw/ear_protection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `rf_earplug` | 37 MB | 780 | 782 | ✅ (2026-04-15, 2-class earplug) |

### raw/zone_intrusion
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `rf_intrusion_3k` | 280 MB | 3,141 | 3,143 | ✅ (2026-04-15, intrusion_detection) |
| `rf_intrusion_face_3k` | 88 MB | 3,366 | 3,368 | ✅ (2026-04-15, face/not_face/person) |

### raw/phone_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `fpi_det` | 8 KB | 0 | — | **STUB — Baidu only** (repo KvCgRv/FPI-Det; code `Mofo`) |
| `phone_call_usage` | 128 MB | 3,115 | 3,117 txt | ✅ (2026-04-15, via MCP) |
| `roboflow_mobile_phone_ineuron` | 66 MB | 1,522 | 1,524 | ✅ (phone object only) |
| `roboflow_phone_usage_dms` | 30 MB | 997 | 999 | ✅ (`phone usage` 1-class) |
| `rf_phone_using_fyp` | 264 MB | 4,216 | ~4,218 | ✅ (2026-04-15, Phone) |
| `rf_using_phone_unimonitor` | 143 MB | 3,748 | ~3,750 | ✅ (2026-04-15, **Using Phone / mobile**) |
| `rf_phone_texting_4class` | 316 MB | 3,193 | ~3,195 | ✅ (2026-04-15, using phone) |
| `rf_texting_calling_drinking` | 87 MB | 867 | ~869 | ✅ (2026-04-15, **Texting/Calling/Drinking** action labels) |
| `state_farm_distracted` | 4 KB | 0 | — | **STUB** (requires competition rules acceptance) |

### raw/shoes_detection
| Source | Size | Images | Labels | Status |
|---|---|---|---|---|
| `keremberke_ppe` | 198 MB | 11,978 | Parquet | ✅ (10 classes incl. shoes/no_shoes; needs Parquet→YOLO conversion) |
| `shoe_ppe` | 108 MB | 1,570 | 1,572 txt | ⚠️ (1 opaque class `"14"`) |
| `rf_barefoot` | 378 MB | 1,258 | 1,260 txt | ✅ (2026-04-15, **`no_shoes` — barefoot PPE negative**) |
| `rf_footwear_sandal_shoe` | 119 MB | 1,967 | 1,969 txt | ✅ (2026-04-15, **Sandal / Shoe** — sandal violation negative) |
| `rf_feet_segmentation` | 835 MB | 17,888 | 17,890 txt | ✅ (2026-04-15, foot seg masks — shoe_region pipeline) |

---

## Downloading Datasets (via MCP tools — no scripts)

No bootstrap script. Fetch via MCP servers (credentials in `.mcp.json` / user MCP settings).

- **Roboflow Universe**: `mcp__roboflow__download_universe_dataset` — args `universe_workspace`, `universe_project`, `version_number`, `location` (absolute path), `model_format` (default `yolov8`). Use `mcp__roboflow__search_universe` / `list_versions` first if slug/version unknown.
- **Kaggle**: `mcp__kaggle__download_dataset` returns signed URL; `curl -L` it then `unzip` into `location`.
- **Hugging Face**: `mcp__huggingface__hub_repo_search` → `hub_repo_details`, then `huggingface-cli download` inside `uv run` for snapshot.
- **Direct URL** (GitHub releases, Zenodo, academic): `curl -L -o` + `unzip` in Bash.

Before re-downloading a STUB, clear the placeholder: `rm -rf raw/<cat>/<source>` then re-fetch.

---

## Dataset Source Registry

Canonical public source-of-truth for every `raw/` folder. All reproducible from scratch.

### Fire / smoke detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/fire_detection/d_fire` | kaggle `sayedgamal99/smoke-fire-detection-yolo` | CC-BY-4.0 | 21k imgs |
| `raw/fire_detection/fasdd_cv` | kaggle `yuulind/fasdd-cv-coco` | CC-BY-4.0 | ~95k imgs, COCO JSON |
| `raw/fire_detection/zenodo_indoor_fire` | url `https://zenodo.org/api/records/15826133/files/Indoor%20Fire%20Smoke.zip/content` | CC-BY-4.0 | 10k imgs |
| `raw/fire_detection/roboflow_fire_seg` | roboflow `fire-nxmk4/fire-smoke-detection-elt66` v1 | MIT | seg masks for p03 aug |
| `raw/fire_detection/industrial_hazards` | kaggle `vigneshnachu/industrial-hazards-detection` | CC-BY-4.0 | 1343 imgs, yolo |

### Helmet / PPE detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/helmet_detection/hardhat_vest_v3` | kaggle `muhammetzahitaydn/hardhat-vest-dataset-v3` | CC-BY-4.0 | 22k imgs |
| `raw/helmet_detection/sfchd` | github `https://github.com/lijfrank/SFCHD-SCALE` (repo has Baidu Netdisk link + extraction code; **no direct CLI download**) | CC-BY-4.0 | 12k imgs, 7 classes |
| `raw/helmet_detection/sh17_ppe` | kaggle `mugheesahmad/sh17-dataset-for-ppe-detection` | CC-BY-NC-SA-4.0 | 8k Pexels imgs, 17 classes |
| `raw/helmet_detection/hard_hat_workers` | kaggle `andrewmvd/hard-hat-detection` | PDDL | VOC XML — needs conversion |
| `raw/helmet_detection/construction_safety` | roboflow `custom-model/construction-safety-nvqbd` v2 | CC-BY-4.0 | 2777 imgs, 10 classes |
| `raw/helmet_detection/roboflow_hardhat` | roboflow `yus-space/construction-site-safety-helmet` v1 | CC-BY-4.0 | 18k imgs, head/helmet/person |
| `raw/helmet_detection/roboflow_vest_helmet_9k` | roboflow `uhhh/vest-helmet-wlbch` v1 | CC-BY-4.0 | 9526 imgs, 2-class |
| `raw/helmet_detection/roboflow_ppe_6class` | roboflow `dataset-ppe/ppe-vest-helmet` v1 | MIT | 3481 imgs, 6 classes (Boots/Gloves/Goggles/Helmet/Person/Vest) |
| `raw/helmet_detection/shlokraval_ppe_yolov8` | kaggle `shlokraval/ppe-dataset-yolov8` | Apache-2.0 | 44k imgs, 14 classes incl. NO-* negatives |
| `raw/helmet_detection/rf_siabar_ppe` | roboflow `siabar/ppe-dataset-for-workplace-safety` v2 | CC-BY-4.0 | 1613 imgs, 8 classes incl. Boots + Ear-protection |
| `raw/helmet_detection/rf_chemical_ppe_11class` | roboflow `data2-4uqki/chemical-8oaqi` v3 | CC-BY-4.0 | 1180 imgs, 11-class chemical PPE incl. Faceshield/NO-Shoes |
| `raw/helmet_detection/shwd` | github `https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset` (git clone — only 21 sample imgs; full 7,581-img set is Baidu-only per upstream README) | MIT | Chinese construction helmet |

### Harness detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/harness_detection/rf_body_harness` | roboflow `construction-images/body_harness` v2 | CC-BY-4.0 | 2069 imgs, safety_harness/worker |
| `raw/harness_detection/rf_safety_harness_v2` | roboflow `techling/safety_harness_v2` v1 | CC-BY-4.0 | 1234 imgs, anchored/non_anchored/no_harness |

### Apron detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/apron_detection/rf_apron` | roboflow `knowledgeflex/apron-detection` v6 | CC-BY-4.0 | 1379 imgs, Wearing-Apron |

### Mask detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/mask_detection/mask_type_det` | kaggle `takiaibnath19/mask-type-detection-dataset` | CC0 | 276 imgs, VOC XML, mask subtypes |
| `raw/mask_detection/rf_n95` | roboflow `yolo-uwp4e/n95-sdn3f` v2 | CC-BY-4.0 | 2461 imgs, N95/ear/earplug |
| `raw/mask_detection/rf_mask_3class` | roboflow `data-labeling/face-mask-detection-euti3` v2 | CC-BY-4.0 | 2307 imgs, 3-class incl. mask_weared_incorrect |
| `raw/mask_detection/rf_mask_3k` | roboflow `cat-vs-dog-classification/face-mask-detection-j3gwp` v1 | CC-BY-4.0 | 7182 imgs, with/without mask |

### Glove detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/glove_detection/datacluster_gloves` | kaggle `dataclusterlabs/gloves-dataset-covid-safety-wear` | CC0 | 236 imgs, VOC XML |
| `raw/glove_detection/rf_hand_gloves` | roboflow `terry-kanana-21-gmail-com/hand-gloves-ik2gg` v1 | CC-BY-4.0 | 945 imgs, HAND-GLOVES |

### Smoking detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/smoking_detection/rf_cigarette_5k` | roboflow `cigarette-c6554/cigarette-ghnlk` v7 | CC-BY-4.0 | 10,291 imgs, cigarette/face — top-rated smoking set |

### Ear protection
| dest | source | license | notes |
|---|---|---|---|
| `raw/ear_protection/rf_earplug` | roboflow `garyspace/earplug-jocg0` v1 | Public Domain | 780 imgs, 2-class earplug |

### Zone intrusion
| dest | source | license | notes |
|---|---|---|---|
| `raw/zone_intrusion/rf_intrusion_3k` | roboflow `ardhas-workspace/intrusion-vstvg` v2 | CC-BY-4.0 | 3141 imgs, intrusion_detection |
| `raw/zone_intrusion/rf_intrusion_face_3k` | roboflow `none-n1imd/intrusion-7t7sz` v2 | CC-BY-4.0 | 3366 imgs, face/not_face/person |

### Shoes detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/shoes_detection/keremberke_ppe` | hf `keremberke/protective-equipment-detection` | AGPL-inherited | 11978 imgs, 10 classes incl. shoes/no_shoes, Parquet |
| `raw/shoes_detection/shoe_ppe` | roboflow `harnessafesite/shoe_ppe` v1 | CC-BY-4.0 | 1570 imgs, opaque 1-class |
| `raw/shoes_detection/rf_barefoot` | roboflow `mohamed-nihal/barefoot` v1 | CC-BY-4.0 | 1258 imgs, `no_shoes` (barefoot) |
| `raw/shoes_detection/rf_footwear_sandal_shoe` | roboflow `ariel-hzk0a/footwear-2ws0y` v1 | CC-BY-4.0 | 1967 imgs, Sandal/Shoe |
| `raw/shoes_detection/rf_feet_segmentation` | roboflow `unlimited-robotics/feet-segmentation-q30ii` v5 | CC-BY-4.0 | 17888 imgs, foot seg masks |

### Fall detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/fall_detection/coco_keypoints` | kaggle `asad11914/coco-2017-keypoints` | CC-BY-4.0 | keypoints for pose |
| `raw/fall_detection/roboflow_fall` | roboflow `roboflow-universe-projects/fall-detection-ca3o8` | CC-BY-4.0 | |
| `raw/fall_detection/cctv_fall` | kaggle `simuletic/cctv-incident-dataset-fall-and-lying-down-detection` | CC-BY-4.0 | |
| `raw/fall_detection/ur_fall` | url `http://fenix.ur.edu.pl/~mkepski/ds/data/Fall_DVDSD.zip` | research-only | academic |
| `raw/fall_detection/roboflow_person_fall_8k` | roboflow `xzjs/person-fall-ewpyk` | CC-BY-4.0 | 7947 imgs, fall/nofall |

### Phone detection
| dest | source | license | notes |
|---|---|---|---|
| `raw/phone_detection/fpi_det` | github `https://github.com/KvCgRv/FPI-Det` (repo has Baidu Netdisk link, code `Mofo`; **no direct CLI download**) | MIT | 22,879 imgs, face+phone synced |
| `raw/phone_detection/phone_call_usage` | roboflow `phoneusagedetection/phone-call-usage` | CC-BY-4.0 | |
| `raw/phone_detection/roboflow_mobile_phone_ineuron` | roboflow `ineuron-8bdse/mobile-phone-b83c7` v1 | MIT | 1523 imgs |
| `raw/phone_detection/roboflow_phone_usage_dms` | roboflow `dms-ejczp/phone-usage-detection-a6oou` v1 | CC-BY-4.0 | 997 imgs |
| `raw/phone_detection/rf_phone_using_fyp` | roboflow `fyp-odgmu/phone-using-dhjqe` v3 | CC-BY-4.0 | 4216 imgs, Phone |
| `raw/phone_detection/rf_using_phone_unimonitor` | roboflow `unimonitor-03qbp/using-phone-pqaut` v1 | CC-BY-4.0 | 3748 imgs, Using Phone / mobile |
| `raw/phone_detection/rf_phone_texting_4class` | roboflow `cv2-kyb30/mobile-phone-texting` v4 | CC-BY-4.0 | 3193 imgs, using phone |
| `raw/phone_detection/rf_texting_calling_drinking` | roboflow `hand-rules/texting-calling-drinking` v1 | CC-BY-4.0 | 867 imgs, **Texting/Calling/Drinking action labels** |
| `raw/phone_detection/state_farm_distracted` | kaggle competition `state-farm-distracted-driver-detection` | competition-use | requires rules acceptance |

### COCO (zone intrusion eval)
| dest | source | license | notes |
|---|---|---|---|
| `raw/_coco_val` | hf dataset `lmms-lab/COCO` | CC-BY-4.0 | val2017 for zone-intrusion eval |

### Internal-only (no public source, populate on-site)
- `site_collected/<category>/` — Nitto Denko on-site footage; collected directly from factory cameras, no public equivalent.
- `site_collected/test_video` — Nitto demo video for end-to-end validation.
- `training_ready/` — derived; rebuild via `core/p00_data_prep/` from `raw/` + `site_collected/`.
- `raw/fire_detection/fire_cctv`, `fire_smoke_datacluster` — previously Datacluster samples; if public equivalent needed, search kaggle `dataclusterlabs/fire-detection`.
- `raw/fall_detection/fall_detection_imgs`, `raw/phone_detection/action_recognition`, `raw/phone_detection/mobile_phone` — internal curated sets, no public source.

---

## Pretrained Weights (reference only — not in dataset_store)

Location: `ai/pretrained/` (outside this tree).

Public weights (re-downloadable):
- `yolox_{nano,tiny,s,m,l}.pth` — [YOLOX releases](https://github.com/Megvii-BaseDetection/YOLOX/releases)
- `scrfd_500m.onnx` — [SCRFD / InsightFace](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- `ultralytics_cache/` — auto-downloaded by Ultralytics on first use

Internal (no public source):
- `nitto_denko/` — Nitto-specific fine-tuned baselines
- `smart_parking/` — smart_parking project checkpoints

---

## Action Items

- **Baidu-only (need manual fetch via Chinese Netdisk)**:
  - `helmet_detection/sfchd` — 12k imgs, chemical plant, 7 classes
  - `helmet_detection/shwd` — 7,581-img full set (only 21 samples from git)
  - `phone_detection/fpi_det` — 22,879 imgs, face+phone synced
- **Competition acceptance**: `state_farm_distracted` (Kaggle rules)
- **Academic mirror**: `ur_fall`
- **Annotation conversion**: `coco_keypoints` (JSON→YOLO), `fasdd_cv` (COCO→YOLO), `keremberke_ppe` (Parquet→YOLO), `hard_hat_workers` (VOC→YOLO)
- **PPE gaps — no public data, needs site_collected + SAM3 auto-label:**
  - **Chemical-resistant apron / split apron** (Phase 2 Model J) — only generic apron class available
  - **Glove subtypes** with detection boxes: chem-resistant / cut-resistant / insulated (Phase 2 Model E)
  - **Mask subtypes** with detection boxes: N95 / respirator / face_shield_with_mask (Phase 2 Model D)
  - **Harness hook-at-altitude** — have anchored/non_anchored but may need factory-specific examples
