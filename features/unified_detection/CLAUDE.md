# unified_detection

**Type:** Detection (multi-source, multi-task) | **Training:** Fine-tune (19 classes)

Single shared-backbone detector covering Phase 1 + Phase 2 detection targets. One forward pass per camera replaces per-task models; per-task features become reference/legacy once R1 is validated.

## Class taxonomy (19 classes, sequential IDs)

| ID | Class | Active in R1? |
|---:|---|---|
| 0 | person | ✅ |
| 1 | fallen_person | ✅ |
| 2 | fire | ✅ |
| 3 | smoke | ✅ |
| 4 | phone_usage | ✅ |
| 5 | helmet | ✅ |
| 6 | no_helmet | ✅ |
| 7 | nitto_hat | ✅ (rare) |
| 8 | safety_shoes | ✅ |
| 9 | no_safety_shoes | ✅ |
| 10 | mask | ✅ |
| 11 | no_mask | ✅ |
| 12 | n95 | ✅ |
| 13 | gloves | ✅ |
| 14 | apron | ✅ |
| 15 | harness | ✅ |
| 16 | no_harness | ✅ |
| 17 | harness_hooked | ✅ |
| 18 | harness_unhooked | ✅ |

Phase 2 unaddressed classes (no data yet): `safety_glasses`, `goggles`, `face_shield`, `chemical/cut/insulated_gloves`, `chemical/split_apron`, `forklift`, `pallet_jack`. Added later via HF `num_labels` resize + `ignore_mismatched_sizes=True` — standard pattern, no scratch retraining.

## Dataset

Built in two stages:
1. **`scripts/merge_unified_detection_training_ready.py`** — merges 12 sources, copies files (no symlinks), prefixes filenames with source key (`fire__abc.jpg`)
2. **`scripts/dedup_split.py --name unified_detection --thresh 6`** — pHash dedup + stratified re-split (eliminates Roboflow augment-then-split leakage)

Final state: `dataset_store/training_ready/unified_detection/`

| Split | Images |
|---|---:|
| train | 103,551 |
| val | 14,458 |
| test | 11,535 |
| **Total** | **129,544** |

**Quality**:
- Cross-split duplicate pairs: **0** (verified at hamming ≤ 6)
- Malformed labels removed: **243** (out-of-bounds coords, sub-pixel boxes, invalid sizes)
- 42.9% of pre-dedup images were augmented siblings — without dedup, val mAP would be inflated ~50%
- 12 GB on disk

**Per-class counts**: see `dataset_store/training_ready/unified_detection/DATASET_REPORT.md`. Highlights:
- Most numerous: `safety_shoes` 49k, `helmet` 43k, `person` 21k
- Rarest active: `harness_hooked` 300, `nitto_hat` 926, `gloves` 1.3k

**Known limitations** (documented, accepted for R1):
- 6 classes have no test instances (helmet, nitto_hat, apron, no_harness, harness_hooked, harness_unhooked) — Roboflow leakage so severe that ~all source test images had train siblings → dedup moved them to train. Evaluate these on val mAP instead.
- 3 classes have thin test (<200 instances): no_helmet, n95, gloves

## Files

```
features/unified_detection/
├── CLAUDE.md
├── configs/
│   ├── 05_data.yaml                    # 19 classes, per-source registry
│   ├── 06_training_dfine_n.yaml        # 4M baseline
│   ├── 06_training_dfine_m.yaml        # 31M main R1
│   ├── 06_training_rtdetr_r50.yaml     # 42M alt arch
│   └── 10_inference.yaml
├── code/                               # empty (no custom code needed for R1)
├── samples/, notebooks/, tests/
└── runs/, eval/, export/, predict/, release/

scripts/
├── merge_unified_detection_training_ready.py
└── qa_unified_detection.py

dataset_store/training_ready/unified_detection/
├── data.yaml
├── valid_classes.json        # 129,544 entries: filename → source's valid class IDs
├── splits.json
├── DATASET_REPORT.md
├── qa/
│   ├── qa_report.md
│   ├── removal_candidates.json
│   └── gallery/<class>/      # 16 sample images per class with GT boxes
└── train, val, test/
    └── {images, labels}/
```

## Training recipe (R1)

All configs share the **hard-example focus** recipe (proven on safety-fire_smoke_fall):

| Knob | Value | Rationale |
|---|---|---|
| `num_classes` | 19 | sequential IDs |
| `num_queries` | 30 | factory scenes have moderate instance counts |
| `num_denoising` | 100 | HF default |
| `label_noise_ratio` | 0.5 | HF default — helps generalization across heterogeneous source labels |
| `focal_loss_alpha` | 0.9 | heavier weight on rare positives |
| `focal_loss_gamma` | 3.0 | down-weights easy preds 10× |
| `weight_loss_vfl` | 2.0 | cls loss 2× vs bbox/giou |
| `matcher_class_cost` | 4.0 | Hungarian prefers right class |
| `checkpoint.metric` | `eval_map` | COCO challenge mAP@[.5:.95] — production-relevant tightness |

Per-arch deltas:
- `dfine_n`: bs=32, lr=1e-4 (parity baseline)
- `dfine_m`: bs=16, lr=1e-4 (main R1 candidate — more data justifies bigger backbone)
- `rtdetr_r50`: bs=16, lr=5e-5, bf16=true (alt arch)

## Per-source loss masking — **NOT implemented for R1**

In theory, each image should only contribute cls loss for classes its source actually annotates (a fire image has no helmet labels, so don't penalize the model for predicting a helmet). In practice:
- DETR's "no_object" pressure on unmatched queries is gated by `eos_coefficient: 0.0001` (essentially off)
- The remaining cross-class softmax suppression is mild
- Expected `person` recall drop in cross-source scenes: ~10-15% (worst case)
- Other classes: 2-5% mAP drop

Skipped to keep R1 simple. `valid_classes.json` is generated and available for R2 if needed. Re-evaluate after R1 results.

## R2 — contingent, per-class targeted fine-tune

**Trigger**: any class drops >10% mAP vs its per-task champion after R1.

**Approach** — load from unified dataset, filter on-the-fly:

```python
# features/<task>/code/unified_slice.py (write only if R2 is triggered)
class UnifiedSliceDataset(YOLOXDataset):
    """Read unified_detection, keep only images with target classes, remap IDs."""
    def __init__(self, *args, unified_classes: list[int],
                 local_remap: dict[int, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.img_paths = [p for p in self.img_paths
                          if self._has_target_class(p, unified_classes)]
        self._unified_classes = set(unified_classes)
        self._remap = local_remap

    def _load_label(self, img_path):
        raw = super()._load_label(img_path)
        mask = np.isin(raw[:, 0], list(self._unified_classes))
        out = raw[mask]
        out[:, 0] = np.array([self._remap[int(c)] for c in out[:, 0]])
        return out
```

Wire via `data.dataset_class: features.<task>.code.unified_slice.UnifiedSliceDataset`. No data duplication, no `core/` changes, no symlink views. Reuses dedup-clean QA-clean unified data.

R2 recipe sketch (only if triggered):
- Pretrained from unified R1 best ckpt
- Discriminative LR: head 1e-3, backbone 1e-5
- 10-15 epochs
- Same hard-example focal recipe

## Pipeline checklist

- [x] Scaffold from `_TEMPLATE`
- [x] `scripts/merge_unified_detection_training_ready.py` — 12 sources merged
- [x] pHash dedup (cross-split leakage = 0)
- [x] QA (243 malformed labels removed)
- [x] `data.yaml`, `valid_classes.json`, `splits.json`, `DATASET_REPORT.md`, `qa/gallery/`
- [x] `configs/05_data.yaml` — 19 classes
- [x] `configs/06_training_{dfine_n,dfine_m,rtdetr_r50}.yaml` — hard-example recipe
- [ ] R1 training (start when ready)
- [ ] Evaluate per-class mAP vs per-task champions
- [ ] R2 (only if needed) — UnifiedSliceDataset + per-class fine-tune

## Gotchas

- **Roboflow leakage is severe in our sources** — 42.9% of pre-dedup images were augmented siblings. Always run pHash dedup before training; never skip even for "smoke test" runs.
- **6 classes have no test data** — known limitation; evaluate on val for those (helmet, nitto_hat, apron, no_harness, harness_hooked, harness_unhooked).
- **`unified_detection/` is the single source of truth** — R2 reads from it via class-filter wrapper. Do NOT pre-build symlink directories per task.
- **`raw/<source>/` for Phase 1 sources has been DELETED** (fire, fall, helmet, shoes, phone) — they were duplicates of `training_ready/<source>/`. Phase 2 raw (apron, glove, harness, mask, n95) is retained as the sole training source for those classes.
- **Re-running merge requires Phase 2 raw + Phase 1 training_ready/** — both still on disk. Merge is idempotent: rerun anytime sources change.
- **DETR-family `bf16` rules** — D-FINE: `bf16: false` mandatory (DFL stalls). RT-DETRv2: `bf16: true amp: false` safe. YOLOX: not used here (no native unified head).
- **`num_classes: 19` everywhere** — `05_data.yaml`, all `06_training_*.yaml`, and HF model `config.json` after training. Mismatch silently re-inits the cls head.
