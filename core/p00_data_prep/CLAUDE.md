# core/p00_data_prep — data preparation

Merges multi-source raw datasets (YOLO/COCO/VOC) into a single
`dataset_store/training_ready/<name>/{train,val,test}/` layout.

## Dedup-aware splitting (added 2026-05-03)

Splits are now produced by `core/dedup.py` when `dedup.enabled: true` (default
in newly-authored configs). Pipeline:

1. pHash every merged image (`compute_phashes`)
2. Connected components at hamming ≤ `hamming_thresh` (`build_groups`)
3. **Per-source** group → split assignment (`stratified_group_split`)
   — when `source` is in `stratify_by`, groups are partitioned by source
   and each source is split independently. This is the structural fix for
   the failure mode where a small source family is entirely absorbed into
   train because its image budget barely dents the overall target.
4. Optional `apply_max_per_group_eval` cap on val/test groups
5. `verify_no_leakage` sanity check (asserts zero cross-split pairs)

Config schema lives in `validate_dedup_config(...)` — hard-errors on unknown
keys (typo guard). See the annotated block in
`features/safety-fire_detection/configs/00_data_preparation.yaml` for the
authoritative example.

### Case study — fire_detection 2026-05-03

Pre-fix splits had **0 industrial/hazard images in val and test** despite
711 such images in the raw `industrial_hazards` source: the previous
class-only stratifier (`SplitGenerator._stratified_split`, and the same
logic mirrored in the `scripts/dedup_split.py` wrapper) only balanced
per-class box counts, so smaller sources collapsed into whatever split
their largest groups landed in. The canonical implementation now lives in
`core/p00_data_prep/core/dedup.py` (invoked by p00's main pipeline);
`scripts/dedup_split.py` is a thin wrapper for re-deduping an existing
`training_ready/<name>/` without re-running p00 from raw.

Re-running p00 with the new pipeline at `hamming_thresh: 3` produced:

```
train: 13,420 imgs (d_fire=9021, industrial=532, zenodo=3867)
val:    1,981 imgs (d_fire=1338, industrial=77,  zenodo=566)
test:   1,972 imgs (d_fire=1330, industrial=75,  zenodo=567)
cross-split leakage at hamming ≤ 3: 0
```

## Per-source + temporal-aware splitting (added 2026-05-03, recommended)

Set `dedup.split_strategy: per_source_with_temporal` to enable. This is the
new default for video-heavy / multi-source detection datasets where the
legacy `class_aware` strategy left small sources entirely absorbed into one
split (e.g. WEB 7,145/0/1 on fire_detection — 53% of train, 0% of eval).

How it works:

1. Group every image by pHash as before (`build_groups`).
2. Each pHash group is assigned to a **majority source** (alphabetical
   tie-break for ties).
3. For each source, classify every group as **VIDEO** or **STILL**:
   - VIDEO if size ≥ `temporal.min_group_size_for_video` (default 20) AND
     all filenames have monotonically increasing trailing numeric suffix
     with no gap > 10× the median step (e.g. `AoF01029.jpg`,
     `AoF01030.jpg`, `AoF01031.jpg`).
   - STILL otherwise (random scenes, Roboflow `*_jpg.rf.<hash>.jpg`
     augmentation hashes, single-image groups).
4. **VIDEO groups** are split *temporally* with a buffer gap inside the
   group: `[train_pct] | gap | [val_pct] | gap | [test_pct]`. Gap frames
   are dropped (≈5% of the group, configurable via `temporal.gap_fraction`
   and `temporal.min_gap_frames`). This tests temporal generalization
   instead of memorization of nearby frames.
5. **STILL groups** are whole-group assigned greedily (largest first) to
   the split with the largest deficit, capped at +10% above target.
6. Sources with < 7 total images get dumped entirely into train (eval would
   be statistically meaningless for them).

When to use which strategy:

| Strategy | When |
|---|---|
| `per_source_with_temporal` | Multi-source datasets, video-derived footage, datasets where one source could otherwise dominate train or eval |
| `class_aware` (legacy) | Single-source datasets, small datasets where per-source stratification removes too many degrees of freedom |

### Case study — fire_detection 2026-05-03 (per-source + temporal)

Pre-fix per-source × per-split balance under `class_aware` was severely
skewed: WEB 7,145/0/1, AoF 832/395/478, frame_video 460/0/0, etc. Several
small sources collapsed entirely into train; AoF dominated 67%/63% of the
eval splits.

After switching to `per_source_with_temporal` at `hamming_thresh: 6`,
`split_ratios: [0.7, 0.15, 0.15]`:

```
            train   val   test    tr%   val%   test%
d_fire       8177  1757   1755  70%    15%    15%
zenodo       3504   746    750  70%    15%    15%
industrial    480   103    101  70%    15%    15%
TOTAL       12161  2606   2606
fire boxes  13717  2567   2615
smoke boxes 11232  2525   2557
boxes/img    2.05  1.95   1.98
fire %      55.0% 50.4%  50.6%
cross-split leakage at hamming ≤ 6: 0
```

Temporal gap dropped ~10–50 train frames per video group; a worthwhile cost
for honest val/test that test temporal generalization.

### Tuning the gap

`temporal.gap_fraction` (default 0.05) and `temporal.min_gap_frames`
(default 5) together control the buffer. For slow-motion or high-fps video
(many adjacent frames are visually identical), bump both — e.g.
`gap_fraction: 0.10, min_gap_frames: 15`. For surveillance footage at low
fps where each frame is already different, the defaults are fine.

### Legacy entry point

`core/p00_data_prep/core/dedup.py` is the canonical module — called by
p00's main pipeline from raw. `scripts/dedup_split.py` is a thin wrapper
around the same module, kept only for re-deduping an existing
`training_ready/<name>/` directory without re-running p00 from raw.
Warning: the wrapper derives `source` from the leading underscore-token
of the filename (lossy fallback heuristic); running p00 from raw uses the
metadata-driven `source.name` from each sample's adapter for accuracy.

## VOC source config knobs (`voc_annotations_dir`, `voc_images_dir`)

The VOC parser (`parsers/voc.py`) auto-detects standard dir names
(`images/`, `Annotations/`, `labels/`) but you can override either via
source-config keys. **Required** for archives that ship YOLO `labels/`
AND VOC XML side-by-side under a non-standard dir name (e.g. sh17_ppe
ships `voc_labels/` + `labels/`). Without the override, the parser picks
`labels/` (matches `"labels"` in its candidate list), globs for `*.xml`,
finds none, and **silently contributes 0 imgs** with no error.

```yaml
sources:
  - name: sh17_ppe
    format: voc
    voc_annotations_dir: voc_labels   # ← required: real XML dir, not labels/
    voc_images_dir: images
```

Symptom this fixes: `DATASET_REPORT.md` shows a source with
`⚠️ 0 images contributed` despite raw files being present. Diagnosed
2026-05-19 — sh17_ppe contributed 0 imgs to `training_ready/helmet_detection`
for ~10 days before the parser was patched to honor these keys
(`parsers/voc.py` 2026-05-19). Any new VOC source that ships beside
a YOLO `labels/` dir must set `voc_annotations_dir` explicitly.
