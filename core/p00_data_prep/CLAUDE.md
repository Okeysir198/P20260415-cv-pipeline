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
class-only stratifier (`SplitGenerator._stratified_split` + the standalone
`scripts/dedup_split.py::_stratified_group_split`) only balanced per-class
box counts, so smaller sources collapsed into whatever split their largest
groups landed in.

Re-running p00 with the new pipeline at `hamming_thresh: 3` produced:

```
train: 13,420 imgs (d_fire=9021, industrial=532, zenodo=3867)
val:    1,981 imgs (d_fire=1338, industrial=77,  zenodo=566)
test:   1,972 imgs (d_fire=1330, industrial=75,  zenodo=567)
cross-split leakage at hamming ≤ 3: 0
```

### Legacy entry point

`scripts/dedup_split.py` is now a thin wrapper that imports from
`core/p00_data_prep/core/dedup.py`. Use it only when you already have a
`training_ready/<name>/` directory and don't want to re-run p00 from raw.
The wrapper derives `source` from the leading underscore-token of the
filename (lossy heuristic) — new datasets going through p00 use the real
`source.name` from each sample's adapter metadata.
