# Format Decision — YOLO vs COCO vs VOC

`core/p00_data_prep/parsers/` has three parsers. Pick based on what the source actually ships with.

## Cheat sheet

| Source has… | Use parser | `format:` value | Notes |
|---|---|---|---|
| `data.yaml` + `{train,valid,test}/{images,labels}/*.txt` | `yolo.py` | `yolo` | Default Roboflow export. Class names come from `data.yaml:names`. |
| `{train,valid,test}/_annotations.coco.json` + images | `coco.py` | `coco` | HF datasets (e.g. keremberke), Roboflow COCO export. |
| `voc_labels/*.xml` + `images/` | `voc.py` | `voc` | Pascal VOC. Class names in each XML `<name>` tag. |

## When a source ships multiple formats

Some datasets (e.g. `sh17_ppe`) ship **both** VOC and YOLO labels. In that case prefer **VOC** because:

- Class names are explicit in every XML file — you can't mis-order them.
- YOLO ids depend on a `classes.txt` or `data.yaml` that may be missing or wrong.

Specify which to use via source-specific fields:

```yaml
- name: "sh17_ppe"
  path: "../../../dataset_store/raw/helmet_detection/sh17_ppe"
  format: "voc"
  voc_annotations_dir: "voc_labels"   # subdir containing .xml files
  voc_images_dir: "images"            # subdir containing jpg/png
```

## YOLO with non-standard class names

If a YOLO dataset's `data.yaml` has numeric class names (e.g. `names: ['0', '1']`) or is missing entirely, pass `source_classes` explicitly:

```yaml
- name: "some_source"
  path: "..."
  format: "yolo"
  source_classes: ["fire", "smoke"]   # ordered list matches class ids 0,1
  class_map:
    "fire": "fire"
    "smoke": "smoke"
```

## COCO specifics

COCO parsers look for `_annotations.coco.json` by default. If the filename differs, set:

```yaml
annotations_file: "instances_train2017.json"
```

## Unsupported formats

p00 does **not** handle:
- **Parquet** (HF datasets sometimes use this). Workaround: convert to YOLO offline or use the HF loader in `p05_data/`.
- **TFRecord**. No workaround — drop or convert externally.
- **LabelMe JSON** (per-image). Convert to VOC XML first.

If a source is unsupported, drop it from v1 rather than hacking a parser.
