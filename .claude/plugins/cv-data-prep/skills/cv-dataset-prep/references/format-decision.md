# Format Decision

| Source has… | `format:` | Notes |
|---|---|---|
| `data.yaml` + `{train,valid,test}/labels/*.txt` | `yolo` | Default Roboflow export |
| `_annotations.coco.json` + images | `coco` | Add `annotations_file:` if filename differs |
| `voc_labels/*.xml` + `images/` | `voc` | Add `voc_annotations_dir:` and `voc_images_dir:` |
| Both VOC and YOLO | `voc` | Prefer VOC — class names are explicit, can't mis-order |

## YOLO with numeric class names

If `data.yaml` has `names: ['0', '1']` or is missing, add `source_classes:` in the correct id order:

```yaml
source_classes: ["fire", "smoke"]
class_map:
  "fire": "fire"
  "smoke": "smoke"
```

## Unsupported formats

p00 does not handle Parquet, TFRecord, or LabelMe JSON. Drop the source rather than hacking a parser.
