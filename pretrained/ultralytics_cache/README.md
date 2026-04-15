# Ultralytics Weight Cache

Auto-downloaded YOLO weights from Ultralytics (YOLOv8 / YOLO11 variants).
These are **not** tracked in git (see `.gitignore` `*.pt` rule) — they are
fetched on first use by `ultralytics.YOLO("yolov8n.pt")` etc. and cached
here for repeated use.

Moved from the repo root on 2026-04-13 to de-clutter the top level.

**License note:** Ultralytics YOLOv8 / YOLO11 weights are AGPL-3.0. Use only
for internal experimentation / baselining. Do not bake these checkpoints
into redistributed products — retrain on an Apache-2.0 backbone (YOLOX,
D-FINE, RT-DETR, RF-DETR) for shipping.
