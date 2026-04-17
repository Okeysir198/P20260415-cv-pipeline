# access-zone_intrusion

**Type:** Detection + zone logic | **Training:** Pretrained only (COCO person detector + polygon geometry)

## Overview

Detects when a person enters a restricted polygon zone. No custom training needed — COCO-pretrained person detector is sufficient. Zone polygons are defined per-site in `configs/10_inference.yaml`.

## Classes

| ID | Name | Source |
|---|---|---|
| 0 | person | COCO pretrained |

## Pipeline Checklist

- [x] `configs/10_inference.yaml` — yolo11m.pt + demo zone polygons configured
- [x] `code/zone_intrusion.py` — `ZoneIntrusionDetector` class implemented
- [ ] Smoke test via `app_demo` zone tab
- [ ] Define real site polygon zones for production deployment

## Benchmark Results — val split (2026-04-17, 8 labeled samples)

| Model | Accuracy | F1 | Latency ms | Notes |
|---|---|---|---|---|
| **yolox_tiny** | **1.000** | **1.000** | 6.9 | Best accuracy |
| yolo11n | 0.875 | 0.800 | 12.5 | |
| yolo11s | 0.875 | 0.800 | 8.9 | |
| yolo11m | 0.875 | 0.800 | 8.2 | |
| yolov10n | 0.875 | 0.800 | 4.6 | Fastest overall |
| yolov10s | 0.875 | 0.800 | 7.6 | |
| yolov10m | 0.750 | 0.667 | 9.9 | |
| yolox_s | 0.750 | 0.667 | 5.1 | |
| yolox_m | 0.750 | 0.667 | 6.1 | |
| yolox_l | 0.750 | 0.667 | 7.8 | |
| yolox_nano | 0.625 | 0.000 | 7.4 | Missed all intrusions |
| yolov12n | error | — | — | AAttn attr missing (Ultralytics version) |
| yolov12s | error | — | — | AAttn attr missing (Ultralytics version) |

**Recommendation:** `yolox_tiny` for highest accuracy (F1=1.0); `yolov10n` (4.6ms) for edge/latency-critical deployment.

Full results: `eval/benchmark_results.json` | `eval/benchmark_report.md`

## Key Files

```
configs/10_inference.yaml   — model path + zone polygon definitions
code/zone_intrusion.py      — ZoneIntrusionDetector(config_path) → detect(image_bgr) / draw()
code/benchmark.py           — pretrained model benchmark script
eval/benchmark_results.json — raw benchmark output
```

## Notes

- Zone polygons are site-specific — must be reconfigured per deployment location
- yolov12 models fail with current Ultralytics version due to `AAttn` attribute missing; upgrade Ultralytics to fix
- No fine-tuning ever needed unless a non-person class (vehicle, forklift) is required
