# safety-poketenashi

Pose-based "manners / safety etiquette" rules for Japanese factory floors:
`hands_in_pockets`, `phone_usage`, `no_handrail`, `unsafe_stair_crossing`,
plus Gap-G3 `yubisashi` (point-and-call). All five rules consume person-pose
keypoints; three of the five additionally need hand/finger keypoints, so
this feature standardises on a **wholebody (133-kpt)** pose model.

| Field | Value |
|-------|-------|
| Task | Pose estimation + CPU rule engine |
| Recommended model | DWPose-L (Apache-2.0, 133-kpt COCO-WholeBody, 384x288 ONNX) |
| Backup | RTMW-m (smaller, 133-kpt cocktail14) |
| Detector | YOLOX-Tiny (re-used from baseline) |
| Datasets | COCO-WholeBody pretrain (no fine-tune required); custom factory video for rule-threshold calibration |

See `ai/docs/technical_study/safety-poketenashi-sota.md` and
`ai/docs/03_platform/safety-poketenashi.md` for the model survey and the
platform-level rule spec.

## Pretrained weights

Weights live in `ai/pretrained/safety-poketenashi/`:

| File | Purpose |
|---|---|
| `dw-ll_ucoco_384.onnx` | DWPose-L wholebody ONNX (primary). |
| `rtmpose-s_coco-wholebody.pth` | RTMPose reference checkpoint for fine-tune comparison. |

## Running the SOTA quality check

```bash
cd edge_ai/ai
uv run python features/safety-poketenashi/code/eval_sota.py
```

Outputs:
- `predict/yolov8n-pose_body17/<sample>.jpg` — body-17 baseline render.
- `predict/dwpose-l_wholebody133/<sample>.jpg` — wholebody render.
- `predict/summary.json` — machine-readable verdicts.
- `predict/QUALITY_REPORT.md` — write-up.

## Layout

```
features/safety-poketenashi/
  configs/        phase YAMLs (TBD — current run uses script defaults)
  code/           eval_sota.py
  samples/        10 standing-worker reference images
  predict/        SOTA quality-check outputs
  runs/ eval/ export/ release/ tests/ notebooks/
```
