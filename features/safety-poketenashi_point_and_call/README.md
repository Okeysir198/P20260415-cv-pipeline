# safety-poketenashi_point_and_call

Rule-based detector for the Japanese **指差呼称** (*shisa-kanko* / *yubisashi-koshō*,
"point-and-call") crosswalk safety gesture. A worker stops at the curb, points
**right** ("右ヨシ!"), points **left** ("左ヨシ!"), optionally points **front**
("前ヨシ!"), then crosses. Failing to perform the full sequence is the alert
condition.

| Field | Value |
|-------|-------|
| Task | Pose-driven temporal gesture recognition |
| Mode | 🔧 Pretrained only (v1) — no fine-tuning |
| Pose model | DWPose-L wholebody (`pretrained/safety-poketenashi/dw-ll_ucoco_384.onnx`) |
| Person detector | YOLO11n (`pretrained/access-zone_intrusion/yolo11n.pt`) |
| Per-frame head | Geometric direction classifier (right / left / front / down / none) |
| Sequence matcher | Temporal state machine over per-frame directions |

## Citations

- **MHLW anzeninfo** — Ministry of Health, Labour and Welfare *Anzen Eisei Joho
  Center* guidance on 指差呼称 as a workplace error-prevention practice.
  <https://anzeninfo.mhlw.go.jp/>
- **JR East / JR West shisa-kanko** — Japan Railways operational standard for
  point-and-call at platform edges and crossings; the reference pattern this
  feature targets ("右ヨシ・左ヨシ・前ヨシ" before crossing).
- **RTRI 1994** — Railway Technical Research Institute study (Hiroshi Sigemori
  et al.) showing point-and-call reduces error rates by ~85 % vs no
  countermeasure on a button-press task. Standard citation behind the
  industrial adoption of the practice.

## Pipeline

```
Frame ─► YOLO11n person detector ─► person crops
              │
              └─► DWPose-L (133-kpt) ─► per-person keypoints
                          │
                          └─► geometric direction classifier
                                  (shoulder + elbow + wrist angles)
                                  ─► {right, left, front, down, none}
                                          │
                                          └─► temporal sequence matcher
                                                  ─► point_and_call_complete
                                                  ─► point_and_call_missing
```

**v1 status:** pretrained-only. The per-frame direction classifier is a hand-tuned
geometric rule (no learned head) and the sequence matcher is a deterministic
state machine. Both are configured from `configs/10_inference.yaml`.

## How to run

```bash
# Orchestrator smoke test (samples/ → eval/orchestrator_smoke_test.json)
uv run features/safety-poketenashi_point_and_call/code/orchestrator.py \
  --samples features/safety-poketenashi_point_and_call/samples/ \
  --output  features/safety-poketenashi_point_and_call/eval/orchestrator_smoke_test.json

# Benchmark per-frame direction classifier on samples
uv run features/safety-poketenashi_point_and_call/code/benchmark.py --split samples

# Multi-tab Gradio demo (the safety-poketenashi_point_and_call tab loads
# configs/10_inference.yaml for thresholds + sequence config)
uv run demo
```

For the standard feature layout and the end-to-end pipeline CLI, see
[`features/README.md`](../README.md) and the [root README](../../README.md).
