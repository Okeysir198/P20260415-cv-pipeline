# safety-poketenashi_no_handrail

Pose-rule feature: detects when a worker traversing stairs is **not** holding
the handrail. Uses DWPose COCO-17 keypoints + site-configured handrail zone
polygons. The rule fires when both wrists are visible and farther than
`hand_to_railing_px` from every configured handrail zone.

Split out of the umbrella `features/safety-poketenashi/` orchestrator. See
`CLAUDE.md` for the full pipeline + deployment recipe.

## Quick Start

```bash
# Smoke test on sample images (default DWPose backend)
uv run features/safety-poketenashi_no_handrail/code/predictor.py --smoke-test

# Run on a video
uv run features/safety-poketenashi_no_handrail/code/predictor.py \
  --video features/safety-poketenashi_no_handrail/samples/<clip>.mp4

# Latency benchmark
uv run features/safety-poketenashi_no_handrail/code/benchmark.py

# Unit tests (synthetic keypoints — no GPU/data needed)
uv run -m pytest features/safety-poketenashi_no_handrail/tests/ -v
```

## Configs

- `configs/05_data.yaml` — `dataset_name: safety_poketenashi_no_handrail`,
  single rule-output viz class.
- `configs/10_inference.yaml` — `pose_rules.no_handrail.{hand_to_railing_px,
  handrail_zones}` + ByteTrack stub. The rule is **disabled** until
  `handrail_zones:` is populated for the deployment site.

## Files

```
configs/05_data.yaml
configs/10_inference.yaml
code/_base.py                  PoseRule + RuleResult (self-contained)
code/handrail_detector.py      HandrailDetector (verbatim copy from umbrella)
code/predictor.py              Thin orchestrator: DWPose + this rule + CLI
code/benchmark.py              Latency benchmark on sample images
tests/test_handrail.py         Synthetic-keypoint unit tests
samples/                       Site clips (gitignored)
eval/                          Smoke-test outputs (gitignored)
```
