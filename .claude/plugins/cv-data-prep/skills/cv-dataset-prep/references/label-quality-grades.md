# Label Quality Grades

p02 outputs a grade per image: `good` / `review` / `bad` (default thresholds: good ≥ 0.80, bad < 0.50).

## Decision table

| Distribution | Action |
|---|---|
| `good ≥ 80%`, `bad ≤ 5%` | Accept. Ready for training. |
| `bad 5–20%` | Run p01 `--filter bad` to re-label, then re-QA. |
| `bad > 20%` | Stop. Almost always a class_map bug. Re-check class names with `inspect_source.sh`. |
| `good < 50%` | Drop the source — genuinely poor labels. |
| All classes grade `bad` | Text prompts too broad. Tune `text_prompts:` in `05_data.yaml`. |

## Threshold tuning

Lower `good` threshold to 0.70 if: small objects at distance, heavy occlusion, or SAM3 struggles with the object type (thin straps, transparent objects).

```bash
uv run core/p02_annotation_qa/run_qa.py ... \
  --override scoring.thresholds.good=0.70 scoring.thresholds.review=0.45
```

## Acting on `worst_images.json`

- Label clearly wrong → re-label with p01 (`bad` grade)
- Label correct but score low → lower thresholds or tune SAM3 prompts
- SAM3 caught a missed object → re-run p01 `--mode hybrid`
- Sample unrecoverable → use Label Studio to move to `drop` split
