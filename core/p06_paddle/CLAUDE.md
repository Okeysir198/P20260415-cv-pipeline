# CLAUDE.md — core/p06_paddle/

Paddle is **not a backend** in `core/p06_training`. It's a separate package
that drives upstream `ppdet.engine.Trainer` directly inside `.venv-paddle/`.
Convergence with the rest of the pipeline happens at ONNX, not `nn.Module`.

## Entry points (run from `.venv-paddle/`, never `uv run`)

| File | Purpose |
|---|---|
| `train.py` | `Trainer(cfg, mode="train").train()` + writes `best.pdparams` + skeleton observability tree |
| `eval.py` | `Trainer(cfg, mode="eval").evaluate()` on a `.pdparams` checkpoint |
| `export.py` | `paddle2onnx` wrapper → `model.onnx` |
| `_translator.py` | Our `06_training_paddle_*.yaml` → upstream ppdet config patches |

## Workflow

```bash
bash scripts/setup-paddle-venv.sh                              # one-time
.venv-paddle/bin/python core/p06_paddle/train.py \
  --config configs/_test/06_training_paddle_det.yaml
.venv-paddle/bin/python core/p06_paddle/export.py \
  --config <same.yaml> --checkpoint <run>/best.pdparams --out <run>/model.onnx
# From here, main venv handles eval/infer/demo via standard ORT path:
uv run core/p08_evaluation/evaluate.py --model <run>/model.onnx --config <05_data>.yaml
```

## v1 scope

Detection only — `paddle-picodet-{s,m,l}`, `paddle-ppyoloe[-plus]-{s,m,l,x}`.
Add new task families by appending to `_TASK_DISPATCH` in `train.py`; each
family is a thin driver around its upstream Trainer/Engine. No torch dependency.

## Gotchas

- **PaddleDetection's pip wheel strips `configs/` + `tools/`** — `setup-paddle-venv.sh` clones the upstream repo to `.venv-paddle/PaddleDetection/`. `_find_ppdet_config` searches there first.
- **Train-from-scratch on small datasets explodes** — paddle reports as `Cannot allocate 4.5 PB` (integer underflow on shape). Always use `model.pretrained: <url>` for CI smokes.
- **YOLO labels → COCO JSON** — auto-converted by `utils.paddle_bridge.yolo_to_coco`, cached as `<dataset_root>/<split>_paddle_coco.json`. `image_dir` for ppdet's `COCODataSet` is the dataset root (file_name in our JSON is already prefixed with `<split>/images/`).
- **Hardware**: paddle 3.3.x doesn't support sm_120 (RTX 50xx Blackwell). Tests skip-with-reason on these GPUs until upstream paddle ships a release. Verified working on sm_90 and below.
