# scripts/ — repo helpers

Bootstrap scripts for sibling venvs that can't share the main `.venv/` due to
pin conflicts. Each lives in its own directory at repo root and is created
idempotently — re-running the bootstrap is a no-op when the venv already
exists.

- `setup-export-venv.sh` → `.venv-export/` — `optimum[onnxruntime]` quantized
  ONNX export. Pins `transformers<4.58`, conflicts with main venv's git
  transformers.
- `setup-notebook-venv.sh` → `.venv-notebook/` — DETR-family reference
  notebooks (`notebooks/detr_finetune_reference/`). Pins `albumentations==1.4.6`
  for byte-for-byte parity with qubvel's published notebooks.
- `setup-yolox-venv.sh` → `.venv-yolox-official/` — official Megvii YOLOX
  package alongside the in-repo custom YOLOX. Selected at runtime via
  `model.impl=official`.
- `setup-paddle-venv.sh` → `.venv-paddle/` — native PaddlePaddle backend
  (PaddleDetection / PaddleClas / PaddleSeg + paddle2onnx). Paddle's
  `paddlepaddle-gpu` wheel bundles its own CUDA 12.x runtime; the main venv
  uses CUDA 13 torch and the two cannot coexist in one venv. Run paddle-backed
  training via `.venv-paddle/bin/python core/p06_training/train.py ...`.
