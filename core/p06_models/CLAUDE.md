# CLAUDE.md — core/p06_models/

Library-only phase. `build_model(config)` dispatches on `config["model"]["arch"]` to the architecture registered under that key. New architectures register themselves via decorator — no edit to `build_model` needed.

## Registry pattern

```python
# core/p06_models/<my_arch>.py
from core.p06_models.registry import register_model

@register_model("my-arch")
class MyDetector(nn.Module): ...
```

Separate registries (see `core/CLAUDE.md` for the full table):

| Registry | File | Used by |
|---|---|---|
| `@register_model` | `registry.py` | yolox, timm, hf_detection, hf_classification, hf_segmentation, hf_keypoint, paddle (picodet, pp-yoloe, pp-lcnet, pp-hgnet, pp-mobilenetv3, pp-liteseg, pp-mobileseg, pp-tinypose) |
| `@register_pose_model` | `pose_registry.py` | rtmpose, mediapipe_pose |
| `@register_face_detector` / `@register_face_embedder` | `face_registry.py` | scrfd, mobilefacenet |

## Paddle archs — lazy imports

Paddle archs (`PicoDet`, `PPYOLOE`, `PPLCNet`, `PPLiteSeg`, `PPTinyPose`, …) live alongside torch/HF archs in this directory but **must not import `paddle` at module top**. The decorator must register a class whose `__init__` does `import paddle` (not the module body), so:

- `core.p06_models` import-time on the main venv stays paddle-free — registries register the class object, no paddle wheel touched.
- Only the training/inference subprocess running under `.venv-paddle/bin/python` actually imports paddle, when `build_model()` instantiates the class.

Same pattern as the YOLOX-official adapter (`yolox.py::_OfficialYOLOXAdapter`) — gated behind a sibling venv, lazy import inside `__init__`.

Checkpoints from paddle archs are paddle-native `.pdparams` + `.pdiparams` pairs; the model class exposes `.save(path)` / `.load(path)` that wrap paddle's I/O. Conversion to `.onnx` is done by `paddle2onnx` from `.venv-paddle/` so downstream `p09`/`p10` consumers stay framework-neutral.

## Pretrained weight sanity check

`check_pretrained.py` runs COCO inference on YOLOX-M, D-FINE-S, RT-DETRv2-R18 on one image and writes a side-by-side grid — use it to confirm pretrained weights load before training. See `core/CLAUDE.md` for the invocation.

## Rule

`core/` may define registries; `features/<name>/code/` may register feature-specific variants via dotted-path imports. `core/` must never import from any feature folder.
