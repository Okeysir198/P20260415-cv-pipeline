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
| `@register_model` | `registry.py` | yolox, timm, hf_detection, hf_classification, hf_segmentation, hf_keypoint |
| `@register_pose_model` | `pose_registry.py` | rtmpose, mediapipe_pose |
| `@register_face_detector` / `@register_face_embedder` | `face_registry.py` | scrfd, mobilefacenet |

## Paddle is not in this registry

Paddle archs (PicoDet, PP-YOLOE) live in `core/p06_paddle/` and run in `.venv-paddle/`. See `core/p06_paddle/CLAUDE.md`.

## Pretrained weight sanity check

`check_pretrained.py` runs COCO inference on YOLOX-M, D-FINE-S, RT-DETRv2-R18 on one image and writes a side-by-side grid — use it to confirm pretrained weights load before training. See `core/CLAUDE.md` for the invocation.

## Rule

`core/` may define registries; `features/<name>/code/` may register feature-specific variants via dotted-path imports. `core/` must never import from any feature folder.
