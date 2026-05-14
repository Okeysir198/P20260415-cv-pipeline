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

## Registering a new architecture

`@register_model("<arch>")` from `core.p06_models.registry` accepts either a builder function (preferred for any arch needing extra setup — processor wiring, kwarg filtering) or a class. `build_model(config)` looks up `config["model"]["arch"]` in `MODEL_REGISTRY` and calls the registered callable with the full config dict. Auto-discovery happens in `core/p06_models/__init__.py` — add a soft-import there so the decorator runs at package import:

```python
# core/p06_models/my_arch.py
from core.p06_models.registry import register_model
from core.p06_models.base import DetectionModel

@register_model("my-arch")
def build_my_arch(config: dict) -> DetectionModel:
    cfg = config["model"]
    return MyDetector(num_classes=cfg["num_classes"], ...)
```

```yaml
# features/<name>/configs/06_training_my_arch.yaml
model:
  arch: my-arch
  num_classes: 3
```

See `hf_model.py::build_hf_model` (`@register_model("hf_detection")`, line 319) for a full factory example; `yolox.py` for the class form.

## Paddle is not in this registry

Paddle archs (PicoDet, PP-YOLOE) live in `core/p06_paddle/` and run in `.venv-paddle/`. See `core/p06_paddle/CLAUDE.md`.

## HF wrappers

`HFDetectionModel` (`hf_model.py`) wraps any HF `ForObjectDetection` model so HF Trainer can backprop.

### GPU memory invariant — eval-mode `ModelOutput` strip (load-bearing)

`HFDetectionModel.forward` (`hf_model.py:72-143`) replaces heavy `ModelOutput` fields with a 0-element placeholder tensor at eval time (the strip loop lives at `hf_model.py:113-142`). Affected fields: `encoder_last_hidden_state`, `encoder_hidden_states`, `encoder_attentions`, `decoder_hidden_states`, `decoder_attentions`, `cross_attentions`, `intermediate_hidden_states`, `intermediate_reference_points`, `intermediate_logits`, `intermediate_predicted_corners`, `initial_reference_points`, `init_reference_points`, `auxiliary_outputs`, `enc_topk_logits`, `enc_topk_bboxes`, `enc_outputs_class`, `enc_outputs_coord_logits`, `denoising_meta_values`. HF Trainer's `prediction_loop` accumulates every `ModelOutput` field on CPU across the full eval split before invoking `compute_metrics`; without the strip, `encoder_last_hidden_state` alone hoards ~30 MB/batch at 960² → ~38 GB CPU per eval at val=2606, and Python's allocator does not return freed pages to the OS so RAM grows ~+30 GB per epoch (verified 2026-05-04: ep1 → ep3 went 38 → 122 GB → OOM). The placeholder must be a 0-element tensor, NOT `None` — accelerate's `pad_across_processes` raises `TypeError: Unsupported types NoneType`. New HF detection wrappers MUST mirror this behaviour or the leak regresses. `compute_metrics` only consumes `loss`/`logits`/`pred_boxes`; the strip is lossless.

## Weights-only resume (continue training from a previous run)

HF Trainer's `--resume` flag restores model + optimizer + scheduler + epoch counter — it does NOT let you start a fresh training run with new hyperparameters from a saved checkpoint. For that case, set `model.pretrained` to the checkpoint directory in the training YAML:

```yaml
model:
  arch: dfine-n
  pretrained: ../runs/<previous_run>/checkpoint-N    # or the run root dir
```

`build_hf_model::_resolve_pretrained_path` (`hf_model.py`) detects path-like strings (containing `/`, `..`, `~`, or absolute) and resolves them against the config dir / CWD. It also auto-strips the `hf_model.` wrapper prefix from `pytorch_model.bin` into a temp dir before calling `from_pretrained`, so checkpoints saved by `_DetectionTrainer._save` load transparently. HF Hub repo IDs (e.g. `PekingU/rtdetr_v2_r18vd`) pass through unchanged. The config path is stashed by `train_with_hf` into `config["_config_path"]` so the helper can resolve relative-to-YAML paths.

## Pretrained weight sanity check

`check_pretrained.py` runs COCO inference on YOLOX-M, D-FINE-S, RT-DETRv2-R18 on one image and writes a side-by-side grid — use it to confirm pretrained weights load before training. See `core/CLAUDE.md` for the invocation.

## Rule

`core/` may define registries; `features/<name>/code/` may register feature-specific variants via dotted-path imports. `core/` must never import from any feature folder.
