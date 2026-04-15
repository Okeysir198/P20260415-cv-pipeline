# Adding New Frameworks and Use Cases

This guide shows how to extend the pipeline with a new use case (no code changes) or a new model framework (minimal code). Every example references real files in the codebase.

---

## Quick Start: New Use Case (Same Model)

Adding a new use case requires only two YAML files and one experiment directory. No Python changes needed.

**Step 1 — Create configs (copy and edit)**

```bash
mkdir configs/vehicle_detection
cp features/safety-fire_detection/configs/05_data.yaml features/detect_vehicle/configs/05_data.yaml
cp features/safety-fire_detection/configs/06_training.yaml features/detect_vehicle/configs/06_training.yaml
```

Edit `features/detect_vehicle/configs/05_data.yaml` — update `dataset_name`, `path`, `names`, `num_classes`.
Edit `features/detect_vehicle/configs/06_training.yaml` — update `model.num_classes`, `logging.run_name`.

See `features/safety-fire_detection/configs/05_data.yaml` and `features/safety-fire_detection/configs/06_training.yaml` for the full schemas.

**Step 2 — Create experiment directory**

```bash
mkdir experiments/vehicle_detection
cp features/safety-fire_detection/experiments/train.py features/detect_vehicle/experiments/train.py
```

Edit the one constant at the top of `train.py`:

```python
DEFAULT_CONFIG = "features/detect_vehicle/configs/06_training.yaml"
```

The full ~40-line pattern is in `features/safety-fire_detection/experiments/train.py`.

**Step 3 — Train**

```bash
uv run features/detect_vehicle/experiments/train.py
# or with overrides:
uv run features/detect_vehicle/experiments/train.py --override training.lr=0.005
```

Config rules: no inheritance between files — each config is self-contained. Paths are relative from project root. See `configs/CLAUDE.md` for the full schema reference.

---

## Adding a HuggingFace Detection Model (1 file + config)

Any HuggingFace `ForObjectDetection` model can be added with zero Python code by registering it in `HF_MODEL_REGISTRY` and adding variant aliases.

### How it works

`core/p06_models/hf_model.py` defines `HFDetectionModel` — a thin adapter that wraps any HF model and handles I/O format conversion. The registry maps arch names to `(ModelClass, ConfigClass, default_pretrained)` tuples:

```python
# From core/p06_models/hf_model.py
HF_MODEL_REGISTRY: Dict[str, Tuple[Any, Any, str]] = {
    "dfine-s": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_s_coco"),
    "dfine-n": (DFineForObjectDetection, DFineConfig, "ustc-community/dfine_n_coco"),
    "rtdetr-r18": (RTDetrV2ForObjectDetection, RTDetrV2Config, "PekingU/rtdetr_v2_r18vd"),
    ...
}
```

`core/p06_models/dfine.py` shows the simplest variant alias pattern — the entire file is 5 lines:

```python
# core/p06_models/dfine.py
from core.p06_models.registry import _VARIANT_MAP

for _v in ("dfine", "dfine-s", "dfine-n", "dfine-m"):
    _VARIANT_MAP[_v] = "hf_detection"
```

### Steps to add a new HF model

1. Add the entry to `HF_MODEL_REGISTRY` in `core/p06_models/hf_model.py`:

```python
from transformers import NewModelForObjectDetection, NewModelConfig

HF_MODEL_REGISTRY["newmodel-s"] = (
    NewModelForObjectDetection,
    NewModelConfig,
    "org/newmodel-s-coco",
)
```

2. Add variant aliases in a new file `core/p06_models/newmodel.py`:

```python
from core.p06_models.registry import _VARIANT_MAP

for _v in ("newmodel", "newmodel-s", "newmodel-l"):
    _VARIANT_MAP[_v] = "hf_detection"
```

3. Import it in `core/p06_models/__init__.py` (see existing pattern at line 29–31):

```python
import core.p06_models.newmodel  # noqa: F401
```

4. Write a config — all `model:` keys except `arch`, `pretrained`, `input_size`, `num_classes`, `depth`, `width` pass directly to HF's `from_pretrained()`:

```yaml
model:
  arch: newmodel-s
  num_classes: 4
  input_size: [640, 640]
  # Any HF config param works here:
  num_queries: 300
  decoder_layers: 6
```

No loss config needed — `HFDetectionModel.forward_with_loss()` uses the HF model's built-in loss. The trainer detects `forward_with_loss()` automatically via `hasattr(model, 'forward_with_loss')` (see `core/p06_training/trainer.py:639`).

---

## Adding a Custom Framework (New Architecture)

Use this path when integrating a model that doesn't fit the HF adapter pattern.

### Step 1 — Implement DetectionModel ABC

Subclass `DetectionModel` from `core/p06_models/base.py`. Three abstract members are required:

```python
from core.p06_models.base import DetectionModel

class MyModel(DetectionModel):

    @property
    def output_format(self) -> str:
        # Controls trainer/evaluator dispatch. Use "yolox" for standard
        # anchor-free detection, "detr" for query-based detection.
        return "yolox"

    @property
    def strides(self) -> List[int]:
        return [8, 16, 32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, H, W)
        # Output layout described by output_format
        ...
```

For built-in loss computation (HF pattern), also implement:

```python
    def forward_with_loss(
        self, images: torch.Tensor, targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # Returns: (total_loss, loss_dict, predictions)
        # loss_dict keys: "cls_loss", "obj_loss", "reg_loss"
        ...
```

When `forward_with_loss()` is present, the trainer skips `build_loss()` entirely — no separate loss function is needed.

See `core/p06_models/yolox.py` for a full custom model example (CSPDarknet + PAFPN + Decoupled Head). See `core/p06_models/hf_model.py` for the `forward_with_loss()` pattern.

### Step 2 — Register with @register_model

```python
from core.p06_models.registry import register_model

@register_model("mymodel")
def build_mymodel(config: dict) -> MyModel:
    model_cfg = config.get("model", {})
    return MyModel(
        num_classes=model_cfg["num_classes"],
        # ... other params from config
    )
```

The decorator stores the builder in `MODEL_REGISTRY`. `build_model(config)` looks up `config["model"]["arch"]` and calls the builder.

### Step 3 — Add variant aliases

```python
# core/p06_models/mymodel_variants.py
from core.p06_models.registry import _VARIANT_MAP

_VARIANT_MAP["mymodel-s"] = "mymodel"
_VARIANT_MAP["mymodel-l"] = "mymodel"
```

`build_model()` resolves aliases via `_VARIANT_MAP` before looking up `MODEL_REGISTRY`.

### Step 4 — Import in \_\_init\_\_.py

```python
# core/p06_models/__init__.py — add after existing imports
import core.p06_models.mymodel         # noqa: F401
import core.p06_models.mymodel_variants  # noqa: F401
```

Imports trigger `@register_model` decoration at module load time.

### Step 5 — Write config

```yaml
model:
  arch: mymodel-s
  num_classes: 3
  input_size: [640, 640]
  # model-specific params read by your builder:
  hidden_dim: 256
```

### Step 6 — Register loss (standard path only)

If your model uses the standard `forward()` + separate loss path (not `forward_with_loss()`), register a loss:

Add a new loss class directly in `core/p06_training/losses.py` (or in a new `core/p06_training/my_loss.py` that you import there):

```python
from core.p06_training.losses import register_loss, DetectionLoss, LOSS_REGISTRY, _ARCH_LOSS_MAP

@register_loss("myloss", arch_aliases=["mymodel", "mymodel-s", "mymodel-l"])
class MyLoss(DetectionLoss):
    def forward(self, predictions, targets, grids=None):
        # Returns: (total_loss, loss_dict)
        ...
```

Add `loss.type: myloss` to the training config, or rely on `arch_aliases` for auto-detection.

---

## The output_format Contract

`output_format` is the single value that controls trainer dispatch, metrics, and postprocessing:

```
config["model"]["arch"]  →  build_model()  →  model.output_format  →  trainer dispatch
       "yolox-m"         →  build_yolox()  →  "yolox"              →  YOLOXLoss + compute_map()
       "dfine-s"         →  build_hf_model()→  "detr"              →  HF built-in loss + compute_map()
       "timm"            →  build_timm()   →  "classification"     →  CrossEntropy + accuracy
       "hf-segformer"    →  build_hf_seg() →  "segmentation"       →  HF built-in loss + mIoU
```

**Built-in formats:**

| Value | Task | Metrics | Loss path |
|---|---|---|---|
| `"yolox"` | Detection | mAP@0.5 via `compute_map()` | `YOLOXLoss` (separate) |
| `"detr"` | Detection | mAP@0.5 via `compute_map()` | HF built-in via `forward_with_loss()` |
| `"classification"` | Classification | accuracy, top-5 | HF/timm built-in |
| `"segmentation"` | Segmentation | mIoU | HF built-in |

**Adding a new format** requires changes in `core/p06_training/trainer.py`:

- `_validate()` — add a metrics branch for the new format
- `_decode_predictions()` — add a decode branch for the new format

Check `core/CLAUDE.md` for the dispatch points.

---

## Integrating External Libraries (MMDetection, Detectron2, SAM2)

Wrap the library model in a `DetectionModel` subclass. The key is adapting I/O format, not reimplementing the library:

```python
class MMDetAdapter(DetectionModel):

    def __init__(self, mmdet_model):
        super().__init__()
        self.mm = mmdet_model

    @property
    def output_format(self) -> str:
        return "yolox"   # or "detr" depending on output layout

    @property
    def strides(self) -> List[int]:
        return [8, 16, 32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call library's forward, convert output to standard tensor layout
        raw = self.mm(x)
        return self._convert_to_standard(raw)

    def forward_with_loss(self, images, targets):
        # Use library's built-in loss computation
        loss, raw = self.mm.forward_train(images, self._convert_targets(targets))
        predictions = self._convert_to_standard(raw)
        return loss, {"cls_loss": loss * 0, "reg_loss": loss}, predictions
```

The pattern mirrors `HFDetectionModel` in `core/p06_models/hf_model.py`.

---

## ONNX Export for New Frameworks

**Standard path (automatic for DetectionModel subclasses):** `core/p09_export/exporter.py` uses `torch.onnx.export()` with dynamic batch size. Nothing needed if your model is a standard `nn.Module`.

**HF Optimum path (automatic for HFDetectionModel subclasses):** The exporter detects `HFDetectionModel` and routes through Optimum's exporter, which handles HF-specific ops.

**Custom export:** Add an `export_onnx()` method to your model class if the default trace fails (e.g., control flow, custom ops). The exporter in `core/p09_export/exporter.py` checks for this method and calls it instead of the default path:

```python
class MyModel(DetectionModel):

    def export_onnx(self, path: str, input_size: tuple) -> None:
        dummy = torch.zeros(1, 3, *input_size)
        torch.onnx.export(
            self,
            dummy,
            path,
            opset_version=17,
            input_names=["images"],
            output_names=["predictions"],
            dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
        )
```

Run export:

```bash
uv run core/p09_export/export.py \
  --model runs/mymodel/best.pt \
  --training-config configs/myusecase/06_training.yaml \
  --export-config configs/_shared/09_export.yaml
```
