# NVIDIA TAO ActionRecognitionNet (`trainable_rgb_3d`) — Investigation

Date: 2026-04-14
Scope: Evaluate as candidate for edge_ai Phase 1 action-classification use
cases — specifically `fall_classification` (g) and `poketenashi` (h).

## 1. Model overview

`ActionRecognitionNet` is NVIDIA's pre-trained action recognition model shipped
via the NGC catalog under the TAO (Train-Adapt-Optimize) Toolkit. The
`trainable_rgb_3d` version is a 3D-CNN RGB-only variant meant as a starting
point for transfer learning before deployment on NVIDIA DeepStream / TensorRT.

- Model card: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet?version=trainable_rgb_3d
- TAO docs: https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/action_recognition_net.html
- Intended use (NVIDIA): "Recognizes actions of people in video"; must be
  fine-tuned before production deployment.

## 2. Architecture & training data

| Attribute            | Value                                                    |
|----------------------|----------------------------------------------------------|
| Architecture         | 3D CNN, ResNet18 backbone                                |
| Input tensor         | `3 x 32 x 224 x 224` (C x D x H x W)                     |
| Precision shipped    | FP32, FP16                                               |
| Model file size      | ~332 MB (`.tlt`, encrypted; load key `nvidia_tao`)       |
| Training data        | 1024 videos from HMDB5 (a 5-class subset of HMDB51)      |
| Classes (5)          | `walk` (494), `ride_bike` (93), `run` (209), `fall_floor` (123), `push` (105) |
| Reported accuracy    | center 84.69% / conv 85.59% on HMDB5 eval                |

Notes:
- HMDB5 is a tiny subset (≈1k clips). NVIDIA itself states "performance on
  other video sources may be lower" — the pretraining is not Kinetics-400
  scale, so it is really a *starter* checkpoint, not a general feature
  extractor.
- A 2D variant (`96 x 224 x 224`) also exists under the same model entry.

Source: NGC model card (fetched 2026-04-14).

## 3. License & commercial terms

- License shown on NGC: **NVIDIA AI Enterprise EULA**
  https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/
- Legacy TAO Toolkit SLA:
  https://developer.download.nvidia.com/licenses/tao_toolkit_21-08.pdf
  (redirected from `developer.nvidia.com/tao-toolkit-software-license-agreement`).

Key implications (from NVIDIA AI Enterprise EULA + TAO SLA review):

- Commercial use is **permitted** only under the terms of the NVIDIA AI
  Enterprise EULA; it is **not** an OSI-compatible license (not Apache-2.0,
  not MIT, not BSD).
- Redistribution of the pre-trained weights outside NVIDIA channels is
  restricted; derivative models trained via TAO inherit EULA obligations.
- The SLA historically restricts deployment optimizations to NVIDIA platforms
  (TensorRT / DeepStream / Jetson / data-center GPUs). Running fine-tuned
  `.etlt` artifacts on non-NVIDIA accelerators is outside the intended
  deployment path and not covered by support.
- The model file itself is **encrypted** with load key `nvidia_tao`; you must
  decrypt via the TAO toolkit to touch weights.

Net: this is **not** an Apache-2.0 / MIT model. It conflicts with our
preferred license posture.

## 4. Fine-tune workflow

Per https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/action_recognition_net.html :

- Entry point: `tao model action_recognition train -e <spec.yaml>`
- Dataset layout (per-class, per-clip, pre-extracted frames):
  ```
  /Dataset/<class>/<video_id>/rgb/        # RGB frames (jpg/png)
                             /u /v        # optical-flow x/y (optional)
  ```
- The TAO launcher wraps a **Docker container**; the container bundles
  PyTorch + CUDA + NVIDIA-specific layers. An NVIDIA GPU with a recent CUDA
  driver is required (documented `num_gpus` ≥ 1).
- There is **no** "plain Ubuntu + pip install pytorch" path for this model:
  the checkpoint is encrypted and the training scripts live inside the
  closed-source TAO container. Weights cannot be loaded in vanilla
  `torchvision` / `timm`.
- Multi-GPU is supported (DDP inside the container).

So: fine-tuning **requires NVIDIA GPU + TAO Docker**. It cannot be done on a
chip-agnostic laptop or a non-NVIDIA training host.

## 5. Export options

- Export command:
  `tao model action_recognition export -e <spec> export.checkpoint=<tlt> export.onnx_file=<out.onnx>`
- As of TAO 5.x, `.etlt` is being phased out and ONNX is the primary export
  format (confirmed by TAO 5 forum threads, e.g.
  https://forums.developer.nvidia.com/t/onnx-output-in-tao-5-0-how-to-get-an-etlt-model-in-tao-5-0-0/268597 ).
- The exported ONNX is **standard ONNX** and can, in principle, be consumed
  by ONNX Runtime / other toolchains. However:
  - The graph often contains NVIDIA-specific plugins or custom ops targeted
    at TensorRT; portability to Hailo / RKNN / OpenVINO is not guaranteed
    without op rewriting.
  - INT8 PTQ is designed around **TensorRT** calibration. TAO provides
    `tao deploy` tooling that is TensorRT-centric; generic INT8 (ONNX
    Runtime, NNCF, etc.) is not a first-class flow.
- Fine-tune → ONNX is possible; fine-tune → ONNX → non-NVIDIA edge NPU is
  *feasible but unsupported* and requires a porting effort per target.

## 6. Fit for Phase 1 fall & poketenashi

Honest assessment against our constraints:

| Constraint                         | TAO ActionRecognitionNet      |
|------------------------------------|-------------------------------|
| Chip-agnostic deployment           | Fails — TensorRT/DeepStream first; other NPUs unsupported |
| Apache-2.0 / MIT preference        | Fails — NVIDIA AI Enterprise EULA |
| Fine-tune on plain PyTorch/Ubuntu  | Fails — TAO Docker + NVIDIA GPU mandatory |
| ~18 TOPS INT8 budget               | Plausible for ResNet18-3D at 32×224², but INT8 path is TRT-centric |
| Label coverage for `fall_floor`    | Matches (fall_floor is a native class) |
| Label coverage for `poketenashi`   | None — not in training data, requires custom dataset + fine-tune |
| Input format (clip of 32 frames)   | Adds buffering latency; OK for fall, acceptable for poketenashi |

For `fall_classification`: the `fall_floor` class is in the pretrain set,
which gives a warm-start advantage over Kinetics-only alternatives. BUT the
pretrain corpus (HMDB5, 123 fall clips) is small and the license/ops lock-in
outweighs the warm-start.

For `poketenashi` (hands-in-pockets / improper service posture): the
pretrain label set offers zero overlap. Any solution needs custom data;
there is no meaningful advantage to starting from ActionRecognitionNet vs.
a Kinetics-pretrained Apache-2.0 backbone.

## 7. Open-source alternatives

| Model                 | License        | Params (approx) | Pretrain                     | Edge-INT8 friendliness | Fall/pose relevance | Notes |
|-----------------------|----------------|-----------------|------------------------------|-------------------------|---------------------|-------|
| **X3D-S / X3D-M**     | Apache-2.0 (PySlowFast) | 3.8M / 3.8M | Kinetics-400            | Good (pure conv3D)      | Strong              | Best accuracy/FLOPs trade; torchvision + PySlowFast. |
| **MoViNet-A0..A2**    | Apache-2.0 (TF Model Garden) | 3.1M / 4.6M / 4.8M | Kinetics-600      | Excellent (designed for mobile, streamable) | Strong          | Streaming inference — no 32-frame buffer; ideal for edge. |
| **SlowFast R50**      | Apache-2.0     | ~34M            | Kinetics-400                 | Medium (two-path, heavy)| Strong (benchmark) | Overkill at 18 TOPS INT8. |
| **I3D**               | Apache-2.0 (various impls) | ~12M | Kinetics-400 / ImageNet-inflated | OK                 | Medium              | Older; superseded by X3D. |
| **R(2+1)D**           | BSD (torchvision) | ~31M         | Kinetics-400                 | OK                      | Medium              | Clean torchvision weights; simple export. |
| **TPN**               | Apache-2.0 (MMAction2) | ~25M    | Kinetics-400                 | OK                      | Medium              | Good temporal modelling but heavier. |
| **UniFormerV2-B**     | Apache-2.0     | ~115M           | Kinetics-710                 | Poor (transformer, too heavy) | Strong        | Too large for 18 TOPS. |
| **VideoMAE-B**        | CC-BY-NC-4.0 (non-commercial) | ~87M | Kinetics-400 SSL         | Poor                    | Strong              | **Flag: non-commercial license** — reject for product. |

Recommended shortlist for Phase 1:

1. **MoViNet-A1 / A2 (Apache-2.0, Kinetics-600)** — streamable, edge-oriented,
   small, license-clean. Primary candidate for both fall and poketenashi.
2. **X3D-M (Apache-2.0, Kinetics-400)** — stronger offline accuracy if
   streaming is not required; clean PyTorch weights via PySlowFast.

Both can be fine-tuned in plain PyTorch on any GPU, exported to standard ONNX,
and quantized with generic tooling (ONNX Runtime, NNCF, or each vendor's
own PTQ — Hailo SDK, RKNN Toolkit, OpenVINO POT).

## 8. Verdict & rationale

**REJECT** for Phase 1 adoption. Conditional re-look only if the program
ever pivots to an NVIDIA-only edge stack (Jetson + DeepStream + TensorRT).

Rationale:

1. **License conflict.** NVIDIA AI Enterprise EULA is incompatible with our
   Apache-2.0 / MIT preference and introduces redistribution constraints
   on derivative fine-tuned weights.
2. **Vendor lock-in.** The entire toolchain (training container, `.tlt`
   encrypted weights, `.etlt` export, `tao deploy` INT8) is NVIDIA-centric.
   This directly contradicts our chip-agnostic mandate — the model
   recommendations must not assume Hailo / RKNN / TensorRT.
3. **Training friction.** Requires NVIDIA GPU + Docker + TAO launcher; no
   "plain PyTorch" route. Fine-tuning on a mixed / non-NVIDIA training
   fleet is impossible.
4. **Weak pretrain corpus.** HMDB5 (1,024 clips) is far smaller than
   Kinetics-400/600 used by Apache-2.0 alternatives. The `fall_floor`
   warm-start advantage does not offset points 1–3.
5. **Exportability caveat.** ONNX export exists but carries TRT-targeted
   plugins and assumes TensorRT INT8; porting to non-NVIDIA NPUs is
   unsupported effort, not a shipped feature.

Action items for the roadmap:

- Replace ActionRecognitionNet with **MoViNet-A1/A2** (primary) or
  **X3D-M** (secondary) in the Phase 1 action-classification entries.
- Record the rejection in the Phase 1 model selection log so it does not
  get re-proposed.

## 9. References

- NGC model card (overview, version list):
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet
- NGC model card (`trainable_rgb_3d`):
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet?version=trainable_rgb_3d
- TAO ActionRecognitionNet fine-tune docs:
  https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/action_recognition_net.html
- NVIDIA AI Enterprise EULA:
  https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/
- TAO Toolkit SLA (legacy PDF):
  https://developer.download.nvidia.com/licenses/tao_toolkit_21-08.pdf
- TAO 5.0 ONNX export discussion:
  https://forums.developer.nvidia.com/t/onnx-output-in-tao-5-0-how-to-get-an-etlt-model-in-tao-5-0-0/268597
- TAO tutorials notebook:
  https://github.com/NVIDIA/tao_tutorials/blob/main/notebooks/tao_launcher_starter_kit/action_recognition_net/actionrecognitionnet.ipynb
- PySlowFast (X3D / SlowFast, Apache-2.0):
  https://github.com/facebookresearch/SlowFast
- MoViNet (TF Model Garden, Apache-2.0):
  https://github.com/tensorflow/models/tree/master/official/projects/movinet
- VideoMAE (license note — CC-BY-NC):
  https://github.com/MCG-NJU/VideoMAE
- MMAction2 model zoo (TPN / I3D / SlowFast / UniFormerV2):
  https://mmaction2.readthedocs.io/en/latest/model_zoo/recognition.html
