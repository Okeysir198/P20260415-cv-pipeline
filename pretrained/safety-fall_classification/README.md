# safety-fall_classification — Pretrained Weights

## Summary

Single-frame and temporal (clip-based) action-recognition candidates for fall
classification, plus 22 symlinks into the sibling `visual_core` action /
segmentation checkpoints. Total **46 on-disk entries** (24 file artefacts
+ 22 symlinks) across **~3.5 GB**. See brief:
[../../docs/technical_study/safety-fall_classification-sota.md](../../docs/technical_study/safety-fall_classification-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `efficientnetv2_rw_s.ra2_in1k.bin` | 93 MB | EfficientNetV2-S (timm) | Apache-2.0 | timm | **Top single-frame pick (production)**. |
| `mobilenetv4_conv_small.e2400_r224_in1k.bin` | 15 MB | MobileNetV4-S (timm) | Apache-2.0 | timm | Lightweight single-frame. |
| `dinov2-small.bin` | 85 MB | DINOv2 ViT-S/14 backbone | Apache-2.0 | facebook | Clean-license ViT baseline. |
| `fall_resnet18_popkek00.safetensors` + `..._config.json` | 43 MB | ResNet18 fall fine-tune | Apache-2.0 (check) | popkek00 | Community fall classifier. |
| `videomae-small-finetuned-kinetics.bin` | 85 MB | VideoMAE-S Kinetics-400 | **CC-BY-NC-4.0** | MCG-NJU | Benchmark only — research-only. |
| `videomae-base-finetuned-kinetics.bin` | 331 MB | VideoMAE-B Kinetics-400 | **CC-BY-NC-4.0** | MCG-NJU | Benchmark only. |
| `videomaev2_base_k710.safetensors` | 329 MB | VideoMAE-V2 Base K710 | **CC-BY-NC-4.0** | OpenGVLab | Benchmark only. |
| `videomaev2_large_k710.safetensors` | 1.2 GB | VideoMAE-V2 Large K710 | **CC-BY-NC-4.0** | OpenGVLab | Benchmark only. |
| `uniformerv2_b16_k400_k710.pyth` | 438 MB | UniFormerV2-B K400/K710 | **CC-BY-NC-4.0** | Sense-X | Benchmark only. |
| `slowfast_r50_k400.pyth` | 265 MB | SlowFast R50 K400 | Apache-2.0 | PyTorchVideo | Temporal baseline. |
| `slowfast_r101_k400.pyth` | 481 MB | SlowFast R101 K400 | Apache-2.0 | PyTorchVideo | Larger temporal. |
| `x3d_xs.pyth` / `x3d_s.pyth` / `x3d_m.pyth` / `x3d_l.pyth` | 30 / 30 / 30 / 48 MB | X3D K400 | Apache-2.0 | PyTorchVideo | Efficient 3D conv family. |
| `movinet_a1_base.tar.gz` | 19 MB | MoViNet-A1 base (TF) | Apache-2.0 | TF Hub | Streaming-friendly. |
| `movinet_a2_base.tar.gz` | 21 MB | MoViNet-A2 base (TF) | Apache-2.0 | TF Hub | |
| `movinet_a2_stream.tar.gz` | 29 MB | MoViNet-A2 stream (TF) | Apache-2.0 | TF Hub | Streaming variant. |
| `movinet_a3_base.tar.gz` | 29 MB | MoViNet-A3 base (TF) | Apache-2.0 | TF Hub | |
| `yolov8_fall_kamalchibrani.pt` | 6 MB | YOLOv8n fall detect | **AGPL-3.0** | kamalchibrani | Benchmark only. |
| `yolov11_fall_melihuzunoglu.pt` | 5.3 MB | YOLOv11 fall detect | **AGPL-3.0** | melihuzunoglu | Benchmark only. |
| `visualcore_asformer_gtea_split1.pt` | symlink | ASFormer GTEA | (sibling project) | → `visual_core/...` | Action-segmentation stage-1 step. |
| `visualcore_c2f_tcn_gtea_split1.pt` | symlink | C2F-TCN GTEA | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_diffact_gtea_split{1..4}.pt` | 4 symlinks | DiffAct GTEA splits | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_fact_breakfast_split{1..4}.pt` | 4 symlinks | FACT Breakfast splits | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_fact_gtea_split{1..4}.pt` | 4 symlinks | FACT GTEA splits | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_fact_egoprocel_split1.pt` | symlink | FACT EgoProceL | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_fact_epic_kitchens_split1.pt` | symlink | FACT Epic-Kitchens | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_ltcontext_breakfast_split{1..4}.pth` | 4 symlinks | LTContext Breakfast | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_ltcontext_gtea_split1.pt` | symlink | LTContext GTEA | (sibling project) | → `visual_core/...` | Stage-2 action. |
| `visualcore_mstcn_gtea_split1.pt` | symlink | MS-TCN GTEA | (sibling project) | → `visual_core/...` | Stage-1 step. |
| `_imagenet_classes.json` | 16 KB | ImageNet class map | — | — | Probe helper. |
| `DOWNLOAD_MANIFEST.md` | 2.6 KB | Manifest w/ SHA256s | — | — | |

All `visualcore_*` entries are symlinks to
`/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/visual_core/01_code/checkpoints/...`.

## Recommended defaults (from SOTA brief)

- **Single-frame (Apache-2.0, deploy-ready)** → `efficientnetv2_rw_s.ra2_in1k.bin` (on disk).
- **Lightweight single-frame** → `mobilenetv4_conv_small.e2400_r224_in1k.bin` (on disk).
- **Temporal (clean-license)** → SlowFast R50 (`slowfast_r50_k400.pyth`) / X3D-M (`x3d_m.pyth`) on disk.
- The brief explicitly **excludes for production**: VideoMAE/V2, UniFormerV2, DINOv3, ConvNeXt-V2 — all CC-BY-NC.

## Gated / skipped / 404

No gated items outstanding. All CC-BY-NC temporal models are kept strictly for benchmarking.

## Related docs

- SOTA brief: `../../docs/technical_study/safety-fall_classification-sota.md`
- Deep dive: `../../docs/technical_study/safety-fall_oss-pretrained-deep-dive.md`
- Bulk log: `../../docs/technical_study/safety-fall_bulk-download-log.md`
- NVIDIA TAO investigation: `../../docs/technical_study/nvidia-tao-actionrecognitionnet-investigation.md`
- Sibling inventory: `../../docs/technical_study/sibling-projects-inventory.md`
- Quality report: `../../features/safety-fall_classification/predict/QUALITY_REPORT.md`
- SHA256s: `./DOWNLOAD_MANIFEST.md`
