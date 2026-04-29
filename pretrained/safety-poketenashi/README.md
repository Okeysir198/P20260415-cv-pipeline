# safety-poketenashi — Pretrained Weights

## Summary

Whole-body (133-keypoint COCO-WholeBody) pose candidates for the
poketenashi hospitality/behaviour pipeline. Total **16 files** across
**~3.3 GB**. See brief:
[../../docs/technical_study/safety-poketenashi-sota.md](../../docs/technical_study/safety-poketenashi-sota.md).

## Files on disk (as of 2026-04-14)

| File / Subfolder | Size | Type | License | Source | Notes |
|---|---|---|---|---|---|
| `dw-ll_ucoco_384.onnx` | 129 MB | DWPose-L WholeBody 384×288 ONNX | Apache-2.0 | DWPose | **SOTA pick A — 133-kpt head.** sha256 `724f4ff…`. |
| `rtmpose-s_coco-wholebody.pth` | 69 MB | RTMPose-S WholeBody | Apache-2.0 | MMPose | Lighter WB alternative (stand-in for RTMW-m). |
| `rtmw-l_256x192.zip` | 203 MB | RTMW-L WholeBody 256×192 (mmpose SDK) | Apache-2.0 | MMPose | SDK-packaged ONNX. |
| `rtmw-l_384x288.zip` | 204 MB | RTMW-L WholeBody 384×288 (mmpose SDK) | Apache-2.0 | MMPose | |
| `rtmw-l_cocktail14_384x288.pth` | 220 MB | RTMW-L Cocktail14 checkpoint | Apache-2.0 | MMPose | Highest-AP RTMW variant. |
| `_hf_facebook_sapiens-pose-0.6b-torchscript/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2` | 2.5 GB | Sapiens 0.6B pose (TorchScript) | **Sapiens License (non-commercial)** | facebook | Benchmark only — license blocker. |
| `.cache/huggingface/` | — | HF download metadata | — | — | Tooling cache. |
| `DOWNLOAD_MANIFEST.md` | 665 B | Manifest w/ SHA256s | — | — | |

## Recommended defaults (from SOTA brief)

- **Recommendation A — YOLOX-Tiny + DWPose-L + CPU rule engine** → DWPose head = `dw-ll_ucoco_384.onnx` (on disk); YOLOX-Tiny lives in shared `ai/pretrained/yolox_tiny.pth`.
- **Recommendation B — YOLOX-Tiny + RTMW-m (lighter WB)** → closest on-disk proxy is `rtmpose-s_coco-wholebody.pth`; `rtmw-l_*` zips are the L-tier fallback.
- **Recommendation C — RTMO-m + DWPose-L crop-pose** → RTMO-m not downloaded here; see `safety-fall_pose_estimation/rtmo-s_body7_640x640.pth` for RTMO-S, or upgrade path to RTMO-m.

## Gated / skipped / 404

- **Sapiens 0.6B (TorchScript)** — fetched via gated-retry; kept for benchmark only (Sapiens License non-commercial).
- **Sapiens 1B** — skipped (size > 3 GB policy cap).

## Related docs

- SOTA brief: `../../docs/technical_study/safety-poketenashi-sota.md`
- Pose bulk log: `../../docs/technical_study/access-pose-bulk-download-log.md`
- Gated retry log: `../../docs/technical_study/gated-retry-download-log.md`
- Quality report: see each feature's `predict/QUALITY_REPORT.md` —
  the umbrella `features/safety-poketenashi/` was split (2026-04-29)
  into 5 sibling feature folders (`safety-poketenashi_phone_usage`,
  `_hands_in_pockets`, `_no_handrail`, `_stair_diagonal`,
  `_point_and_call`); all share this DWPose storage directory.
- SHA256s: `./DOWNLOAD_MANIFEST.md`
