# Gated HF Repo Retry Download Log

Date: 2026-04-14
Operator: automated retry pass with authenticated HF token

## 1. Auth method

- HF token was loaded from `/home/nthanhtrung/.../edge_ai/.env` via `set -a; source .env; set +a`
  so `HF_TOKEN` was present in the environment for `hf download --token "$HF_TOKEN" ...`.
- The token value was never printed, logged, or committed. Any stderr that might have echoed it was
  passed through `sed "s|$HF_TOKEN|<REDACTED>|g"` before being written to logs.
- CLI: `hf` v1.10.2 (new `huggingface_hub` CLI; `huggingface-cli` also available).

## 2. Per-repo results

Downloads land in gitignored per-repo subfolders named `_hf_<owner>_<repo>/` under each
feature's `ai/pretrained/<feature>/` directory. Only the primary weight file is hashed.

### PPE -> `ai/pretrained/ppe-helmet_detection/`

| Repo | Result | Size on disk | Primary file | SHA256 |
|---|---|---|---|---|
| `keremberke/yolov5n-hard-hat-detection` | NOT_FOUND (404) | - | - | - |
| `keremberke/yolov5s-hard-hat-detection` | NOT_FOUND (404) | - | - | - |
| `keremberke/yolov5m-hard-hat-detection` | NOT_FOUND (404) | - | - | - |
| `Advantech-EIOT/qualcomm-ultralytics-ppe_detection` | still-gated ("Access denied. This repository requires approval.") | - | - | - |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | OK | 83 MB | `model.safetensors` | `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | OK | 327 MB | `model.safetensors` | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` |

Note: the three `keremberke/yolov5*-hard-hat-detection` repos now 404 (repository has been
deleted / renamed by the author — not a token/auth issue). They were gated in the prior bulk
pass but have since been removed from the Hub. Nothing to retry.

### Shoes -> `ai/pretrained/ppe-shoes_detection/`

| Repo | Result | Size on disk | Primary file | SHA256 |
|---|---|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | OK | 83 MB | `model.safetensors` | `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | OK | 327 MB | `model.safetensors` | `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` |

Hashes match the PPE copies (same upstream artifacts) — confirms clean mirror.

### Pose -> `ai/pretrained/safety-fall_pose_estimation/`

| Repo | Result | Size on disk | Primary file | SHA256 |
|---|---|---|---|---|
| `facebook/sapiens-pose-0.3b` | OK | 1.3 GB | `sapiens_0.3b_goliath_best_goliath_AP_573.pth` | `7b65f896508346f8b8fa7f82449082fcf801c811b8b122cf9fc446f6125a4ec0` |

### Poketenashi -> `ai/pretrained/safety-poketenashi/`

| Repo | Result | Size on disk | Primary file | SHA256 |
|---|---|---|---|---|
| `facebook/sapiens-pose-0.6b-torchscript` | OK | 2.5 GB | `sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2` | `7d5407351b9f20dddea72cfa7be26783835f58d4d565e3aaf4746cccce9b677a` |
| `facebook/sapiens-pose-1b-torchscript` | too-large (4.68 GB > 3 GB cap) — skipped per instructions | - | - | - |

Size for 1b was checked via `GET /api/models/facebook/sapiens-pose-1b-torchscript/tree/main?recursive=true`
(authenticated): total 4,677,007,508 bytes.

## 3. Summary

- Attempted: 10 repo requests (8 unique repos, dinov3 downloaded into two feature dirs).
- OK: 6 (dinov3-vits16 x2, dinov3-vitb16 x2, sapiens-pose-0.3b, sapiens-pose-0.6b-torchscript).
- still-gated: 1 (`Advantech-EIOT/qualcomm-ultralytics-ppe_detection`).
- NOT_FOUND (repo deleted): 3 (keremberke/yolov5{n,s,m}-hard-hat-detection).
- too-large, skipped: 1 (`facebook/sapiens-pose-1b-torchscript`, 4.68 GB).

Counts against the original "previously gated" list: **6 succeeded / 1 still blocked**
(the 3 keremberke repos are no longer recoverable from HF at all; the 1b torchscript
was a size-policy skip, not a gate).

## 4. Follow-up — user action needed

- `Advantech-EIOT/qualcomm-ultralytics-ppe_detection` — HF returned
  `Error: Access denied. This repository requires approval.` even with the token.
  User must visit https://huggingface.co/Advantech-EIOT/qualcomm-ultralytics-ppe_detection
  with the `okeysir` account (the one tied to the token) and click "Agree and access
  repository" on the model card. Afterwards a re-run of this script will pull it.
- `keremberke/yolov5{n,s,m}-hard-hat-detection` — these now 404. No user action will fix it;
  drop them from any bulk manifest. Equivalent weights can be sourced from
  `keremberke/yolov5*-construction-safety` or alternative hard-hat repos already in
  `ai/pretrained/ppe-helmet_detection/`.
- `facebook/sapiens-pose-1b-torchscript` — only the size policy is blocking. If the user
  wants it, re-run with the >3 GB cap lifted.

## Notes

- All downloads live under gitignored `_hf_<owner>_<repo>/` directories — no weights are
  committed.
- Token never appeared in stdout/stderr in this pass; the sed redaction filter was a
  belt-and-suspenders measure.

## Round 2 — Advantech-EIOT retry after Agree

Date: 2026-04-14
User clicked "Agree and access repository" on the model card, then requested retry.

Command: `hf download Advantech-EIOT/qualcomm-ultralytics-ppe_detection --token "$HF_TOKEN" --local-dir .../ai/pretrained/ppe-helmet_detection/_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection/`

Result: **OK** — 3 files fetched.

| File | Size | SHA256 |
|---|---|---|
| `ppe_yolov11n_w8a16_160x160_pics1000.dlc` | 6.2 MB (6,414,831 B) | `14e8e4a7c2b6a69a8b352572664e3616546bed4e0a1f03d646b24bdf39fe87f3` |
| `README.md` | 2,024 B | `899b44c2cbf21219d1f7410317e883e4d195052e6b472b573106622ea3a0ff89` |
| `.gitattributes` | 1,673 B | `06405fd0e7b15f68245e5698ef6a6472c04229d1558582a5d9eb1b4ced398dfb` |

Downloads land under gitignored `_hf_Advantech-EIOT_qualcomm-ultralytics-ppe_detection/`. Primary artifact is a pre-quantized (`w8a16`) SNPE DLC for Qualcomm NPU, YOLOv11n at 160x160 input, calibrated on 1000 pictures. Access gate is now cleared for the `okeysir` account.
