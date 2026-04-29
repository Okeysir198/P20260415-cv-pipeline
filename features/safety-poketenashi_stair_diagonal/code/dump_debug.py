"""Per-frame debug-CSV dump for stair_diagonal failure-mode analysis.

For each requested video, runs the predictor and writes a CSV with every
frame's rule debug fields (trajectory_angle_deg, threshold_deg, buffer_size,
trigger flag, reason) plus extra derived fields the rule does not expose
(hip_x, hip_y, per-keypoint scores for the hip-relevant joints).

The CSV is the input for offline analysis comparing TP frames vs FP/FN frames
to confirm the discriminating feature for each Phase 2 intervention.

The rule is **stateful** — the predictor's per-track ``StairSafetyDetector``
buffer accumulates across frames. The dumper instantiates ONE predictor per
video so the buffer evolves naturally.

Usage:
    uv run python features/safety-poketenashi_stair_diagonal/code/dump_debug.py \\
        VIDEO_NAME [VIDEO_NAME ...]
    # or, if no args, dumps the two default samples in this feature.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_FEAT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FEAT / "code"))
sys.path.insert(0, str(_FEAT.parent.parent))

from predictor import StairDiagonalPredictor  # noqa: E402

_CONFIG = _FEAT / "configs" / "10_inference.yaml"
_SAMPLES = _FEAT / "samples"
_EVAL_DIR = _FEAT / "eval"

_PRIORITY = [
    "04_NA_diagonal_crossing.mp4",
    "NA_diagonal_crossing_spkepcmwi.mp4",
]

_FIELDS = [
    "frame", "t",
    "label", "alert", "reason",
    "trajectory_angle_deg", "threshold_deg", "buffer_size",
    "hip_x", "hip_y",
    "score_l_hip", "score_r_hip",
    "score_l_shoulder", "score_r_shoulder",
    "score_l_knee", "score_r_knee",
]

_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_SH, _R_SH = 5, 6


def dump_one(name: str) -> Path | None:
    src = _SAMPLES / name
    if not src.exists():
        print(f"  ! {name}  (missing)")
        return None
    p = StairDiagonalPredictor(_CONFIG)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = _EVAL_DIR / f"debug_{Path(name).stem}.csv"

    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = p.process_frame(frame)
            row: dict = {
                "frame": fi,
                "t": round(fi / fps, 3),
                "alert": "stair_diagonal" if out.triggered_track_ids else "",
            }
            if out.tracks:
                tr = out.tracks[0]
                d = tr.behavior.debug_info
                k = tr.keypoints
                s = tr.kp_scores
                # mid-hip weighted by visibility — mirrors the rule's logic
                if s[_L_HIP] >= 0.3 and s[_R_HIP] >= 0.3:
                    hip = 0.5 * (k[_L_HIP] + k[_R_HIP])
                elif s[_L_HIP] >= 0.3:
                    hip = k[_L_HIP]
                elif s[_R_HIP] >= 0.3:
                    hip = k[_R_HIP]
                else:
                    hip = np.array([np.nan, np.nan], dtype=np.float32)
                row.update({
                    "label": "person",
                    "reason": d.get("reason", ""),
                    "trajectory_angle_deg": d.get("trajectory_angle_deg"),
                    "threshold_deg": d.get("threshold_deg"),
                    "buffer_size": d.get("buffer_len"),
                    "hip_x": round(float(hip[0]), 1) if not np.isnan(hip[0]) else "",
                    "hip_y": round(float(hip[1]), 1) if not np.isnan(hip[1]) else "",
                    "score_l_hip": round(float(s[_L_HIP]), 3),
                    "score_r_hip": round(float(s[_R_HIP]), 3),
                    "score_l_shoulder": round(float(s[_L_SH]), 3),
                    "score_r_shoulder": round(float(s[_R_SH]), 3),
                    "score_l_knee": round(float(s[_L_KNEE]), 3),
                    "score_r_knee": round(float(s[_R_KNEE]), 3),
                })
            else:
                row["label"] = "no_person"
            w.writerow(row)
            fi += 1
    cap.release()
    print(f"  + {name}  -> {out_path.relative_to(_FEAT.parent.parent)}  ({fi}/{n} frames)")
    return out_path


def main() -> None:
    targets = sys.argv[1:] or _PRIORITY
    print(f"Dumping {len(targets)} video(s) to {_EVAL_DIR.relative_to(_FEAT.parent.parent)}")
    for name in targets:
        dump_one(name)


if __name__ == "__main__":
    main()
