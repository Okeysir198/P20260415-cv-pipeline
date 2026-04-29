"""Per-frame debug-CSV dump for hands-in-pockets failure-mode analysis.

For each requested video, runs the predictor and writes a CSV with every
frame's pose-rule debug fields plus extra derived fields the rule does not
expose (wrist-vs-hip y offset per side, wrist-vs-torso x offset per side,
torso geometry, all 17 keypoint scores).

The CSV is the input for offline analysis comparing TP frames vs FP/FN frames
to confirm the discriminating feature for each Phase 2 intervention.

Usage:
    uv run python features/safety-poketenashi_hands_in_pockets/code/dump_debug.py \\
        VIDEO_NAME [VIDEO_NAME ...]
    # or, if no args, dumps the priority videos for Phase 1 (the 2 in samples/).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_FEAT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FEAT / "code"))
sys.path.insert(0, str(_FEAT.parent.parent))

from predictor import HandsInPocketsPredictor  # noqa: E402

_CONFIG = _FEAT / "configs" / "10_inference.yaml"
_SAMPLES = _FEAT / "samples"
_EVAL_DIR = _FEAT / "eval"

_PRIORITY = [
    "01_PO_hands_in_pockets.mp4",
    "PO_hands_in_pockets_spkepcmwi.mp4",
]

# COCO-17 indices (mirroring hands_in_pockets_detector.py).
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_FIELDS = [
    "frame", "t",
    "label", "alert", "triggered",
    "wrist_below_hip_l", "wrist_below_hip_r",            # boolean (rule-style with margin)
    "wrist_torso_offset_l", "wrist_torso_offset_r",      # px |wrist_x - torso_cx|
    "wrist_y_minus_hip_y_l", "wrist_y_minus_hip_y_r",    # px (positive = wrist below hip)
    "torso_height_px", "torso_width_px",
    "left_hit", "right_hit",
    "score_l_shoulder", "score_r_shoulder",
    "score_l_wrist", "score_r_wrist",
    "score_l_hip", "score_r_hip",
    "score_nose", "score_l_elbow", "score_r_elbow",
    "score_l_knee", "score_r_knee",
]


def dump_one(name: str) -> Path | None:
    src = _SAMPLES / name
    if not src.exists():
        print(f"  ! {name}  (missing)")
        return None
    p = HandsInPocketsPredictor(_CONFIG)
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
                "alert": "|".join(out["alerts"]) if out["alerts"] else "",
                "triggered": "hands_in_pockets" in out["alerts"],
            }
            if out["persons"]:
                # Use the first (largest detected) person's keypoints + rule debug.
                pb = out["persons"][0]
                k = pb.keypoints
                s = pb.kp_scores
                d = pb.result.debug_info or {}
                row["label"] = pb.result.behavior if pb.result.triggered else "no_violation"
                row["left_hit"] = bool(d.get("left_hit", False))
                row["right_hit"] = bool(d.get("right_hit", False))
                # Geometry derived directly (rule does not export raw offsets).
                if (
                    s[_L_HIP] > 0 and s[_R_HIP] > 0
                    and s[_L_SHOULDER] > 0 and s[_R_SHOULDER] > 0
                ):
                    mid_hip = 0.5 * (k[_L_HIP] + k[_R_HIP])
                    mid_sh = 0.5 * (k[_L_SHOULDER] + k[_R_SHOULDER])
                    torso_h = float(np.linalg.norm(mid_sh - mid_hip))
                    torso_w = float(abs(k[_L_HIP, 0] - k[_R_HIP, 0]))
                    torso_cx = float(mid_hip[0])
                else:
                    torso_h = torso_w = torso_cx = math.nan
                torso_known = not math.isnan(torso_cx)
                row.update({
                    "wrist_below_hip_l": bool(d.get("left_wrist_below_hip", False)),
                    "wrist_below_hip_r": bool(d.get("right_wrist_below_hip", False)),
                    "wrist_torso_offset_l": round(abs(float(k[_L_WRIST, 0]) - torso_cx), 1)
                    if torso_known else "",
                    "wrist_torso_offset_r": round(abs(float(k[_R_WRIST, 0]) - torso_cx), 1)
                    if torso_known else "",
                    "wrist_y_minus_hip_y_l": round(
                        float(k[_L_WRIST, 1]) - float(k[_L_HIP, 1]), 1
                    ),
                    "wrist_y_minus_hip_y_r": round(
                        float(k[_R_WRIST, 1]) - float(k[_R_HIP, 1]), 1
                    ),
                    "torso_height_px": round(torso_h, 1) if torso_known else "",
                    "torso_width_px": round(torso_w, 1) if torso_known else "",
                    "score_l_shoulder": round(float(s[_L_SHOULDER]), 3),
                    "score_r_shoulder": round(float(s[_R_SHOULDER]), 3),
                    "score_l_wrist": round(float(s[_L_WRIST]), 3),
                    "score_r_wrist": round(float(s[_R_WRIST]), 3),
                    "score_l_hip": round(float(s[_L_HIP]), 3),
                    "score_r_hip": round(float(s[_R_HIP]), 3),
                    "score_nose": round(float(s[0]), 3),
                    "score_l_elbow": round(float(s[7]), 3),
                    "score_r_elbow": round(float(s[8]), 3),
                    "score_l_knee": round(float(s[13]), 3),
                    "score_r_knee": round(float(s[14]), 3),
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
