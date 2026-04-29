"""Per-frame debug-CSV dump for failure-mode analysis.

For each requested video, runs the orchestrator and writes a CSV with every
frame's pose-rule debug fields (label, side, elbow angle, forearm elevation,
azimuth, suppression reason), plus extra derived fields the rule does not
expose (wrist-to-ear ratio per side, hip position, all 17 keypoint scores).

The CSV is the input for offline analysis comparing TP frames vs FP/FN frames
to confirm the discriminating feature for each Phase 2 intervention.

Usage:
    uv run python features/safety-poketenashi_point_and_call/code/dump_debug.py \\
        VIDEO_NAME [VIDEO_NAME ...]
    # or, if no args, dumps the four priority videos for Phase 1.
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

from orchestrator import PointAndCallOrchestrator  # noqa: E402

_CONFIG = _FEAT / "configs" / "10_inference.yaml"
_SAMPLES = _FEAT / "samples"
_EVAL_DIR = _FEAT / "eval"

_PRIORITY = [
    "POKETENASHI_anzen_daiichi_lecture.mp4",   # 51 FPs cluster
    "POKETENASHI.mp4",                          # phone-on-ear FP cluster (~t=82s)
    "SHI_point_and_call_spkepcmwi.mp4",         # far-field FN cluster
    "05_SHI_point_and_call.mp4",                # TP reference for comparison
]

_FIELDS = [
    "frame", "t",
    "label", "side", "elbow_angle", "arm_elevation", "azimuth", "suppressed_by",
    "matcher_progress", "alerts",
    "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y",
    "shoulder_w_px", "wrist_ear_ratio_min",
    "score_l_shoulder", "score_l_elbow", "score_l_wrist",
    "score_r_shoulder", "score_r_elbow", "score_r_wrist",
    "score_nose", "score_l_ear", "score_r_ear",
    "hip_mid_x", "hip_mid_y",
]


def _wrist_ear_ratio_min(k: np.ndarray, shoulder_w: float) -> float:
    """Min over 4 wrist-to-ear distances divided by shoulder width."""
    dists = [
        np.linalg.norm(k[10] - k[3]),  # R wrist - L ear
        np.linalg.norm(k[10] - k[4]),  # R wrist - R ear
        np.linalg.norm(k[9] - k[3]),   # L wrist - L ear
        np.linalg.norm(k[9] - k[4]),   # L wrist - R ear
    ]
    return float(min(dists)) / shoulder_w


def dump_one(name: str) -> Path:
    src = _SAMPLES / name
    if not src.exists():
        print(f"  ! {name}  (missing)")
        return None
    o = PointAndCallOrchestrator(_CONFIG)
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
            out = o.process_frame(frame, timestamp=fi / fps)
            row: dict = {"frame": fi, "t": round(fi / fps, 3),
                          "alerts": "|".join(out.alerts) if out.alerts else "",
                          "matcher_progress": "|".join(out.sequence_progress)}
            if out.persons:
                p = out.persons[0]
                d = p.direction_result.debug_info
                k = p.keypoints
                s = p.kp_scores
                sw = float(np.linalg.norm(k[6] - k[5])) + 1e-6
                hip = 0.5 * (k[11] + k[12])
                row.update({
                    "label": p.direction_label,
                    "side": d.get("side"),
                    "elbow_angle": d.get("elbow_angle"),
                    "arm_elevation": d.get("arm_elevation"),
                    "azimuth": d.get("azimuth"),
                    "suppressed_by": d.get("suppressed_by", ""),
                    "wrist_l_x": round(float(k[9, 0]), 1), "wrist_l_y": round(float(k[9, 1]), 1),
                    "wrist_r_x": round(float(k[10, 0]), 1), "wrist_r_y": round(float(k[10, 1]), 1),
                    "shoulder_w_px": round(sw, 1),
                    "wrist_ear_ratio_min": round(_wrist_ear_ratio_min(k, sw), 3),
                    "score_l_shoulder": round(float(s[5]), 3),
                    "score_l_elbow": round(float(s[7]), 3),
                    "score_l_wrist": round(float(s[9]), 3),
                    "score_r_shoulder": round(float(s[6]), 3),
                    "score_r_elbow": round(float(s[8]), 3),
                    "score_r_wrist": round(float(s[10]), 3),
                    "score_nose": round(float(s[0]), 3),
                    "score_l_ear": round(float(s[3]), 3),
                    "score_r_ear": round(float(s[4]), 3),
                    "hip_mid_x": round(float(hip[0]), 1), "hip_mid_y": round(float(hip[1]), 1),
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
