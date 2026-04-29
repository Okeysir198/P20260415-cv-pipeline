"""Per-frame debug-CSV dump for safety-poketenashi_no_handrail failure analysis.

For each requested video, runs ``NoHandrailPredictor`` (with the per-video
handrail polygon from ``eval/ground_truth.json`` applied at runtime) and writes
a CSV with every frame's pose-rule debug fields:

    frame, t, label, alert,
    wrist_l_x, wrist_l_y, wrist_r_x, wrist_r_y,
    dist_l_to_zone_px, dist_r_to_zone_px,
    score_l_wrist, score_r_wrist, score_nose, score_l_shoulder, score_r_shoulder,
    score_l_hip, score_r_hip, score_l_knee, score_r_knee, score_l_ankle, score_r_ankle

Videos with no handrail polygon (``handrail_zones_norm: null``) are skipped —
the rule cannot fire without one, so the CSV would be vacuous.

Usage:
    uv run python features/safety-poketenashi_no_handrail/code/dump_debug.py \\
        VIDEO_NAME [VIDEO_NAME ...]
    # or, no args -> dumps every video listed in eval/ground_truth.json that has
    # a polygon configured.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

_FEAT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FEAT / "code"))
sys.path.insert(0, str(_FEAT.parent.parent))

from predictor import NoHandrailPredictor  # noqa: E402

_CONFIG = _FEAT / "configs" / "10_inference.yaml"
_SAMPLES = _FEAT / "samples"
_EVAL_DIR = _FEAT / "eval"
_GT_PATH = _EVAL_DIR / "ground_truth.json"

_FIELDS = [
    "frame", "t", "label", "alert",
    "wrist_l_x", "wrist_l_y", "wrist_r_x", "wrist_r_y",
    "dist_l_to_zone_px", "dist_r_to_zone_px",
    "score_l_wrist", "score_r_wrist",
    "score_nose", "score_l_shoulder", "score_r_shoulder",
    "score_l_hip", "score_r_hip",
    "score_l_knee", "score_r_knee",
    "score_l_ankle", "score_r_ankle",
]


def _load_zones_for(name: str) -> list | None:
    if not _GT_PATH.exists():
        return None
    gt = json.loads(_GT_PATH.read_text())
    meta = gt.get("videos", {}).get(name, {})
    return meta.get("handrail_zones_norm")


def dump_one(name: str) -> Path | None:
    src = _SAMPLES / name
    if not src.exists():
        print(f"  ! {name}  (missing video)")
        return None
    zones_norm = _load_zones_for(name)
    if zones_norm is None:
        print(f"  ~ {name}  (skipped — no polygon configured in ground_truth.json)")
        return None

    predictor = NoHandrailPredictor(_CONFIG)
    predictor._rule._zones = zones_norm  # noqa: SLF001 — harness reach-in

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  ! {name}  (cannot open)")
        return None
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
            out = predictor.process_frame(frame)
            row: dict = {
                "frame": fi,
                "t": round(fi / fps, 3),
                "alert": "|".join(out.alerts) if out.alerts else "",
            }
            if out.persons:
                p = out.persons[0]
                k = p.keypoints
                s = p.kp_scores
                d = p.rule_result.debug_info
                row.update({
                    "label": p.rule_result.behavior if p.rule_result.triggered else "ok",
                    "wrist_l_x": round(float(k[9, 0]), 1),
                    "wrist_l_y": round(float(k[9, 1]), 1),
                    "wrist_r_x": round(float(k[10, 0]), 1),
                    "wrist_r_y": round(float(k[10, 1]), 1),
                    "dist_l_to_zone_px": d.get("left_dist_px"),
                    "dist_r_to_zone_px": d.get("right_dist_px"),
                    "score_l_wrist": round(float(s[9]), 3),
                    "score_r_wrist": round(float(s[10]), 3),
                    "score_nose": round(float(s[0]), 3),
                    "score_l_shoulder": round(float(s[5]), 3),
                    "score_r_shoulder": round(float(s[6]), 3),
                    "score_l_hip": round(float(s[11]), 3),
                    "score_r_hip": round(float(s[12]), 3),
                    "score_l_knee": round(float(s[13]), 3),
                    "score_r_knee": round(float(s[14]), 3),
                    "score_l_ankle": round(float(s[15]), 3),
                    "score_r_ankle": round(float(s[16]), 3),
                })
            else:
                row["label"] = "no_person"
            w.writerow(row)
            fi += 1
    cap.release()
    print(f"  + {name}  -> {out_path.relative_to(_FEAT.parent.parent)}  ({fi}/{n} frames)")
    return out_path


def main() -> None:
    if sys.argv[1:]:
        targets = sys.argv[1:]
    else:
        if not _GT_PATH.exists():
            print(f"[dump_debug] no ground_truth.json at {_GT_PATH}")
            return
        gt = json.loads(_GT_PATH.read_text())
        targets = list(gt.get("videos", {}).keys())
    print(f"Dumping {len(targets)} video(s) to {_EVAL_DIR.relative_to(_FEAT.parent.parent)}")
    for name in targets:
        dump_one(name)


if __name__ == "__main__":
    main()
