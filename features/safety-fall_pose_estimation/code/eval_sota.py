"""SOTA pretrained pose model quick-eval for fall detection.

Two models:
  * Two-stage baseline: YOLOX-tiny (humanart) + RTMPose-S (body7) via rtmlib
  * One-stage         : RTMO-S (body7) via rtmlib

For each sample image: detect persons, predict 17 COCO keypoints, draw the
skeleton, apply a simple geometric fall rule (bbox aspect ratio + torso angle)
and write a verdict overlay to predict/<model_name>/<image>.

Run:
    uv run --extra all python ai/features/safety-fall_pose_estimation/code/eval_sota.py

Outputs:
    ai/features/safety-fall_pose_estimation/predict/<model>/*.jpg
    ai/features/safety-fall_pose_estimation/predict/results.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMO, YOLOX, RTMPose, draw_skeleton

FEATURE = Path(__file__).resolve().parents[1]
SAMPLES = FEATURE / "samples"
PRED = FEATURE / "predict"
PRED.mkdir(parents=True, exist_ok=True)

# rtmlib auto-downloads ONNX SDK packages to ~/.cache/rtmlib
DET_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
POSE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
RTMO_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip"

# COCO-17 keypoint indices
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10,
    "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16,
}
KPT_THR = 0.35


def fall_verdict(kpts: np.ndarray, scores: np.ndarray) -> tuple[str, dict]:
    """Geometric rule. Returns (verdict, debug)."""
    if kpts is None or len(kpts) == 0:
        return "no_person", {}
    # Use highest-confidence person
    person_scores = scores.mean(axis=1)
    idx = int(np.argmax(person_scores))
    k, s = kpts[idx], scores[idx]

    def avg(a, b):
        if s[a] < KPT_THR or s[b] < KPT_THR:
            return None
        return (k[a] + k[b]) / 2

    sho = avg(KP["l_sho"], KP["r_sho"])
    hip = avg(KP["l_hip"], KP["r_hip"])
    ank_pts = [k[i] for i in (KP["l_ank"], KP["r_ank"]) if s[i] >= KPT_THR]
    visible = k[s >= KPT_THR]
    if len(visible) < 4:
        return "low_quality", {"visible_kpts": int(len(visible))}

    # bbox of visible kpts
    x0, y0 = visible[:, 0].min(), visible[:, 1].min()
    x1, y1 = visible[:, 0].max(), visible[:, 1].max()
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    aspect = w / h  # >1 -> wider than tall

    torso_angle = None
    if sho is not None and hip is not None:
        dx, dy = hip[0] - sho[0], hip[1] - sho[1]
        # angle from vertical: 0 = upright, 90 = horizontal
        torso_angle = abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    # Hip above ankle margin
    hip_above_ank = None
    if hip is not None and ank_pts:
        ank_y = float(np.mean([p[1] for p in ank_pts]))
        hip_above_ank = ank_y - hip[1]  # positive = upright

    fall = False
    reasons = []
    if aspect > 1.2:
        fall = True
        reasons.append(f"aspect={aspect:.2f}")
    if torso_angle is not None and torso_angle > 55:
        fall = True
        reasons.append(f"torso={torso_angle:.0f}deg")
    if hip_above_ank is not None and hip_above_ank < h * 0.15:
        fall = True
        reasons.append(f"hip~ank(d={hip_above_ank:.0f})")

    return ("fall" if fall else "upright"), {
        "aspect": round(aspect, 2),
        "torso_deg": None if torso_angle is None else round(torso_angle, 1),
        "hip_above_ank": None if hip_above_ank is None else round(hip_above_ank, 1),
        "reasons": reasons,
        "visible_kpts": int(len(visible)),
    }


def annotate(img: np.ndarray, kpts, scores, verdict: str, dbg: dict) -> np.ndarray:
    out = img.copy()
    if kpts is not None and len(kpts) > 0:
        out = draw_skeleton(out, kpts, scores, openpose_skeleton=False, kpt_thr=KPT_THR)
    color = {"fall": (0, 0, 255), "upright": (0, 200, 0)}.get(verdict, (0, 165, 255))
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), -1)
    keys = ("aspect", "torso_deg", "visible_kpts")
    txt = f"{verdict.upper()}  " + " ".join(f"{k}={dbg[k]}" for k in keys if k in dbg)
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def run_model(name: str, infer_fn, samples: list[Path]) -> dict:
    out_dir = PRED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            rows.append({"image": img_path.name, "error": "unreadable"})
            continue
        t0 = time.time()
        try:
            kpts, scores = infer_fn(img)
        except Exception as e:  # noqa: BLE001
            rows.append({"image": img_path.name, "error": str(e)[:200]})
            continue
        dt = (time.time() - t0) * 1000
        verdict, dbg = fall_verdict(np.asarray(kpts), np.asarray(scores))
        vis = annotate(img, kpts, scores, verdict, dbg)
        cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), vis)
        rows.append({
            "image": img_path.name, "ms": round(dt, 1),
            "n_persons": int(len(kpts)) if kpts is not None else 0,
            "verdict": verdict, **dbg,
        })
        print(f"  [{name}] {img_path.name:25s} {dt:6.1f}ms  -> {verdict}  {dbg.get('reasons', [])}")
    return {"model": name, "results": rows}


def main():
    exts = (".jpg", ".jpeg", ".png")
    samples = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in exts)
    assert samples, f"no samples in {SAMPLES}"
    print(f"Found {len(samples)} samples")

    print("\n== loading two-stage YOLOX-tiny + RTMPose-S ==")
    det = YOLOX(DET_URL, model_input_size=(416, 416), backend="onnxruntime", device="cpu")
    pose = RTMPose(POSE_URL, model_input_size=(192, 256), backend="onnxruntime", device="cpu")

    def two_stage(img):
        bboxes = det(img)
        return pose(img, bboxes=bboxes)

    print("\n== loading one-stage RTMO-S ==")
    rtmo = RTMO(RTMO_URL, model_input_size=(640, 640), backend="onnxruntime", device="cpu")

    out = []
    print("\n--- two_stage_yoloxtiny_rtmposes ---")
    out.append(run_model("two_stage_yoloxtiny_rtmposes", two_stage, samples))
    print("\n--- one_stage_rtmo_s ---")
    out.append(run_model("one_stage_rtmo_s", rtmo, samples))

    (PRED / "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved results to {PRED/'results.json'}")


if __name__ == "__main__":
    main()
