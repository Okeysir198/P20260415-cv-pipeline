"""Evaluate SOTA pose models for safety-poketenashi on 10 sample images.

Compares:
  - Baseline: YOLOv8n-pose (COCO 17-kpt body) — proxy for RTMPose-S.
  - SOTA   : DWPose-L (COCO-WholeBody 133-kpt: body + feet + face + hands).

For each sample we render keypoints + a heuristic "hands_in_pockets" rule
verdict, save the visualization to predict/<model>/, and emit a summary
JSON consumed by QUALITY_REPORT.md.

Run:
    uv run python ai/features/safety-poketenashi/code/eval_sota.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

ROOT = Path(__file__).resolve().parents[4]
FEAT = ROOT / "ai" / "features" / "safety-poketenashi"
PRETRAIN = ROOT / "ai" / "pretrained" / "safety-poketenashi"
SAMPLES = FEAT / "samples"
PRED = FEAT / "predict"

import sys as _sys  # noqa: E402

_sys.path.insert(0, str(ROOT / "ai"))
from utils.viz import VizStyle, annotate_keypoints, classification_banner  # noqa: E402

# COCO-17 indices used by the baseline.
COCO17 = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

# COCO-WholeBody 133-kpt layout (mmpose order):
# 0-16 body, 17-22 feet, 23-90 face, 91-111 left-hand, 112-132 right-hand.
WB_BODY = slice(0, 17)
WB_LHAND = slice(91, 112)
WB_RHAND = slice(112, 133)


@dataclass
class Verdict:
    sample: str
    model: str
    n_kpts: int
    persons: int
    rule_hands_in_pockets: bool
    rule_evidence: str
    notes: str = ""


# ---------------------------------------------------------------------------
# DWPose ONNX wrapper (RTMPose SimCC head)
# ---------------------------------------------------------------------------

class DWPose:
    INPUT_HW = (384, 288)  # H, W

    def __init__(self, onnx_path: Path):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    @staticmethod
    def _affine(box_xyxy: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        """Build affine matrix from person bbox to model input (top-down crop)."""
        x0, y0, x1, y1 = box_xyxy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = x1 - x0, y1 - y0
        oh, ow = out_hw
        aspect = ow / oh
        if bw / bh > aspect:
            bh = bw / aspect
        else:
            bw = bh * aspect
        bw *= 1.25
        bh *= 1.25
        # 3-point affine: src(center, center+w/2, center+h/2) -> dst.
        src = np.array([[cx, cy], [cx + bw / 2, cy], [cx, cy + bh / 2]], dtype=np.float32)
        dst = np.array([[ow / 2, oh / 2], [ow, oh / 2], [ow / 2, oh]], dtype=np.float32)
        return cv2.getAffineTransform(src, dst)

    def __call__(self, img_bgr: np.ndarray, box_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        oh, ow = self.INPUT_HW
        M = self._affine(box_xyxy, self.INPUT_HW)
        crop = cv2.warpAffine(img_bgr, M, (ow, oh), flags=cv2.INTER_LINEAR)
        # RTMPose preprocessing: BGR mean/std.
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        x = (crop.astype(np.float32) - mean) / std
        x = x.transpose(2, 0, 1)[None]  # NCHW
        simcc_x, simcc_y = self.sess.run(None, {self.in_name: x})
        # Decode SimCC: argmax over bins, scale = 2.0 (RTMPose default).
        sx = simcc_x[0].argmax(axis=-1).astype(np.float32) / 2.0
        sy = simcc_y[0].argmax(axis=-1).astype(np.float32) / 2.0
        max_x = simcc_x[0].max(axis=-1)
        max_y = simcc_y[0].max(axis=-1)
        scores = np.minimum(max_x, max_y)
        # Map (sx, sy) in input pixels back to original image via inverse affine.
        Minv = cv2.invertAffineTransform(M)
        ones = np.ones((sx.shape[0], 1), dtype=np.float32)
        pts_in = np.concatenate([sx[:, None], sy[:, None], ones], axis=1)
        pts_orig = pts_in @ Minv.T
        return pts_orig.astype(np.float32), scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Rule: hands_in_pockets
# ---------------------------------------------------------------------------

def rule_hip_17(kpts: np.ndarray, scores: np.ndarray, conf_th: float = 0.3) -> Tuple[bool, str]:
    """17-kpt heuristic: wrist near hip + low wrist confidence (occlusion proxy)."""
    if kpts is None or len(kpts) == 0:
        return False, "no person"
    lw, rw = kpts[9], kpts[10]
    lh, rh = kpts[11], kpts[12]
    lws, rws = scores[9], scores[10]
    ls, rs = kpts[5], kpts[6]
    torso = float(np.linalg.norm(ls - lh) + np.linalg.norm(rs - rh)) / 2 + 1e-6
    dl = float(np.linalg.norm(lw - lh)) / torso
    dr = float(np.linalg.norm(rw - rh)) / torso
    # Wrist near hip AND low wrist confidence (occlusion proxy) on either side.
    hit = (dl < 0.35 and lws < conf_th) or (dr < 0.35 and rws < conf_th)
    ev = f"L d/torso={dl:.2f} conf={lws:.2f}; R d/torso={dr:.2f} conf={rws:.2f}"
    return bool(hit), ev


def rule_hip_wb(kpts: np.ndarray, scores: np.ndarray, conf_th: float = 0.3) -> Tuple[bool, str]:
    """133-kpt rule: hand-keypoint detection rate near hip is what matters.
    If finger keypoints (palm + fingertips) are mostly *missing* and wrist is
    near hip → hands are occluded inside pocket.
    """
    if kpts is None or len(kpts) == 0:
        return False, "no person"
    body = kpts[WB_BODY]
    lw, rw = body[9], body[10]
    lh, rh = body[11], body[12]
    ls, rs = body[5], body[6]
    torso = float(np.linalg.norm(ls - lh) + np.linalg.norm(rs - rh)) / 2 + 1e-6
    near_l = float(np.linalg.norm(lw - lh)) / torso < 0.40
    near_r = float(np.linalg.norm(rw - rh)) / torso < 0.40
    lhand_s = scores[WB_LHAND]
    rhand_s = scores[WB_RHAND]
    lhand_vis = float((lhand_s > conf_th).mean())
    rhand_vis = float((rhand_s > conf_th).mean())
    hidden_l = lhand_vis < 0.30
    hidden_r = rhand_vis < 0.30
    hit = (near_l and hidden_l) or (near_r and hidden_r)
    ev = (
        f"L: near_hip={near_l} hand_vis={lhand_vis:.2f}; "
        f"R: near_hip={near_r} hand_vis={rhand_vis:.2f}"
    )
    return bool(hit), ev


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

SKELETON_17 = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]


def draw_pose(img: np.ndarray, kpts: np.ndarray, scores: np.ndarray,
              wb: bool, conf_th: float = 0.3) -> np.ndarray:
    """Draw COCO-17 skeleton + (optionally) whole-body hand/face kps.

    Input ``img`` is BGR (from cv2.imread); output is BGR for cv2.imwrite.
    Helpers operate on RGB, so convert at boundaries.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    body = kpts[WB_BODY] if wb else kpts
    bs = scores[WB_BODY] if wb else scores

    # Original BGR (0, 200, 255) → RGB (255, 200, 0).
    body_style = VizStyle(kpt_visibility_threshold=conf_th,
                          skeleton_color_rgb=(255, 200, 0))
    rgb = annotate_keypoints(rgb, body, skeleton_edges=SKELETON_17,
                             confidence=bs, style=body_style,
                             color=sv.Color(r=0, g=0, b=255))  # vertices red

    if wb:
        hand_style = VizStyle(kpt_visibility_threshold=conf_th)
        # BGR (0,255,0) = RGB (0,255,0); BGR (255,255,0) = RGB (0,255,255).
        for sl, color_rgb in [(WB_LHAND, (0, 255, 0)), (WB_RHAND, (0, 255, 255))]:
            rgb = annotate_keypoints(rgb, kpts[sl], skeleton_edges=None,
                                     confidence=scores[sl], style=hand_style,
                                     color=sv.Color(r=color_rgb[0], g=color_rgb[1], b=color_rgb[2]))
        # Face: BGR (255, 200, 100) = RGB (100, 200, 255).
        face_style = VizStyle(kpt_visibility_threshold=conf_th)
        rgb = annotate_keypoints(rgb, kpts[23:91], skeleton_edges=None,
                                 confidence=scores[23:91], style=face_style,
                                 color=sv.Color(r=100, g=200, b=255))

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def annotate(img: np.ndarray, label: str, verdict: bool) -> np.ndarray:
    """Overlay a 36px header banner. Input/output BGR."""
    # Original BGR text color: (0,0,200) violation → RGB (200,0,0); (0,180,0) ok → RGB (0,180,0).
    text_rgb = (200, 0, 0) if verdict else (0, 180, 0)
    txt = f"{label}: {'VIOLATION' if verdict else 'OK'}"
    style = VizStyle(banner_height=36, banner_text_scale=0.7)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = classification_banner(rgb, txt, style=style, position="overlay_top",
                                bg_color_rgb=(40, 40, 40), text_color_rgb=text_rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from ultralytics import YOLO

    samples = sorted(SAMPLES.glob("*.jpg"))
    assert len(samples) >= 10, f"need 10 samples, got {len(samples)}"
    samples = samples[:10]

    # Detector + 17-kpt baseline (one model).
    _pose_pt = ROOT / "ai/pretrained/nitto_denko/pose_estimation/yolov8n-pose.pt"
    if not _pose_pt.exists():
        raise FileNotFoundError(f"yolov8n-pose weights not found: {_pose_pt}")
    yolo_pose = YOLO(str(_pose_pt))
    dwpose = DWPose(PRETRAIN / "dw-ll_ucoco_384.onnx")

    out_baseline = PRED / "yolov8n-pose_body17"
    out_dwpose = PRED / "dwpose-l_wholebody133"
    out_baseline.mkdir(parents=True, exist_ok=True)
    out_dwpose.mkdir(parents=True, exist_ok=True)

    verdicts: List[Verdict] = []

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"skip {img_path.name}: unreadable")
            continue

        # ---- Baseline 17-kpt ----
        res = yolo_pose.predict(img, conf=0.35, verbose=False)[0]
        if res.keypoints is not None and len(res.keypoints) > 0:
            # Pick largest person.
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = int(areas.argmax())
            kpts17 = res.keypoints.xy.cpu().numpy()[idx]
            sc17 = res.keypoints.conf.cpu().numpy()[idx]
            box = boxes[idx]
            persons17 = len(boxes)
        else:
            kpts17, sc17, box, persons17 = None, None, None, 0

        if kpts17 is not None:
            v17, ev17 = rule_hip_17(kpts17, sc17)
            vis17 = draw_pose(img, kpts17, sc17, wb=False)
            vis17 = annotate(vis17, "body17 hands_in_pockets", v17)
        else:
            v17, ev17 = False, "no person detected"
            vis17 = annotate(img.copy(), "body17 hands_in_pockets", False)
        cv2.imwrite(str(out_baseline / img_path.name), vis17)
        verdicts.append(Verdict(img_path.name, "yolov8n-pose_body17", 17, persons17, v17, ev17))

        # ---- DWPose-L wholebody ----
        if box is None:
            # Fallback: use whole-image bbox.
            h, w = img.shape[:2]
            box = np.array([0, 0, w, h], dtype=np.float32)
        kpts133, sc133 = dwpose(img, box)
        v133, ev133 = rule_hip_wb(kpts133, sc133)
        vis133 = draw_pose(img, kpts133, sc133, wb=True)
        vis133 = annotate(vis133, "wb133 hands_in_pockets", v133)
        cv2.imwrite(str(out_dwpose / img_path.name), vis133)
        verdicts.append(Verdict(img_path.name, "dwpose-l_wholebody133", 133,
                                max(persons17, 1), v133, ev133))

        print(f"{img_path.name}: body17={v17}  wb133={v133}")

    # Dump summary.
    summary = [asdict(v) for v in verdicts]
    (PRED / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {PRED/'summary.json'}")


if __name__ == "__main__":
    main()
