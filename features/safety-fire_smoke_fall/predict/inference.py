"""Self-contained video inference for safety-fire_smoke_fall.

Zero dependency on `core/` — only `transformers`, `torch`, `torchvision`, `cv2`,
`numpy`, `pyyaml`, `loguru`, and a system `ffmpeg`. Copy this file (plus the
model checkpoint dir + 05_data.yaml) to any host with those deps to deploy.

Three stages:
    extract  — sample frames from a video at target fps (default 5)
    infer    — batched HF DETR inference; raw boxes/scores/labels per frame
               (conf=0.0, NO filtering) saved to raw_predictions.json
    render   — read raw → per-class NMS → per-class threshold → annotated H.264 mp4

`all` runs extract + infer + render in one shot.

NMS notes:
    Although DETR is officially "NMS-free", in practice noisy real-world video
    produces many overlapping low-confidence proposals per object. This script
    applies torchvision.ops.nms per-class at render time (cheap, no GPU needed).

Usage:
    # one-shot at default conf=0.3, NMS IoU=0.5
    uv run features/safety-fire_smoke_fall/predict/inference.py all \\
        --video features/safety-fire_smoke_fall/samples/L20260513170138738.mp4

    # re-render with per-class thresholds (no GPU)
    uv run features/safety-fire_smoke_fall/predict/inference.py render \\
        --predict-dir features/safety-fire_smoke_fall/predict/L20260513170138738 \\
        --conf 0.1 --conf-per-class fire=0.05 smoke=0.05 fallen_person=0.30 \\
        --nms-iou 0.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger

THIS_DIR = Path(__file__).resolve().parent
# Primary model = fire-tuned (best per-class fire 0.203). Use model/best_overall/ if you
# need higher overall mAP50 (0.584 vs 0.534). Both class names come from config.json::id2label.
_DEFAULT_MODEL = THIS_DIR / "model/best_fire"

_CLASS_COLORS_BGR = {
    "fire":          (0, 64, 255),    # red-orange
    "smoke":         (200, 200, 200), # light grey
    "fallen_person": (0, 200, 255),   # yellow
}
_DEFAULT_COLOR = (0, 255, 0)


# ---------------------------------------------------------------------------
# Stage 1 — extract frames at target fps
# ---------------------------------------------------------------------------
def extract_frames(video_path: Path, out_dir: Path, target_fps: float = 5.0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / target_fps))

    logger.info(
        "Video {} | {:.2f} fps × {} frames → sample every {} (~{:.2f} fps)",
        video_path.name, src_fps, total, stride, src_fps / stride,
    )

    written = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            cv2.imwrite(str(out_dir / f"{written:06d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        idx += 1
    cap.release()
    logger.info("Extracted {} frames → {}", written, out_dir)
    return written


# ---------------------------------------------------------------------------
# Stage 2 — load HF DETR model and run batched inference
# ---------------------------------------------------------------------------
def _load_hf_model(model_path: Path, device: torch.device):
    """Load an HF DETR-family model, stripping `hf_model.` prefix if present."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(str(model_path))

    bin_path = model_path / "pytorch_model.bin"
    if bin_path.exists():
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        if any(k.startswith("hf_model.") for k in sd):
            stripped = {k.removeprefix("hf_model."): v for k, v in sd.items()}
            tmp = Path(tempfile.mkdtemp())
            shutil.copy(model_path / "config.json", tmp / "config.json")
            preproc = model_path / "preprocessor_config.json"
            if preproc.exists():
                shutil.copy(preproc, tmp / "preprocessor_config.json")
            torch.save(stripped, tmp / "pytorch_model.bin")
            logger.info("Stripped hf_model. prefix → temp dir {}", tmp)
            model = AutoModelForObjectDetection.from_pretrained(str(tmp))
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            model = AutoModelForObjectDetection.from_pretrained(str(model_path))
    else:
        model = AutoModelForObjectDetection.from_pretrained(str(model_path))

    model.to(device).eval()
    return model, processor


def run_inference(
    frames_dir: Path, model_path: Path, out_json: Path, batch_size: int = 16,
) -> None:
    """Run model on every frame; dump raw predictions (no filtering)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    model, processor = _load_hf_model(model_path, device)
    class_names = {int(k): v for k, v in model.config.id2label.items()}
    logger.info("Loaded model {} ({}) | classes={}",
                model_path, type(model).__name__, class_names)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames in {frames_dir}")
    logger.info("Inferring {} frames at batch={}", len(frame_paths), batch_size)

    all_records: list[dict] = []
    t0 = time.perf_counter()
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images_bgr = [cv2.imread(str(p)) for p in batch_paths]
        images_rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
        target_sizes = torch.tensor(
            [[im.shape[0], im.shape[1]] for im in images_bgr], device=device
        )
        inputs = processor(images=images_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )
        for path, r in zip(batch_paths, results, strict=True):
            all_records.append({
                "frame":  path.name,
                "boxes":  r["boxes"].cpu().numpy().astype(np.float32).tolist(),
                "scores": r["scores"].cpu().numpy().astype(np.float32).tolist(),
                "labels": r["labels"].cpu().numpy().astype(np.int64).tolist(),
            })
        if (start // batch_size) % 10 == 0:
            logger.info("  {}/{} frames",
                        min(start + batch_size, len(frame_paths)), len(frame_paths))

    dt = time.perf_counter() - t0
    logger.info("Inference done in {:.1f}s ({:.1f} fps)", dt, len(frame_paths) / dt)

    out_json.write_text(json.dumps({
        "model_path":  str(model_path),
        "class_names": class_names,
        "frames_dir":  str(frames_dir),
        "predictions": all_records,
    }, indent=2))
    logger.info("Wrote raw predictions → {}", out_json)


# ---------------------------------------------------------------------------
# Stage 3 — per-class NMS + per-class threshold + H.264 mp4 via ffmpeg
# ---------------------------------------------------------------------------
def _apply_per_class_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run torchvision NMS per class; return filtered arrays."""
    if boxes.size == 0:
        return boxes, scores, labels
    keep_idx: list[int] = []
    for cls_id in np.unique(labels):
        mask = labels == cls_id
        cls_keep = torchvision.ops.nms(
            torch.from_numpy(boxes[mask]),
            torch.from_numpy(scores[mask]),
            iou,
        ).numpy()
        keep_idx.extend(np.where(mask)[0][cls_keep].tolist())
    keep_idx = sorted(keep_idx)
    return boxes[keep_idx], scores[keep_idx], labels[keep_idx]


def render_video(
    predict_dir: Path, out_video: Path,
    conf: float = 0.3, conf_per_class: dict[str, float] | None = None,
    nms_iou: float = 0.5, fps: float = 5.0,
) -> None:
    raw_json = predict_dir / "raw_predictions.json"
    payload = json.loads(raw_json.read_text())
    class_names = {int(k): v for k, v in payload["class_names"].items()}
    frames_dir = predict_dir / "frames"

    thresholds = {name: conf for name in class_names.values()}
    if conf_per_class:
        thresholds.update(conf_per_class)
    logger.info("Render | thresholds={} | nms_iou={}", thresholds, nms_iou)

    sample = cv2.imread(str(frames_dir / payload["predictions"][0]["frame"]))
    h, w = sample.shape[:2]

    # ffmpeg pipe writer — H.264, yuv420p, faststart → playable everywhere
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video),
        ],
        stdin=subprocess.PIPE,
    )

    n_kept = 0
    n_raw = 0
    for record in payload["predictions"]:
        frame = cv2.imread(str(frames_dir / record["frame"]))
        boxes  = np.asarray(record["boxes"],  dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(record["labels"], dtype=np.int64).reshape(-1)
        n_raw += boxes.shape[0]

        # 1) per-class confidence threshold
        if boxes.size > 0:
            keep_mask = np.array([
                scores[i] >= thresholds.get(class_names.get(int(labels[i]), ""), conf)
                for i in range(len(labels))
            ])
            boxes, scores, labels = boxes[keep_mask], scores[keep_mask], labels[keep_mask]

        # 2) per-class NMS
        if boxes.size > 0:
            boxes, scores, labels = _apply_per_class_nms(boxes, scores, labels, nms_iou)

        # 3) draw
        for box, score, label in zip(boxes, scores, labels, strict=True):
            name = class_names.get(int(label), str(int(label)))
            x1, y1, x2, y2 = box.astype(int)
            color = _CLASS_COLORS_BGR.get(name, _DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            n_kept += 1

        ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    rc = ffmpeg.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}")
    logger.info(
        "Wrote {} | drew {} boxes (raw {} → kept {:.1%})",
        out_video, n_kept, n_raw, n_kept / max(n_raw, 1),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_per_class(items: list[str] | None) -> dict[str, float]:
    return {k: float(v) for k, v in (s.split("=", 1) for s in (items or []))}


def _video_predict_dir(video: Path) -> Path:
    return THIS_DIR / video.stem


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="mode", required=True)

    model_kw = dict(default=str(_DEFAULT_MODEL), help=f"Model dir (HF format) [default: {_DEFAULT_MODEL.name}]")

    ex = sub.add_parser("extract", help="Extract frames at target fps")
    ex.add_argument("--video", required=True)
    ex.add_argument("--fps", type=float, default=5.0)

    inf = sub.add_parser("infer", help="Batched HF inference on extracted frames")
    inf.add_argument("--video", required=True, help="Used to derive predict/<stem>/")
    inf.add_argument("--model", **model_kw)
    inf.add_argument("--batch-size", type=int, default=16)

    rd = sub.add_parser("render", help="NMS + threshold + H.264 mp4")
    rd.add_argument("--predict-dir", required=True)
    rd.add_argument("--conf", type=float, default=0.3, help="Global confidence threshold")
    rd.add_argument("--conf-per-class", nargs="*", default=[],
                    help="Per-class overrides, e.g. fire=0.05 smoke=0.05")
    rd.add_argument("--nms-iou", type=float, default=0.5, help="Per-class NMS IoU threshold")
    rd.add_argument("--fps", type=float, default=5.0)
    rd.add_argument("--out", default=None,
                    help="Output mp4; default = predict-dir/rendered_conf<X>_nms<Y>.mp4")

    all_p = sub.add_parser("all", help="extract + infer + render")
    all_p.add_argument("--video", required=True)
    all_p.add_argument("--model", **model_kw)
    all_p.add_argument("--fps", type=float, default=5.0)
    all_p.add_argument("--batch-size", type=int, default=16)
    all_p.add_argument("--conf", type=float, default=0.3)
    all_p.add_argument("--conf-per-class", nargs="*", default=[])
    all_p.add_argument("--nms-iou", type=float, default=0.5)

    args = p.parse_args()

    if args.mode == "extract":
        video = Path(args.video)
        extract_frames(video, _video_predict_dir(video) / "frames", args.fps)

    elif args.mode == "infer":
        video = Path(args.video)
        pdir = _video_predict_dir(video)
        run_inference(pdir / "frames", Path(args.model),
                      pdir / "raw_predictions.json", args.batch_size)

    elif args.mode == "render":
        pdir = Path(args.predict_dir)
        tag = f"conf{args.conf:.2f}_nms{args.nms_iou:.2f}"
        out = Path(args.out) if args.out else pdir / f"rendered_{tag}.mp4"
        render_video(pdir, out, args.conf, _parse_per_class(args.conf_per_class),
                     args.nms_iou, args.fps)

    elif args.mode == "all":
        video = Path(args.video)
        pdir = _video_predict_dir(video)
        extract_frames(video, pdir / "frames", args.fps)
        run_inference(pdir / "frames", Path(args.model),
                      pdir / "raw_predictions.json", args.batch_size)
        tag = f"conf{args.conf:.2f}_nms{args.nms_iou:.2f}"
        render_video(pdir, pdir / f"rendered_{tag}.mp4",
                     args.conf, _parse_per_class(args.conf_per_class),
                     args.nms_iou, args.fps)


if __name__ == "__main__":
    main()
