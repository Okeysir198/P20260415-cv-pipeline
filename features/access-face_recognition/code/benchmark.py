"""Benchmark pretrained face detection and recognition models.

Evaluates:
  Face detectors: yunet_2023mar.onnx, yunet_2023mar_int8.onnx, yolov8n-face.pt
  Face recognizers: sface_2021dec.onnx, sface_2021dec_int8.onnx
  InsightFace bundles: buffalo_l, buffalo_m, buffalo_s, buffalo_sc, antelopev2 (if installed)

Usage:
    uv run features/access-face_recognition/code/benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

FEATURE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = FEATURE_DIR / "samples"
EVAL_DIR = FEATURE_DIR / "eval"
WEIGHTS_DIR = REPO / "pretrained" / "access-face_recognition"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Enrollment: _1 images for these identities
ENROLL_NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
# Test: _2 images for these identities
TEST_NAMES = ["alice", "bob", "carol", "dave", "eve"]
SPOOF_STEMS = {"spoof_phone_alice", "spoof_print_alice"}

SIMILARITY_THRESHOLD = 0.5

_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _align_face(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    dst = _ARCFACE_DST * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(landmarks[:3], dst[:3])
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


def _read_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img if img is not None else None


# ---------------------------------------------------------------------------
# YuNet detector
# ---------------------------------------------------------------------------

class _YuNetDetector:
    def __init__(self, model_path: Path) -> None:
        self._det = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=50
        )

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (bbox_xyxy, landmarks_5x2, raw_row) for the largest face, or None."""
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = face[:4]
        bbox = np.array([x, y, x + fw, y + fh], dtype=np.float32)
        lm_raw = face[4:14].reshape(5, 2)
        lm = np.stack([lm_raw[1], lm_raw[0], lm_raw[2], lm_raw[4], lm_raw[3]]).astype(np.float32)
        return bbox, lm, face


# ---------------------------------------------------------------------------
# SFace recognizer
# ---------------------------------------------------------------------------

class _SFaceRecognizer:
    def __init__(self, model_path: Path) -> None:
        self._rec = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, img_bgr: np.ndarray, yunet_raw: np.ndarray) -> np.ndarray:
        aligned = self._rec.alignCrop(img_bgr, yunet_raw)
        feat = self._rec.feature(aligned).reshape(-1)
        return _l2_normalize(feat)


# ---------------------------------------------------------------------------
# YOLOv8-face detector
# ---------------------------------------------------------------------------

class _Yolov8FaceDetector:
    def __init__(self, model_path: Path) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, None] | None:
        """Return (bbox_xyxy, landmarks_5x2, None) for the largest face, or None."""
        results = self._model(img, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None
        # Pick largest face by area
        best_box = None
        best_area = -1.0
        best_kps = None
        for i, box in enumerate(results.boxes):
            xyxy = box.xyxy.cpu().numpy()[0]
            area = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
            if area > best_area:
                best_area = area
                best_box = xyxy
                # yolov8-face exposes keypoints in results.keypoints
                if results.keypoints is not None and i < len(results.keypoints):
                    kps = results.keypoints[i].xy.cpu().numpy().reshape(5, 2)
                    best_kps = kps.astype(np.float32)

        if best_box is None:
            return None
        # No raw YuNet row — SFace cannot use alignCrop for this detector
        return best_box.astype(np.float32), best_kps, None


# ---------------------------------------------------------------------------
# Detection-only benchmark
# ---------------------------------------------------------------------------

def _benchmark_detector(detector_name: str, detector) -> dict:
    """Count how many sample images produce exactly 1 detected face. Also measure latency."""
    all_images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    total = len(all_images)
    exactly_one = 0
    detected_any = 0
    latencies: list[float] = []

    for img_path in all_images:
        img = _read_image(img_path)
        if img is None:
            continue
        t0 = time.perf_counter()
        result = detector.detect(img)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result is not None:
            detected_any += 1
            exactly_one += 1  # detect() returns largest face — it's 1 if result is not None

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "status": "ok",
        "total_images": total,
        "detected_at_least_one": detected_any,
        "detection_rate": round(detected_any / total, 3) if total > 0 else 0.0,
        "latency_ms": round(avg_lat, 1),
    }


# ---------------------------------------------------------------------------
# Recognition benchmark (enroll _1, test _2 + spoofs)
# ---------------------------------------------------------------------------

def _benchmark_recognition(
    detector_name: str,
    detector,
    recognizer_name: str,
    embed_fn,  # (img, raw_row, landmarks) -> np.ndarray | None
) -> dict:
    """Enroll _1 images, test on _2 + spoof images. Return rank-1 accuracy."""
    gallery: dict[str, np.ndarray] = {}

    # Enroll
    for name in ENROLL_NAMES:
        img_path = SAMPLES_DIR / f"{name}_1.jpg"
        if not img_path.exists():
            continue
        img = _read_image(img_path)
        if img is None:
            continue
        result = detector.detect(img)
        if result is None:
            continue
        bbox, lm, raw = result
        emb = embed_fn(img, raw, lm)
        if emb is not None:
            gallery[name] = emb

    # Test
    test_images = []
    for name in TEST_NAMES:
        test_images.append((SAMPLES_DIR / f"{name}_2.jpg", name, False))
    for stem in SPOOF_STEMS:
        test_images.append((SAMPLES_DIR / f"{stem}.jpg", "unknown", True))

    results_log: list[dict] = []
    correct = 0
    total = 0

    for img_path, expected_identity, is_spoof in test_images:
        if not img_path.exists():
            continue
        img = _read_image(img_path)
        if img is None:
            continue

        det_result = detector.detect(img)
        if det_result is None:
            predicted = "no_face"
            similarity = -1.0
        else:
            bbox, lm, raw = det_result
            emb = embed_fn(img, raw, lm)
            if emb is None or not gallery:
                predicted = "unknown"
                similarity = -1.0
            else:
                best_name = "unknown"
                best_sim = -1.0
                for name, gallery_emb in gallery.items():
                    sim = _cosine(emb, gallery_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                similarity = best_sim
                predicted = best_name if best_sim >= SIMILARITY_THRESHOLD else "unknown"

        if is_spoof:
            verdict_correct = predicted == "unknown"
        else:
            verdict_correct = predicted == expected_identity

        if verdict_correct:
            correct += 1
        total += 1

        results_log.append({
            "image": img_path.name,
            "expected": expected_identity,
            "predicted": predicted,
            "similarity": round(float(similarity), 4),
            "correct": verdict_correct,
            "is_spoof": is_spoof,
        })

    non_spoof = [r for r in results_log if not r["is_spoof"]]
    spoof = [r for r in results_log if r["is_spoof"]]
    rank1 = sum(1 for r in non_spoof if r["correct"]) / len(non_spoof) if non_spoof else 0.0
    spoof_reject_rate = sum(1 for r in spoof if r["correct"]) / len(spoof) if spoof else 0.0

    return {
        "status": "ok",
        "gallery_enrolled": sorted(gallery.keys()),
        "rank1_accuracy": round(rank1, 3),
        "spoof_reject_rate": round(spoof_reject_rate, 3),
        "correct": correct,
        "total": total,
        "details": results_log,
    }


# ---------------------------------------------------------------------------
# InsightFace bundle benchmark
# ---------------------------------------------------------------------------

def _benchmark_insightface_bundle(bundle_zip: Path) -> dict:
    """Try loading an InsightFace bundle via insightface.app.FaceAnalysis."""
    try:
        import insightface  # noqa: PLC0415
    except ImportError:
        return {"status": "skipped", "reason": "insightface package not installed"}

    pack_name = bundle_zip.stem
    pack_root = bundle_zip.parent

    try:
        app = insightface.app.FaceAnalysis(
            name=pack_name, root=str(pack_root), providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)
    except Exception as exc:
        return {"status": "error", "error_msg": str(exc)}

    gallery: dict[str, np.ndarray] = {}

    # Enroll _1 images
    for name in ENROLL_NAMES:
        img_path = SAMPLES_DIR / f"{name}_1.jpg"
        if not img_path.exists():
            continue
        img = _read_image(img_path)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        # Largest face
        face = max(faces, key=lambda f: float(
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        gallery[name] = _l2_normalize(face.embedding)

    # Test
    test_images = []
    for name in TEST_NAMES:
        test_images.append((SAMPLES_DIR / f"{name}_2.jpg", name, False))
    for stem in SPOOF_STEMS:
        test_images.append((SAMPLES_DIR / f"{stem}.jpg", "unknown", True))

    results_log: list[dict] = []
    correct = 0
    total = 0
    latencies: list[float] = []

    for img_path, expected_identity, is_spoof in test_images:
        if not img_path.exists():
            continue
        img = _read_image(img_path)
        if img is None:
            continue

        t0 = time.perf_counter()
        faces = app.get(img)
        latencies.append((time.perf_counter() - t0) * 1000)

        if not faces or not gallery:
            predicted = "unknown"
            similarity = -1.0
        else:
            face = max(faces, key=lambda f: float(
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ))
            emb = _l2_normalize(face.embedding)
            best_name, best_sim = "unknown", -1.0
            for name, gallery_emb in gallery.items():
                sim = _cosine(emb, gallery_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            similarity = best_sim
            predicted = best_name if best_sim >= SIMILARITY_THRESHOLD else "unknown"

        verdict_correct = (predicted == "unknown") if is_spoof else (predicted == expected_identity)
        if verdict_correct:
            correct += 1
        total += 1
        results_log.append({
            "image": img_path.name,
            "expected": expected_identity,
            "predicted": predicted,
            "correct": verdict_correct,
            "is_spoof": is_spoof,
        })

    non_spoof = [r for r in results_log if not r["is_spoof"]]
    spoof_tests = [r for r in results_log if r["is_spoof"]]
    rank1 = sum(1 for r in non_spoof if r["correct"]) / len(non_spoof) if non_spoof else 0.0
    spoof_reject = sum(1 for r in spoof_tests if r["correct"]) / len(spoof_tests) if spoof_tests else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "status": "ok",
        "gallery_enrolled": sorted(gallery.keys()),
        "rank1_accuracy": round(rank1, 3),
        "spoof_reject_rate": round(spoof_reject, 3),
        "latency_ms": round(avg_lat, 1),
        "correct": correct,
        "total": total,
        "details": results_log,
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _write_report(det_results: list[dict], recog_results: list[dict]) -> str:
    lines = [
        "# Face Recognition — Pretrained Model Benchmark",
        f"Date: {date.today().isoformat()}",
        f"Samples: {len(ENROLL_NAMES)} enrolled identities, "
        f"{len(TEST_NAMES)} test identities, {len(SPOOF_STEMS)} spoof images",
        f"Similarity threshold: {SIMILARITY_THRESHOLD}",
        "",
        "## Face Detectors",
        "",
        "| Model | Detection Rate | Latency ms | Status |",
        "| --- | --- | --- | --- |",
    ]

    for r in det_results:
        if r.get("status") == "ok":
            lines.append(
                f"| {r['model']} | {r['detection_rate']} "
                f"({r['detected_at_least_one']}/{r['total_images']}) "
                f"| {r['latency_ms']} | ok |"
            )
        else:
            lines.append(
                f"| {r['model']} | — | — | {r.get('error_msg', r.get('status'))} |"
            )

    lines += [
        "",
        "## Face Recognition Pipelines",
        "",
        "Enrollment: `_1` images. Test: `_2` images + spoof images.",
        "",
        "| Pipeline | Rank-1 Accuracy | Spoof Reject Rate | Correct/Total | Status |",
        "| --- | --- | --- | --- | --- |",
    ]

    for r in recog_results:
        if r.get("status") == "ok":
            lines.append(
                f"| {r['model']} | {r['rank1_accuracy']} "
                f"| {r['spoof_reject_rate']} "
                f"| {r['correct']}/{r['total']} | ok |"
            )
        elif r.get("status") == "skipped":
            lines.append(f"| {r['model']} | — | — | — | skipped: {r.get('reason')} |")
        else:
            lines.append(
                f"| {r['model']} | — | — | — | {r.get('error_msg', r.get('status', ''))[:80]} |"
            )

    lines += ["", "## Recommendation", ""]
    ok_recog = [r for r in recog_results if r.get("status") == "ok"]
    if ok_recog:
        best = max(ok_recog, key=lambda r: r.get("rank1_accuracy", -1.0))
        lines.append(
            f"Best recognition pipeline: **{best['model']}** "
            f"(rank-1={best['rank1_accuracy']}, spoof-reject={best['spoof_reject_rate']})"
        )
    else:
        lines.append("No recognition pipelines evaluated successfully.")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    det_results: list[dict] = []
    recog_results: list[dict] = []

    # -----------------------------------------------------------------------
    # YuNet detectors (fp32 + int8)
    # -----------------------------------------------------------------------
    for yunet_file, sface_file in [
        ("yunet_2023mar.onnx", "sface_2021dec.onnx"),
        ("yunet_2023mar_int8.onnx", "sface_2021dec_int8.onnx"),
    ]:
        yunet_path = WEIGHTS_DIR / yunet_file
        sface_path = WEIGHTS_DIR / sface_file
        model_label = yunet_file.replace(".onnx", "")

        if not yunet_path.exists():
            print(f"  Not found: {yunet_path}")
            det_results.append({"model": model_label, "status": "file_not_found"})
            continue

        print(f"=== {model_label} (detector) ===")
        try:
            detector = _YuNetDetector(yunet_path)
            det_metrics = _benchmark_detector(model_label, detector)
            det_results.append({"model": model_label, **det_metrics})
            print(f"  detection_rate={det_metrics['detection_rate']} "
                  f"latency={det_metrics['latency_ms']} ms")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            det_results.append({"model": model_label, "status": "error", "error_msg": str(exc)})
            continue

        # SFace recognizer paired with YuNet
        sface_label = f"{model_label} + {sface_file.replace('.onnx', '')}"
        if not sface_path.exists():
            recog_results.append({"model": sface_label, "status": "file_not_found"})
            continue

        print(f"  Recognizer: {sface_file}")
        try:
            recognizer = _SFaceRecognizer(sface_path)

            def _sface_embed(img, raw, lm, _rec=recognizer):
                if raw is None:
                    return None
                return _rec.embed(img, raw)

            recog_metrics = _benchmark_recognition(
                model_label, _YuNetDetector(yunet_path), sface_label, _sface_embed
            )
            recog_results.append({"model": sface_label, **recog_metrics})
            print(f"  rank1={recog_metrics['rank1_accuracy']} "
                  f"spoof_reject={recog_metrics['spoof_reject_rate']}")
        except Exception as exc:
            print(f"  Recognizer error: {exc}", file=sys.stderr)
            recog_results.append({"model": sface_label, "status": "error", "error_msg": str(exc)})

    # -----------------------------------------------------------------------
    # YOLOv8-face detector
    # -----------------------------------------------------------------------
    yolov8_path = WEIGHTS_DIR / "yolov8n-face.pt"
    print("=== yolov8n-face (detector) ===")
    if not yolov8_path.exists():
        print(f"  Not found: {yolov8_path}")
        det_results.append({"model": "yolov8n-face", "status": "file_not_found"})
    else:
        try:
            yolo_detector = _Yolov8FaceDetector(yolov8_path)
            det_metrics = _benchmark_detector("yolov8n-face", yolo_detector)
            det_results.append({"model": "yolov8n-face", **det_metrics})
            print(f"  detection_rate={det_metrics['detection_rate']} "
                  f"latency={det_metrics['latency_ms']} ms")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            det_results.append({"model": "yolov8n-face", "status": "error", "error_msg": str(exc)})

    # -----------------------------------------------------------------------
    # InsightFace bundles
    # -----------------------------------------------------------------------
    insightface_bundles = [
        "buffalo_l.zip", "buffalo_m.zip", "buffalo_s.zip", "buffalo_sc.zip", "antelopev2.zip"
    ]
    for bundle_file in insightface_bundles:
        bundle_path = WEIGHTS_DIR / bundle_file
        pack_name = Path(bundle_file).stem
        print(f"=== InsightFace: {pack_name} ===")
        if not bundle_path.exists():
            recog_results.append({"model": f"insightface/{pack_name}", "status": "file_not_found"})
            continue
        try:
            metrics = _benchmark_insightface_bundle(bundle_path)
            recog_results.append({"model": f"insightface/{pack_name}", **metrics})
            if metrics.get("status") == "ok":
                print(f"  rank1={metrics['rank1_accuracy']} "
                      f"spoof_reject={metrics['spoof_reject_rate']}")
            else:
                print(f"  {metrics.get('status')}: {metrics.get('reason', metrics.get('error_msg'))}")
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            recog_results.append({"model": f"insightface/{pack_name}", "status": "error",
                                   "error_msg": str(exc)})

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    json_out = EVAL_DIR / "benchmark_results.json"
    json_out.write_text(
        json.dumps(
            {"detectors": det_results, "recognizers": recog_results},
            indent=2, default=str,
        )
    )
    print(f"\nJSON results: {json_out}")

    report = _write_report(det_results, recog_results)
    md_out = EVAL_DIR / "benchmark_report.md"
    md_out.write_text(report)
    print(f"Markdown report: {md_out}")

    ok_det = sum(1 for r in det_results if r.get("status") == "ok")
    ok_rec = sum(1 for r in recog_results if r.get("status") == "ok")
    print(f"\n{ok_det}/{len(det_results)} detectors ok, {ok_rec}/{len(recog_results)} recognizers ok.")


if __name__ == "__main__":
    main()
