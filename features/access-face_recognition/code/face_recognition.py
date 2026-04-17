"""Production face recognition pipeline.

Supports two backends (selected via face.yaml ``pipeline`` key):
  * insightface  — buffalo_l SCRFD + ArcFace (non-commercial)
  * opencv_dnn   — YuNet INT8 + SFace INT8 (Apache-2.0)

InsightFace is attempted first; if the package is not installed the pipeline
falls back to opencv_dnn automatically (regardless of the config setting).

CLI:
  uv run features/access-face_recognition/code/face_recognition.py \\
    --enroll --images features/access-face_recognition/samples/ \\
    --config features/access-face_recognition/configs/face.yaml

  uv run features/access-face_recognition/code/face_recognition.py \\
    --smoke-test \\
    --config features/access-face_recognition/configs/face.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from utils.config import load_config, resolve_path  # noqa: E402

# ---------------------------------------------------------------------------
# ArcFace canonical landmark positions (112×112 aligned crop)
# ---------------------------------------------------------------------------
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FaceDetection:
    box_xyxy: np.ndarray  # (4,) float32
    landmarks: np.ndarray  # (5, 2) float32
    score: float


@dataclass
class FaceResult:
    detection: FaceDetection
    identity: str  # enrolled name or "unknown"
    similarity: float  # cosine similarity; -1.0 if unknown
    embedding: np.ndarray  # (512,) for possible re-use


# ---------------------------------------------------------------------------
# Internal: OpenCV DNN backend (Apache-2.0)
# ---------------------------------------------------------------------------


class _YuNetDetector:
    """YuNet INT8 face detector via OpenCV."""

    def __init__(self, model_path: Path) -> None:
        self._det = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=50
        )
        self.last_raw: np.ndarray | None = None  # needed by SFace alignCrop

    def detect_all(self, img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """Return [(box_xyxy, landmarks_5x2, score, raw_row), ...]."""
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img)
        if faces is None or len(faces) == 0:
            return []
        results = []
        for face in faces:
            x, y, fw, fh = face[:4]
            score = float(face[14])
            box = np.array([x, y, x + fw, y + fh], dtype=np.float32)
            lm_raw = face[4:14].reshape(5, 2)
            # YuNet landmark order: right_eye, left_eye, nose, right_mouth, left_mouth
            # Reorder to ArcFace convention: left_eye, right_eye, nose, left_mouth, right_mouth
            lm = np.stack(
                [lm_raw[1], lm_raw[0], lm_raw[2], lm_raw[4], lm_raw[3]]
            ).astype(np.float32)
            results.append((box, lm, score, face))
        return results


class _SFaceEmbedder:
    """SFace INT8 face recognizer via OpenCV."""

    def __init__(self, model_path: Path) -> None:
        self._rec = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, img_bgr: np.ndarray, yunet_raw: np.ndarray) -> np.ndarray:
        aligned = self._rec.alignCrop(img_bgr, yunet_raw)
        feat = self._rec.feature(aligned).reshape(-1)
        return _l2_normalize(feat)


# ---------------------------------------------------------------------------
# Internal: InsightFace backend (non-commercial)
# ---------------------------------------------------------------------------


class _InsightFaceBackend:
    """Thin wrapper around insightface.app.FaceAnalysis."""

    def __init__(self, model_pack: Path, det_size: tuple[int, int], det_thresh: float) -> None:
        import insightface  # noqa: PLC0415

        # InsightFace loads from a directory named after the pack.
        # We pass the zip path; insightface will unpack if needed.
        pack_name = model_pack.stem  # e.g. "buffalo_l"
        pack_root = model_pack.parent

        self._app = insightface.app.FaceAnalysis(
            name=pack_name,
            root=str(pack_root),
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)

    def detect_and_embed(
        self, img_bgr: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """Return [(box_xyxy, landmarks_5x2, score, embedding), ...]."""
        faces = self._app.get(img_bgr)
        results = []
        for f in faces:
            box = f.bbox.astype(np.float32)  # (4,) xyxy
            lm = f.kps.astype(np.float32)  # (5, 2)
            score = float(f.det_score)
            emb = _l2_normalize(f.embedding)
            results.append((box, lm, score, emb))
        return results


# ---------------------------------------------------------------------------
# Internal: SCRFD ONNX detector (used by OpenCV-DNN pipeline for alignment)
# ---------------------------------------------------------------------------


def _align_face(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp face to ArcFace canonical 112×112 alignment."""
    dst = _ARCFACE_DST * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(landmarks[:3], dst[:3])
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


# ---------------------------------------------------------------------------
# Internal: OpenCV DNN MobileFaceNet for aligned crops (fallback embed path)
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are already L2-normalized


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class FaceRecognitionPipeline:
    """Production face recognition pipeline.

    Loads config from ``face.yaml``, attempts InsightFace backend, falls back
    to OpenCV DNN (YuNet INT8 + SFace INT8) when InsightFace is unavailable.
    """

    def __init__(self, config_path: str | Path) -> None:
        config_path = Path(config_path).resolve()
        self._cfg = load_config(config_path)
        self._cfg_dir = config_path.parent

        self._threshold: float = float(self._cfg.get("similarity_threshold", 0.5))
        self._unknown_label: str = self._cfg.get("unknown_label", "unknown")

        gallery_rel = self._cfg.get("gallery_path", "../eval/gallery.npz")
        self._gallery_path = resolve_path(gallery_rel, self._cfg_dir)

        self._backend_name = self._cfg.get("pipeline", "opencv_dnn")
        self._backend: _InsightFaceBackend | None = None
        self._yunet: _YuNetDetector | None = None
        self._sface: _SFaceEmbedder | None = None

        self._load_backend()

        # Gallery: {name: mean_embedding (512,)}
        self._gallery: dict[str, np.ndarray] = {}
        if self._gallery_path.exists():
            self._load_gallery()

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self) -> None:
        if self._backend_name == "insightface":
            try:
                cfg_if = self._cfg.get("insightface", {})
                pack_path = resolve_path(cfg_if["model_pack"], self._cfg_dir)
                det_size = tuple(cfg_if.get("det_size", [640, 640]))
                det_thresh = float(cfg_if.get("det_thresh", 0.5))
                self._backend = _InsightFaceBackend(pack_path, det_size, det_thresh)
                print(f"[face_recognition] Backend: InsightFace buffalo_l ({pack_path.name})")
                return
            except (ImportError, Exception) as exc:
                print(f"[face_recognition] InsightFace unavailable ({exc}); falling back to opencv_dnn")

        # OpenCV DNN path (default or fallback)
        cfg_cv = self._cfg.get("opencv_dnn", {})
        det_path = resolve_path(cfg_cv["detector"], self._cfg_dir)
        rec_path = resolve_path(cfg_cv["recognizer"], self._cfg_dir)
        self._yunet = _YuNetDetector(det_path)
        self._sface = _SFaceEmbedder(rec_path)
        print(f"[face_recognition] Backend: OpenCV DNN (YuNet + SFace)")

    # ------------------------------------------------------------------
    # Gallery persistence
    # ------------------------------------------------------------------

    def _load_gallery(self) -> None:
        data = np.load(self._gallery_path)
        self._gallery = {k: data[k] for k in data.files}
        print(f"[face_recognition] Gallery loaded: {list(self._gallery.keys())} from {self._gallery_path}")

    def _save_gallery(self) -> None:
        self._gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(self._gallery_path), **self._gallery)

    # ------------------------------------------------------------------
    # Internal: detect + embed a single image
    # ------------------------------------------------------------------

    def _detect_and_embed_all(
        self, img: np.ndarray
    ) -> list[tuple[FaceDetection, np.ndarray]]:
        """Return [(FaceDetection, embedding), ...] for all faces in img."""
        if self._backend is not None:
            raw = self._backend.detect_and_embed(img)
            results = []
            for box, lm, score, emb in raw:
                det = FaceDetection(box_xyxy=box, landmarks=lm, score=score)
                results.append((det, emb))
            return results

        # OpenCV DNN path
        raw = self._yunet.detect_all(img)
        results = []
        for box, lm, score, yunet_raw in raw:
            emb = self._sface.embed(img, yunet_raw)
            det = FaceDetection(box_xyxy=box, landmarks=lm, score=score)
            results.append((det, emb))
        return results

    def _best_face_embedding(self, img: np.ndarray) -> np.ndarray | None:
        """Extract embedding for the largest/highest-score face in the image."""
        detections = self._detect_and_embed_all(img)
        if not detections:
            return None
        # Pick the largest face by bounding-box area
        def _area(det: FaceDetection) -> float:
            b = det.box_xyxy
            return max(0.0, float((b[2] - b[0]) * (b[3] - b[1])))

        det, emb = max(detections, key=lambda pair: _area(pair[0]))
        return emb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(
        self,
        images_dir: str | Path,
        name: str,
        *,
        append: bool = True,
    ) -> int:
        """Enroll all images matching ``<name>_*.ext`` in ``images_dir``.

        Args:
            images_dir: Directory containing enrollment images.
            name: Person name (e.g. "alice").
            append: If True, average new embeddings with any existing gallery entry.

        Returns:
            Number of images successfully processed.
        """
        images_dir = Path(images_dir)
        candidates = [
            p for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in _IMAGE_EXTS and p.stem.split("_")[0] == name
        ]
        embeddings: list[np.ndarray] = []
        for img_path in candidates:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[enroll] Cannot read {img_path.name}, skipping")
                continue
            emb = self._best_face_embedding(img)
            if emb is None:
                print(f"[enroll] No face found in {img_path.name}, skipping")
                continue
            embeddings.append(emb)

        if not embeddings:
            print(f"[enroll] No valid embeddings for '{name}'")
            return 0

        new_mean = np.mean(embeddings, axis=0)
        new_mean = _l2_normalize(new_mean)

        if append and name in self._gallery:
            # Weighted average: treat existing as one "sample"
            combined = np.stack([self._gallery[name], new_mean])
            self._gallery[name] = _l2_normalize(np.mean(combined, axis=0))
        else:
            self._gallery[name] = new_mean

        self._save_gallery()
        print(f"[enroll] '{name}': {len(embeddings)} image(s) enrolled -> {self._gallery_path}")
        return len(embeddings)

    def enroll_directory(self, images_dir: str | Path) -> dict[str, int]:
        """Auto-discover person names from filenames and enroll each.

        Convention: ``<name>_<anything>.ext`` → name is the prefix before
        the first underscore. Filenames starting with "spoof" are skipped.

        Returns:
            {name: images_enrolled_count}
        """
        images_dir = Path(images_dir)
        names: set[str] = set()
        for p in images_dir.iterdir():
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            prefix = p.stem.split("_")[0]
            if prefix == "spoof":
                continue
            names.add(prefix)

        results: dict[str, int] = {}
        for name in sorted(names):
            results[name] = self.enroll(images_dir, name, append=False)
        return results

    def recognize(self, image_bgr: np.ndarray) -> list[FaceResult]:
        """Detect all faces in ``image_bgr`` and match against the gallery.

        Returns:
            One FaceResult per detected face, sorted by box area (largest first).
        """
        detections = self._detect_and_embed_all(image_bgr)
        face_results: list[FaceResult] = []

        for det, emb in detections:
            identity, similarity = self._match_gallery(emb)
            face_results.append(
                FaceResult(detection=det, identity=identity, similarity=similarity, embedding=emb)
            )

        # Sort largest face first
        face_results.sort(
            key=lambda r: float(
                (r.detection.box_xyxy[2] - r.detection.box_xyxy[0])
                * (r.detection.box_xyxy[3] - r.detection.box_xyxy[1])
            ),
            reverse=True,
        )
        return face_results

    def _match_gallery(self, emb: np.ndarray) -> tuple[str, float]:
        """Return (name, similarity) for best gallery match, or (unknown, -1.0)."""
        if not self._gallery:
            return self._unknown_label, -1.0

        best_name = self._unknown_label
        best_sim = -1.0
        for name, gallery_emb in self._gallery.items():
            sim = _cosine(emb, gallery_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < self._threshold:
            return self._unknown_label, best_sim
        return best_name, best_sim

    def draw(self, image_bgr: np.ndarray, results: list[FaceResult]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of image_bgr.

        Green box = known identity; red box = unknown.
        """
        out = image_bgr.copy()
        for r in results:
            b = r.detection.box_xyxy.astype(int)
            x1, y1, x2, y2 = b
            known = r.identity != self._unknown_label
            color = (0, 200, 0) if known else (0, 0, 220)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{r.identity} {r.similarity:.2f}" if r.similarity >= 0 else r.identity
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            for px, py in r.detection.landmarks.astype(int):
                cv2.circle(out, (px, py), 2, (0, 0, 255), -1)

        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face recognition pipeline CLI")
    p.add_argument("--config", required=True, help="Path to face.yaml")
    p.add_argument("--enroll", action="store_true", help="Enroll images from --images dir")
    p.add_argument("--images", help="Directory of enrollment images")
    p.add_argument("--smoke-test", action="store_true", help="Run smoke test on sample images")
    return p.parse_args()


def _cmd_enroll(pipeline: FaceRecognitionPipeline, images_dir: str) -> None:
    counts = pipeline.enroll_directory(images_dir)
    print("\nEnrollment summary:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} image(s)")
    total = sum(counts.values())
    print(f"Total: {len(counts)} identities, {total} images enrolled")


def _cmd_smoke_test(pipeline: FaceRecognitionPipeline, config_path: Path) -> None:
    """Enroll _1 images, test on _2 images and spoof images."""
    cfg = load_config(config_path)
    cfg_dir = config_path.parent
    samples_dir = resolve_path("../samples", cfg_dir)
    eval_dir = resolve_path("../eval", cfg_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: enroll only *_1 images ---
    print("\n[smoke-test] Enrolling _1 images...")
    all_imgs = sorted(p for p in samples_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    enroll_imgs = [p for p in all_imgs if p.stem.endswith("_1") and not p.stem.startswith("spoof")]

    for img_path in enroll_imgs:
        name = img_path.stem.rsplit("_", 1)[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        emb = pipeline._best_face_embedding(img)
        if emb is None:
            print(f"  No face in {img_path.name}")
            continue
        emb_normalized = _l2_normalize(emb)
        pipeline._gallery[name] = emb_normalized

    pipeline._save_gallery()
    print(f"  Enrolled: {sorted(pipeline._gallery.keys())}")

    # --- Step 2: test on _2 images and spoof images ---
    test_imgs = [
        p for p in all_imgs
        if (p.stem.endswith("_2") and not p.stem.startswith("spoof"))
        or p.stem.startswith("spoof")
    ]

    results_log: list[dict] = []
    correct = 0
    total = 0

    for img_path in sorted(test_imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Cannot read {img_path.name}")
            continue

        face_results = pipeline.recognize(img)

        is_spoof = img_path.stem.startswith("spoof")
        if is_spoof:
            # Expected: unknown (spoof should be rejected or matched to alice but we flag it)
            # For smoke test purposes: spoof is "alice" print/phone — if matched it's a weakness
            expected_identity = "spoof"
        else:
            expected_identity = img_path.stem.rsplit("_", 1)[0]

        if not face_results:
            identity = "no_face"
            similarity = -1.0
        else:
            top = face_results[0]
            identity = top.identity
            similarity = top.similarity

        # Correctness logic:
        # - _2 images: correct if identity matches expected_identity
        # - spoof images: correct if identity == "unknown" (correctly rejected)
        if is_spoof:
            verdict_correct = identity == pipeline._unknown_label
        else:
            verdict_correct = identity == expected_identity

        if verdict_correct:
            correct += 1
        total += 1

        entry = {
            "image": img_path.name,
            "expected": expected_identity,
            "predicted": identity,
            "similarity": round(similarity, 4),
            "correct": verdict_correct,
            "is_spoof": is_spoof,
        }
        results_log.append(entry)
        status = "OK" if verdict_correct else "FAIL"
        print(f"  [{status}] {img_path.name}: predicted={identity} sim={similarity:.3f}")

    # --- Step 3: write results ---
    out_path = eval_dir / "smoke_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": {"correct": correct, "total": total}, "results": results_log}, f, indent=2)

    print(f"\nSmoke test: {correct}/{total} correct")
    print(f"Results written to {out_path}")


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    pipeline = FaceRecognitionPipeline(config_path)

    if args.enroll:
        if not args.images:
            print("--enroll requires --images <directory>", file=sys.stderr)
            sys.exit(1)
        _cmd_enroll(pipeline, args.images)
    elif args.smoke_test:
        _cmd_smoke_test(pipeline, config_path)
    else:
        print("Specify --enroll or --smoke-test", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
