"""Test 17: Face Recognition — registry, gallery, and predictor.

Tests cover:
- Face detector and embedder registry dispatch
- FaceGallery enrollment, matching, persistence, and edge cases
- FacePredictor integration (with mock face detector/embedder)
- Optional ONNX model loading tests (skip if models not present)
"""

import sys
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _runner import passed, failed, errors, run_test, run_all
from core.p06_models.face_base import FaceDetector, FaceEmbedder
from core.p06_models.face_registry import (
    FACE_DETECTOR_REGISTRY,
    FACE_EMBEDDER_REGISTRY,
    _FACE_DETECTOR_VARIANT_MAP,
    _FACE_EMBEDDER_VARIANT_MAP,
    build_face_detector,
    build_face_embedder,
)
from core.p10_inference.face_gallery import FaceGallery
from core.p10_inference.face_predictor import FacePredictor

# Trigger registration by importing the model modules
import core.p06_models.scrfd  # noqa: F401
import core.p06_models.mobilefacenet  # noqa: F401

OUTPUTS = Path(__file__).resolve().parent / "outputs" / "17_face_recognition"
OUTPUTS.mkdir(parents=True, exist_ok=True)

SCRFD_ONNX = ROOT / "pretrained" / "scrfd_500m.onnx"
MOBILEFACENET_ONNX = ROOT / "pretrained" / "mobilefacenet_arcface.onnx"


# ---------------------------------------------------------------------------
# Real ONNX model loaders for FacePredictor tests
# ---------------------------------------------------------------------------

def _build_real_face_detector():
    """Build real SCRFD-500M detector from pretrained ONNX."""
    config = {"face_detector": {"arch": "scrfd-500m", "model_path": str(SCRFD_ONNX)}}
    return build_face_detector(config)


def _build_real_face_embedder():
    """Build real MobileFaceNet embedder from pretrained ONNX."""
    config = {"face_embedder": {"arch": "mobilefacenet", "model_path": str(MOBILEFACENET_ONNX)}}
    return build_face_embedder(config)


# ---------------------------------------------------------------------------
# Helper to create L2-normalized random embeddings
# ---------------------------------------------------------------------------

def _make_embedding(rng: np.random.RandomState, dim: int = 512) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    emb = rng.randn(dim).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ---------------------------------------------------------------------------
# Test 1: Face detector registry
# ---------------------------------------------------------------------------

def test_face_detector_registry():
    """Verify SCRFD is registered with correct variants."""
    assert "scrfd" in FACE_DETECTOR_REGISTRY, (
        f"'scrfd' not in FACE_DETECTOR_REGISTRY. Keys: {list(FACE_DETECTOR_REGISTRY.keys())}"
    )
    assert "scrfd-500m" in _FACE_DETECTOR_VARIANT_MAP, "Missing 'scrfd-500m' variant"
    assert "scrfd-2.5g" in _FACE_DETECTOR_VARIANT_MAP, "Missing 'scrfd-2.5g' variant"
    assert _FACE_DETECTOR_VARIANT_MAP["scrfd-500m"] == "scrfd"
    assert _FACE_DETECTOR_VARIANT_MAP["scrfd-2.5g"] == "scrfd"

    # Unknown arch should raise ValueError
    with pytest.raises(ValueError, match="Unknown face detector"):
        build_face_detector({"face_detector": {"arch": "nonexistent_detector"}})


# ---------------------------------------------------------------------------
# Test 2: Face embedder registry
# ---------------------------------------------------------------------------

def test_face_embedder_registry():
    """Verify MobileFaceNet is registered with correct variants."""
    assert "mobilefacenet" in FACE_EMBEDDER_REGISTRY, (
        f"'mobilefacenet' not in FACE_EMBEDDER_REGISTRY. Keys: {list(FACE_EMBEDDER_REGISTRY.keys())}"
    )
    assert "mobilefacenet-arcface" in _FACE_EMBEDDER_VARIANT_MAP, "Missing 'mobilefacenet-arcface' variant"
    assert _FACE_EMBEDDER_VARIANT_MAP["mobilefacenet-arcface"] == "mobilefacenet"

    # Unknown arch should raise ValueError
    with pytest.raises(ValueError, match="Unknown face embedder"):
        build_face_embedder({"face_embedder": {"arch": "nonexistent_embedder"}})


# ---------------------------------------------------------------------------
# Test 3: Gallery enroll and match
# ---------------------------------------------------------------------------

def test_gallery_enroll_and_match(tmp_path):
    """Enroll 3 identities, verify self-match and unknown rejection."""
    gallery = FaceGallery(
        gallery_path=str(tmp_path / "gallery.npz"),
        similarity_threshold=0.4,
    )
    rng = np.random.RandomState(123)
    embeddings = {}

    for name in ["alice", "bob", "charlie"]:
        emb = _make_embedding(rng)
        embeddings[name] = emb
        gallery.enroll(name, emb)

    assert gallery.size == 3

    # Each embedding should match itself with high similarity
    for name, emb in embeddings.items():
        identity, score = gallery.match(emb)
        assert identity == name, f"Expected '{name}', got '{identity}'"
        assert score > 0.99, f"Self-match score too low: {score}"

    # A random unrelated embedding should return "unknown" or low similarity
    random_emb = _make_embedding(np.random.RandomState(999))
    identity, score = gallery.match(random_emb)
    # With random 512-d vectors, cosine similarity is near 0
    assert score < 0.4, f"Random embedding matched with score {score}"


# ---------------------------------------------------------------------------
# Test 4: Gallery save/load roundtrip
# ---------------------------------------------------------------------------

def test_gallery_save_load_roundtrip(tmp_path):
    """Save gallery, create new instance from same path, verify match works."""
    gallery_path = str(tmp_path / "gallery.npz")
    rng = np.random.RandomState(42)

    # Create and populate gallery
    gallery1 = FaceGallery(gallery_path=gallery_path, similarity_threshold=0.4)
    emb_alice = _make_embedding(rng)
    emb_bob = _make_embedding(rng)
    gallery1.enroll("alice", emb_alice)
    gallery1.enroll("bob", emb_bob)
    gallery1.save()

    # Load into new instance
    gallery2 = FaceGallery(gallery_path=gallery_path, similarity_threshold=0.4)
    assert gallery2.size == 2, f"Expected size 2 after load, got {gallery2.size}"

    # Verify matching still works
    identity, score = gallery2.match(emb_alice)
    assert identity == "alice", f"Expected 'alice', got '{identity}'"
    assert score > 0.99


# ---------------------------------------------------------------------------
# Test 5: Empty gallery
# ---------------------------------------------------------------------------

def test_gallery_empty(tmp_path):
    """Empty gallery returns unknown for all queries."""
    gallery = FaceGallery(
        gallery_path=str(tmp_path / "empty.npz"),
        similarity_threshold=0.4,
    )
    assert gallery.size == 0

    # Single match
    rng = np.random.RandomState(0)
    identity, score = gallery.match(_make_embedding(rng))
    assert identity == "unknown"
    assert score == 0.0

    # Batch match with empty input
    results = gallery.match_batch(np.empty((0, 512), dtype=np.float32))
    assert results == []

    # Batch match with non-empty input on empty gallery
    queries = np.stack([_make_embedding(rng) for _ in range(3)])
    results = gallery.match_batch(queries)
    assert len(results) == 3
    for ident, sim in results:
        assert ident == "unknown"
        assert sim == 0.0


# ---------------------------------------------------------------------------
# Test 6: Gallery remove
# ---------------------------------------------------------------------------

def test_gallery_remove(tmp_path):
    """Enroll multiple embeddings, remove one identity, verify removal."""
    gallery = FaceGallery(
        gallery_path=str(tmp_path / "gallery.npz"),
        similarity_threshold=0.4,
    )
    rng = np.random.RandomState(77)

    emb_alice_1 = _make_embedding(rng)
    emb_alice_2 = _make_embedding(rng)
    emb_bob = _make_embedding(rng)

    gallery.enroll("alice", emb_alice_1)
    gallery.enroll("alice", emb_alice_2)
    gallery.enroll("bob", emb_bob)
    assert gallery.size == 3

    # Remove alice (2 embeddings)
    removed = gallery.remove("alice")
    assert removed == 2, f"Expected 2 removed, got {removed}"
    assert gallery.size == 1

    # Alice should no longer match
    identity, _ = gallery.match(emb_alice_1)
    assert identity != "alice" or gallery.size == 0

    # Bob should still match
    identity, score = gallery.match(emb_bob)
    assert identity == "bob"
    assert score > 0.99

    # Removing nonexistent identity returns 0
    removed = gallery.remove("nonexistent")
    assert removed == 0


# ---------------------------------------------------------------------------
# Test 7: Gallery batch match
# ---------------------------------------------------------------------------

def test_gallery_batch_match(tmp_path):
    """Enroll 5 identities, batch match 3 queries against them."""
    gallery = FaceGallery(
        gallery_path=str(tmp_path / "gallery.npz"),
        similarity_threshold=0.4,
    )
    rng = np.random.RandomState(55)
    names = ["person_0", "person_1", "person_2", "person_3", "person_4"]
    embeddings = {}

    for name in names:
        emb = _make_embedding(rng)
        embeddings[name] = emb
        gallery.enroll(name, emb)

    assert gallery.size == 5

    # Query with 3 known embeddings
    query_names = ["person_1", "person_3", "person_4"]
    queries = np.stack([embeddings[n] for n in query_names])
    results = gallery.match_batch(queries)

    assert len(results) == 3
    for i, (ident, sim) in enumerate(results):
        assert ident == query_names[i], f"Query {i}: expected '{query_names[i]}', got '{ident}'"
        assert sim > 0.99, f"Query {i}: similarity too low ({sim})"


# ---------------------------------------------------------------------------
# Test 8: FacePredictor with real ONNX models — violation detections
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SCRFD_ONNX.exists() or not MOBILEFACENET_ONNX.exists(),
    reason="Face ONNX models not found in pretrained/",
)
def test_face_predictor_violation(tmp_path):
    """FacePredictor identifies violators using real SCRFD + MobileFaceNet."""
    from fixtures import real_image

    detector = _build_real_face_detector()
    embedder = _build_real_face_embedder()

    gallery = FaceGallery(
        gallery_path=str(tmp_path / "gallery.npz"),
        similarity_threshold=0.3,
    )

    # Use a real image — enroll the face found in it
    image = real_image(idx=0, split="val")
    h, w = image.shape[:2]

    # Detect face in the full image and enroll it
    full_bbox = np.array([0, 0, w, h], dtype=np.float32)
    face_result = detector.detect_faces(image, full_bbox)
    if face_result["face_boxes"].shape[0] == 0:
        pytest.skip("No face detected in test image — cannot test FacePredictor")

    face_emb = embedder.extract_embedding(
        image, face_result["face_boxes"][0], face_result["landmarks"][0],
    )
    gallery.enroll("person_A", face_emb)

    predictor = FacePredictor(
        face_detector=detector,
        face_embedder=embedder,
        gallery=gallery,
        violation_class_ids=[2],  # head_without_helmet
        expand_ratio=1.5,
    )

    # Simulate detection results: one violation (class 2) covering the face area,
    # one non-violation (class 1) in a different region
    face_box = face_result["face_boxes"][0]
    det_results = {
        "boxes": np.array([face_box, [0, 0, 30, 30]], dtype=np.float32),
        "scores": np.array([0.9, 0.85], dtype=np.float32),
        "labels": np.array([2, 1], dtype=int),
    }

    result = predictor.identify(image, det_results)

    assert len(result["identities"]) == 2
    # First detection is a violation covering the enrolled face → should identify
    assert result["identities"][0] is not None, "Should identify enrolled person"
    assert result["identity_scores"][0] > 0.0
    assert result["face_boxes"][0] is not None

    # Second detection is NOT a violation → None identity
    assert result["identities"][1] is None
    assert result["identity_scores"][1] == 0.0
    assert result["face_boxes"][1] is None
    print(f"    Identified: {result['identities'][0]}, score={result['identity_scores'][0]:.3f}")


# ---------------------------------------------------------------------------
# Test 9: FacePredictor — no violations in detections
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SCRFD_ONNX.exists() or not MOBILEFACENET_ONNX.exists(),
    reason="Face ONNX models not found in pretrained/",
)
def test_face_predictor_no_violations(tmp_path):
    """All detections are non-violation class IDs -> all identities None."""
    from fixtures import real_image

    detector = _build_real_face_detector()
    embedder = _build_real_face_embedder()

    gallery = FaceGallery(
        gallery_path=str(tmp_path / "gallery.npz"),
        similarity_threshold=0.3,
    )
    gallery.enroll("someone", _make_embedding(np.random.RandomState(0)))

    predictor = FacePredictor(
        face_detector=detector,
        face_embedder=embedder,
        gallery=gallery,
        violation_class_ids=[2],
    )

    image = real_image(idx=0, split="val")
    det_results = {
        "boxes": np.array([[50, 50, 150, 150], [200, 200, 350, 350]], dtype=np.float32),
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "labels": np.array([0, 1], dtype=int),  # No class 2
    }

    result = predictor.identify(image, det_results)
    assert all(ident is None for ident in result["identities"])
    assert all(s == 0.0 for s in result["identity_scores"])

    # Empty detections
    result_empty = predictor.identify(image, {
        "boxes": np.empty((0, 4)),
        "scores": np.empty((0,)),
        "labels": np.empty((0,), dtype=int),
    })
    assert result_empty["identities"] == []


# ---------------------------------------------------------------------------
# Test 10: FacePredictor._expand_bbox
# ---------------------------------------------------------------------------

def test_face_predictor_expand_bbox():
    """Verify bbox expansion and clipping to image bounds."""
    # Normal expansion: box [100, 100, 200, 200], ratio 1.5, image 640x480
    bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    expanded = FacePredictor._expand_bbox(bbox, 1.5, 640, 480)
    # Center is (150, 150), original w/h = 100, expanded w/h = 150
    assert expanded[0] == pytest.approx(75.0)   # 150 - 75
    assert expanded[1] == pytest.approx(75.0)   # 150 - 75
    assert expanded[2] == pytest.approx(225.0)  # 150 + 75
    assert expanded[3] == pytest.approx(225.0)  # 150 + 75

    # Clipping: box near top-left corner
    bbox_corner = np.array([0, 0, 40, 40], dtype=np.float32)
    expanded_corner = FacePredictor._expand_bbox(bbox_corner, 2.0, 100, 100)
    assert expanded_corner[0] == 0.0, "x1 should clip to 0"
    assert expanded_corner[1] == 0.0, "y1 should clip to 0"

    # Clipping: box near bottom-right corner
    bbox_br = np.array([580, 440, 640, 480], dtype=np.float32)
    expanded_br = FacePredictor._expand_bbox(bbox_br, 2.0, 640, 480)
    assert expanded_br[2] == 640.0, "x2 should clip to img_w"
    assert expanded_br[3] == 480.0, "y2 should clip to img_h"

    # Ratio 1.0 should keep original box
    bbox_same = np.array([50, 50, 150, 150], dtype=np.float32)
    expanded_same = FacePredictor._expand_bbox(bbox_same, 1.0, 640, 480)
    np.testing.assert_allclose(expanded_same, bbox_same, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 11 (optional): SCRFD ONNX model load
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SCRFD_ONNX.exists(), reason=f"SCRFD ONNX not found at {SCRFD_ONNX}")
def test_scrfd_model_load():
    """Load SCRFD-500M and run face detection on a real image."""
    from fixtures import real_image

    config = {
        "face_detector": {
            "arch": "scrfd-500m",
            "model_path": str(SCRFD_ONNX),
        }
    }
    detector = build_face_detector(config)
    assert isinstance(detector, FaceDetector)

    image = real_image(idx=0, split="val")
    h, w = image.shape[:2]
    bbox = np.array([0, 0, w, h], dtype=np.float32)

    result = detector.detect_faces(image, bbox)
    assert "face_boxes" in result
    assert "face_scores" in result
    assert "landmarks" in result
    assert result["face_boxes"].ndim == 2
    assert result["face_boxes"].shape[1] == 4
    assert result["landmarks"].ndim == 3
    assert result["landmarks"].shape[1:] == (5, 2)


# ---------------------------------------------------------------------------
# Test 12 (optional): MobileFaceNet ONNX model load
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MOBILEFACENET_ONNX.exists(), reason=f"MobileFaceNet ONNX not found at {MOBILEFACENET_ONNX}"
)
def test_mobilefacenet_model_load():
    """Load MobileFaceNet and extract embedding from a real image."""
    from fixtures import real_image

    config = {
        "face_embedder": {
            "arch": "mobilefacenet",
            "model_path": str(MOBILEFACENET_ONNX),
        }
    }
    embedder = build_face_embedder(config)
    assert isinstance(embedder, FaceEmbedder)

    image = real_image(idx=0, split="val")
    h, w = image.shape[:2]
    # Use a centered region as a fake face box
    face_box = np.array([w * 0.3, h * 0.2, w * 0.7, h * 0.6], dtype=np.float32)

    embedding = embedder.extract_embedding(image, face_box)
    assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"
    # Verify L2 normalized
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-4, f"Embedding not L2-normalized: norm={norm}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("\n=== Test 10: Face Recognition ===\n")

    run_test("face_detector_registry", test_face_detector_registry)
    run_test("face_embedder_registry", test_face_embedder_registry)

    # Tests that need tmp_path — create manually for standalone mode
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_test("gallery_enroll_and_match", lambda: test_gallery_enroll_and_match(tmp / "t3"))
        run_test("gallery_save_load_roundtrip", lambda: test_gallery_save_load_roundtrip(tmp / "t4"))
        run_test("gallery_empty", lambda: test_gallery_empty(tmp / "t5"))
        run_test("gallery_remove", lambda: test_gallery_remove(tmp / "t6"))
        run_test("gallery_batch_match", lambda: test_gallery_batch_match(tmp / "t7"))
        if SCRFD_ONNX.exists() and MOBILEFACENET_ONNX.exists():
            run_test("face_predictor_violation", lambda: test_face_predictor_violation(tmp / "t8"))
            run_test("face_predictor_no_violations", lambda: test_face_predictor_no_violations(tmp / "t9"))
        else:
            print("  SKIP: face_predictor tests (ONNX models not found)")


    run_test("face_predictor_expand_bbox", test_face_predictor_expand_bbox)

    # Optional ONNX tests
    if SCRFD_ONNX.exists():
        run_test("scrfd_model_load", test_scrfd_model_load)
    else:
        print(f"  SKIP: scrfd_model_load (ONNX not found at {SCRFD_ONNX})")

    if MOBILEFACENET_ONNX.exists():
        run_test("mobilefacenet_model_load", test_mobilefacenet_model_load)
    else:
        print(f"  SKIP: mobilefacenet_model_load (ONNX not found at {MOBILEFACENET_ONNX})")

    print(f"\nResults: {passed} passed, {failed} failed")
    if errors:
        print("Failures:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    sys.exit(1 if failed else 0)
