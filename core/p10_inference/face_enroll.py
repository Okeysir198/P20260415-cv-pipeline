"""Face enrollment CLI — enroll face embeddings into a gallery.

Supports bulk enrollment from a directory structure (one subdirectory
per identity) or single-person enrollment from a single image.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from core.p06_models.face_registry import build_face_detector, build_face_embedder
from core.p10_inference.face_gallery import FaceGallery
from utils.config import load_config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def enroll_single(
    identity: str,
    image_path: Path,
    face_detector,
    face_embedder,
    gallery: FaceGallery,
) -> bool:
    """Enroll a single face image.

    Detects face in full image (using full-image bbox), extracts embedding,
    and enrolls in gallery.

    Returns True if enrollment succeeded.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Failed to read image: %s", image_path)
        return False

    h, w = image.shape[:2]
    full_bbox = np.array([0, 0, w, h], dtype=np.float32)

    face_result = face_detector.detect_faces(image, full_bbox)
    if len(face_result["face_boxes"]) == 0:
        logger.warning("No face detected in %s", image_path)
        return False

    # Take best face
    best_idx = int(np.argmax(face_result["face_scores"]))
    best_box = face_result["face_boxes"][best_idx]
    best_lm = face_result["landmarks"][best_idx]
    lm = best_lm if np.any(best_lm) else None

    embedding = face_embedder.extract_embedding(image, best_box, lm)
    gallery.enroll(identity, embedding)
    return True


def enroll_directory(
    image_dir: Path,
    face_detector,
    face_embedder,
    gallery: FaceGallery,
) -> dict:
    """Enroll all faces from a directory structure.

    Expects: image_dir/<identity>/<photo>.{jpg,png,...}

    Returns stats dict with enrolled/failed/skipped counts.
    """
    stats = {"enrolled": 0, "failed": 0, "identities": 0}

    for identity_dir in sorted(image_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        identity = identity_dir.name
        stats["identities"] += 1

        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if enroll_single(identity, img_path, face_detector, face_embedder, gallery):
                stats["enrolled"] += 1
                logger.info("Enrolled %s from %s", identity, img_path.name)
            else:
                stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Enroll faces into a gallery")
    parser.add_argument(
        "--config",
        type=str,
        default="features/access-face_recognition/configs/face.yaml",
        help="Face config YAML path",
    )

    # Bulk enrollment
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory with subdirs per identity",
    )

    # Single enrollment
    parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Person identity name (for single enrollment)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image path (for single enrollment)",
    )

    # Override gallery path
    parser.add_argument(
        "--gallery",
        type=str,
        default=None,
        help="Gallery .npz path (overrides config)",
    )

    args = parser.parse_args()

    # Validate args
    if args.image_dir is None and (args.identity is None or args.image is None):
        parser.error(
            "Provide --image-dir for bulk enrollment, or --identity + --image for single"
        )

    # Load config and build models
    config = load_config(args.config)
    face_detector = build_face_detector(config)
    face_embedder = build_face_embedder(config)

    # Gallery path
    gallery_path = args.gallery or config.get("gallery", {}).get(
        "path", "data/face_gallery/default.npz"
    )
    similarity_threshold = config.get("gallery", {}).get("similarity_threshold", 0.4)
    gallery = FaceGallery(gallery_path, similarity_threshold=similarity_threshold)

    if args.image_dir:
        # Bulk enrollment
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            logger.error("Not a directory: %s", image_dir)
            sys.exit(1)

        stats = enroll_directory(image_dir, face_detector, face_embedder, gallery)
        gallery.save()
        logger.info(
            "Enrollment complete: %d embeddings from %d identities (%d failed)",
            stats["enrolled"],
            stats["identities"],
            stats["failed"],
        )
    else:
        # Single enrollment
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            sys.exit(1)

        if enroll_single(
            args.identity, image_path, face_detector, face_embedder, gallery
        ):
            gallery.save()
            logger.info("Enrolled '%s' successfully", args.identity)
        else:
            logger.error("Failed to enroll '%s'", args.identity)
            sys.exit(1)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()
