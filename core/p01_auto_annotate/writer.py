"""Label writer for auto-annotation pipeline.

Writes YOLO-format label files supporting both detection (bbox)
and segmentation (polygon) output formats.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

logger = logging.getLogger(__name__)


class LabelWriter:
    """Write YOLO-format label files.

    Supports three output formats:
    - **bbox**: ``class_id cx cy w h`` (standard YOLO detection)
    - **polygon**: ``class_id x1 y1 x2 y2 ... xN yN`` (YOLO-seg format)
    - **both**: writes bbox as ``.txt`` and polygon as ``_seg.txt``

    Args:
        output_format: Output format — ``"bbox"``, ``"polygon"``, or ``"both"``.
        dry_run: If True, do not write files to disk.
        backup_dir: If set, existing label files are copied here before overwriting.
            Use :meth:`create_backup_dir` to generate a timestamped directory.
    """

    def __init__(
        self,
        output_format: str = "bbox",
        dry_run: bool = False,
        backup_dir: Path | None = None,
    ) -> None:
        self.output_format = output_format
        self.dry_run = dry_run
        self.backup_dir = Path(backup_dir) if backup_dir is not None else None

    def write(
        self,
        image_path: Path,
        detections: list[dict[str, Any]],
    ) -> bool:
        """Write label file(s) for an image.

        Determines the label path based on the image path:
        - YOLO layout (images/labels siblings): ``labels/{stem}.txt``
        - Flat directory: ``{stem}.txt`` beside the image

        Args:
            image_path: Path to the image file.
            detections: List of detection dicts with keys:
                class_id, cx, cy, w, h, and optionally polygon.

        Returns:
            True if labels were written (or would be written in dry_run).
        """
        if not detections:
            logger.debug("No detections for %s, skipping write", image_path.name)
            return False

        image_path = Path(image_path)

        if self.output_format in ("bbox", "both"):
            bbox_path = self._get_label_path(image_path, suffix=".txt")
            bbox_lines = self._format_bbox(detections)
            self._write_file(bbox_path, bbox_lines)

        if self.output_format in ("polygon", "both"):
            if self.output_format == "both":
                poly_path = self._get_label_path(image_path, suffix="_seg.txt")
            else:
                poly_path = self._get_label_path(image_path, suffix=".txt")
            poly_lines = self._format_polygon(detections)
            if poly_lines:
                self._write_file(poly_path, poly_lines)

        return True

    def _format_bbox(self, detections: list[dict[str, Any]]) -> list[str]:
        """Format detections as YOLO bbox lines.

        Args:
            detections: List of detection dicts.

        Returns:
            List of ``"class_id cx cy w h"`` strings.
        """
        lines: list[str] = []
        for det in detections:
            line = (
                f"{det['class_id']} "
                f"{det['cx']:.6f} {det['cy']:.6f} "
                f"{det['w']:.6f} {det['h']:.6f}"
            )
            lines.append(line)
        return lines

    def _format_polygon(self, detections: list[dict[str, Any]]) -> list[str]:
        """Format detections as YOLO-seg polygon lines.

        Falls back to bbox format for detections without polygon data.

        Args:
            detections: List of detection dicts.

        Returns:
            List of ``"class_id x1 y1 x2 y2 ... xN yN"`` strings.
        """
        lines: list[str] = []
        for det in detections:
            polygon = det.get("polygon")
            if polygon and len(polygon) >= 6:
                coords = " ".join(f"{v:.6f}" for v in polygon)
                line = f"{det['class_id']} {coords}"
            else:
                # Fallback to bbox
                line = (
                    f"{det['class_id']} "
                    f"{det['cx']:.6f} {det['cy']:.6f} "
                    f"{det['w']:.6f} {det['h']:.6f}"
                )
            lines.append(line)
        return lines

    def _write_file(self, path: Path, lines: list[str]) -> None:
        """Write lines to a file (respecting dry_run).

        If ``backup_dir`` is set and the target file already exists, the
        existing file is copied into ``backup_dir`` (preserving its path
        relative to the labels directory) before being overwritten.

        Args:
            path: Output file path.
            lines: Lines to write.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would write %d lines to %s", len(lines), path)
            return

        # Back up existing label file before overwriting
        if self.backup_dir is not None and path.exists():
            # Preserve relative structure: find the "labels" component and keep
            # everything from that point onward as the relative path.
            parts = list(path.parts)
            rel_start = None
            for i, part in enumerate(parts):
                if part == "labels":
                    rel_start = i
                    break
            rel_path = Path(*parts[rel_start:]) if rel_start is not None else Path(path.name)

            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            logger.debug("Backed up %s to %s", path, backup_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.debug("Wrote %d annotations to %s", len(lines), path)

    @staticmethod
    def create_backup_dir(labels_dir: Path) -> Path:
        """Create a timestamped backup directory as a sibling of *labels_dir*.

        The directory is created immediately so callers can pass it to
        :class:`LabelWriter` without further setup.

        Args:
            labels_dir: Path to the ``labels/`` directory being overwritten.

        Returns:
            Path to the newly created backup directory, e.g.
            ``dataset_store/fire_detection/train/labels_backup_20260401_160000/``.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = labels_dir.parent / f"labels_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        return backup_dir

    @staticmethod
    def _get_label_path(image_path: Path, suffix: str = ".txt") -> Path:
        """Get the label file path for an image.

        Uses YOLO layout if ``images/`` is in the path, otherwise
        places label beside the image.

        Args:
            image_path: Path to the image file.
            suffix: File suffix (e.g. ``.txt`` or ``_seg.txt``).

        Returns:
            Path to the label file.
        """
        parts = list(image_path.parts)

        # Check for YOLO layout (images/labels siblings)
        for i, part in enumerate(parts):
            if part == "images":
                parts[i] = "labels"
                label_path = Path(*parts)
                if suffix == ".txt":
                    return label_path.with_suffix(".txt")
                else:
                    return label_path.with_suffix("").with_name(
                        label_path.stem + suffix
                    )

        # Flat directory: label beside image
        if suffix == ".txt":
            return image_path.with_suffix(".txt")
        return image_path.with_suffix("").with_name(image_path.stem + suffix)
