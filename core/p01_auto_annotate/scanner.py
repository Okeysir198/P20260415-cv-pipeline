"""Image scanner for auto-annotation pipeline.

Discovers images that need annotation, supporting both YOLO split layouts
and flat directory structures. Filter modes control which images are selected.
"""

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.config import resolve_path
from utils.yolo_io import IMAGE_EXTENSIONS


class ImageScanner:
    """Discover images needing annotation.

    Supports two input modes:
    - **YOLO split layout**: walks ``{split}/images/`` dirs from data config
    - **Flat directory**: scans any directory for image files

    Args:
        data_config: Data config dict (from ``configs/<usecase>/05_data.yaml``). None for flat dir mode.
        config_dir: Directory of the data config file for resolving relative paths.
        image_dir: Path to flat image directory. Mutually exclusive with data_config.
        filter_mode: ``"missing"`` (no label or empty) or ``"all"`` (everything).
        splits: List of splits to scan (for YOLO layout mode).
    """

    def __init__(
        self,
        data_config: dict | None = None,
        config_dir: Path | None = None,
        image_dir: str | None = None,
        filter_mode: str = "missing",
        splits: list[str] | None = None,
    ) -> None:
        self.data_config = data_config
        self.config_dir = Path(config_dir) if config_dir else Path(".")
        self.image_dir = image_dir
        self.filter_mode = filter_mode
        self.splits = splits or ["train", "val", "test"]

        if data_config is None and image_dir is None:
            raise ValueError("Either data_config or image_dir must be provided")

    def scan(self) -> dict[str, list[Path]]:
        """Scan for images based on the configured input mode.

        Returns:
            Dict mapping split name to list of image paths.
            For flat directories, uses ``"default"`` as the split key.
        """
        if self.image_dir is not None:
            return self._scan_flat_dir()
        return self._scan_yolo_layout()

    def _scan_yolo_layout(self) -> dict[str, list[Path]]:
        """Scan YOLO split layout: {split}/images/ directories."""
        assert self.data_config is not None, "data_config required for YOLO layout"
        results: dict[str, list[Path]] = {}

        for split in self.splits:
            if split not in self.data_config:
                logger.debug("Split '%s' not in data config, skipping", split)
                continue

            base_path = resolve_path(self.data_config["path"], self.config_dir)
            images_dir = base_path / self.data_config[split]

            if not images_dir.exists():
                logger.warning("Images directory not found: %s", images_dir)
                results[split] = []
                continue

            all_images = self._list_images(images_dir)
            filtered = self._apply_filter(all_images, yolo_layout=True)

            results[split] = filtered
            logger.info(
                "Split '%s': %d images found, %d after filtering (%s)",
                split,
                len(all_images),
                len(filtered),
                self.filter_mode,
            )

        return results

    def _scan_flat_dir(self) -> dict[str, list[Path]]:
        """Scan a flat directory for image files."""
        assert self.image_dir is not None, "image_dir required for flat dir mode"
        image_dir = Path(self.image_dir)
        if not image_dir.exists():
            logger.warning("Image directory not found: %s", image_dir)
            return {"default": []}

        all_images = self._list_images(image_dir)
        filtered = self._apply_filter(all_images, yolo_layout=False)

        logger.info(
            "Flat dir: %d images found, %d after filtering (%s)",
            len(all_images),
            len(filtered),
            self.filter_mode,
        )

        return {"default": filtered}

    def _list_images(self, directory: Path) -> list[Path]:
        """List all image files in a directory (non-recursive).

        Args:
            directory: Directory to scan.

        Returns:
            Sorted list of image file paths.
        """
        images = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(images)

    def _apply_filter(self, image_paths: list[Path], yolo_layout: bool) -> list[Path]:
        """Apply filter mode to image paths.

        Args:
            image_paths: List of image file paths.
            yolo_layout: If True, labels are in sibling ``labels/`` dir.
                If False, labels are beside images with ``.txt`` extension.

        Returns:
            Filtered list of image paths.
        """
        if self.filter_mode == "all":
            return image_paths

        # "missing" mode: keep only images with no label or empty label
        filtered: list[Path] = []
        for img_path in image_paths:
            label_path = self._get_label_path(img_path, yolo_layout)
            if not label_path.exists():
                filtered.append(img_path)
            elif label_path.read_text().strip() == "":
                filtered.append(img_path)

        return filtered

    @staticmethod
    def _get_label_path(image_path: Path, yolo_layout: bool) -> Path:
        """Get the corresponding label path for an image.

        Args:
            image_path: Path to the image file.
            yolo_layout: If True, use YOLO sibling ``labels/`` convention.
                If False, label is beside the image with ``.txt`` extension.

        Returns:
            Path to the label file.
        """
        if yolo_layout:
            return image_path.parent.parent / "labels" / (image_path.stem + ".txt")
        return image_path.with_suffix(".txt")
