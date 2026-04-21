"""
File operation utilities.

Handles copying, renaming, and duplicate detection for data preparation.
"""

import hashlib
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_data_root(source_config: dict, base_dir: Path) -> Path:
    """Resolve dataset root from source config, supporting absolute and relative paths."""
    path = source_config.get("resolved_path") or source_config["path"]
    if Path(path).is_absolute():
        return Path(path)
    return (base_dir / path).resolve()


class FileOps:
    """
    File operations for data preparation.

    Handles copying images, generating unique names, and tracking duplicates.
    """

    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS

    def __init__(self, handle_duplicates: str = "rename"):
        """
        Initialize file operations.

        Args:
            handle_duplicates: How to handle duplicate filenames
                - "skip": Skip duplicate files
                - "rename": Rename with source prefix (default)
                - "overwrite": Overwrite existing files
        """
        if handle_duplicates not in ("skip", "rename", "overwrite"):
            raise ValueError(
                f"handle_duplicates must be 'skip', 'rename', or 'overwrite', "
                f"got '{handle_duplicates}'"
            )
        self.handle_duplicates = handle_duplicates
        self.copied_files: dict[str, Path] = {}  # original_name -> output_path

    def is_image_file(self, path: Path) -> bool:
        """
        Check if a file is an image based on extension.

        Args:
            path: File path

        Returns:
            True if file has image extension
        """
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def get_image_files(self, directory: Path) -> list[Path]:
        """
        Get all image files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of image file paths
        """
        if not directory.is_dir():
            return []

        return [
            f for f in directory.iterdir()
            if f.is_file() and self.is_image_file(f)
        ]

    def generate_unique_name(
        self,
        filename: str,
        source_name: str,
        output_dir: Path
    ) -> str | None:
        """
        Generate unique filename to avoid conflicts.

        Args:
            filename: Original filename
            source_name: Source dataset name (used as prefix)
            output_dir: Output directory (to check for conflicts)

        Returns:
            Unique filename
        """
        # Try original name first
        if not (output_dir / filename).exists():
            return filename

        # Handle duplicate based on strategy
        if self.handle_duplicates == "skip":
            return None  # Signal to skip

        # Add source prefix or number suffix
        stem, ext = Path(filename).stem, Path(filename).suffix

        if self.handle_duplicates == "rename":
            # Try with source prefix
            prefixed = f"{source_name}_{filename}"
            if not (output_dir / prefixed).exists():
                return prefixed

            # Try with number suffix
            counter = 1
            while True:
                numbered = f"{stem}_{counter}{ext}"
                if not (output_dir / numbered).exists():
                    return numbered
                counter += 1

        # If overwrite, return original
        return filename

    def copy_file(
        self,
        src: Path,
        dst_dir: Path,
        dst_name: str | None = None,
        source_name: str = "source"
    ) -> Path | None:
        """
        Copy a file to destination directory.

        Args:
            src: Source file path
            dst_dir: Destination directory
            dst_name: Optional destination filename (auto-generated if None)
            source_name: Source name for duplicate handling

        Returns:
            Path to copied file, or None if skipped
        """
        if not src.exists():
            return None

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Generate destination filename
        if dst_name is None:
            dst_name = src.name

        unique_name = self.generate_unique_name(dst_name, source_name, dst_dir)

        if unique_name is None:  # Skip strategy
            return None

        dst_path = dst_dir / unique_name

        # Copy file
        shutil.copy2(src, dst_path)

        # Track
        self.copied_files[str(src)] = dst_path

        return dst_path

    def copy_images(
        self,
        image_files: list[Path],
        output_dir: Path,
        source_name: str = "source",
        progress_callback=None
    ) -> dict[str, Path]:
        """
        Copy multiple images to output directory.

        Args:
            image_files: List of image file paths
            output_dir: Output directory
            source_name: Source name for duplicate handling
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping original path -> output path
        """
        results = {}

        for i, img_path in enumerate(image_files):
            output_path = self.copy_file(img_path, output_dir, source_name=source_name)
            if output_path:
                results[str(img_path)] = output_path

            if progress_callback:
                progress_callback(i + 1, len(image_files))

        return results

    def file_hash(self, path: Path) -> str:
        """
        Calculate MD5 hash of a file.

        Args:
            path: File path

        Returns:
            Hexadecimal hash string
        """
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

