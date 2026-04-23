"""Auto-annotation report generation.

Produces machine-readable JSON reports, human-readable text summaries,
and preview visualizations from auto-annotation results.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from utils.langgraph_common import make_serialisable

logger = logging.getLogger(__name__)


class AutoAnnotateReporter:
    """Generate auto-annotation reports.

    Args:
        output_dir: Base output directory (e.g., ``"runs/auto_annotate"``).
        dataset_name: Dataset name used as subdirectory.
        config: Reporting config dict.
    """

    def __init__(self, output_dir: str, dataset_name: str, config: dict) -> None:
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.config = config
        self.report_dir = self.output_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.preview_count: int = int(config.get("preview_count", 20))
        self.save_previews: bool = bool(config.get("save_previews", True))

    def generate_report(self, image_results: list[dict], summary: dict) -> str:
        """Generate all report artefacts.

        Creates:
            1. ``report.json`` — full machine-readable results
            2. ``summary.txt`` — human-readable text summary
            3. ``preview/`` — visualizations of annotated images

        Args:
            image_results: Per-image annotation result dicts.
            summary: Aggregated summary dict.

        Returns:
            Absolute path to the report directory.
        """
        logger.info("Generating auto-annotation report in %s", self.report_dir)

        self._save_report_json(image_results, summary)
        self._save_summary_txt(summary)

        if self.save_previews:
            self._visualize_previews(image_results, self.preview_count)

        logger.info("Auto-annotation report complete: %s", self.report_dir)
        return str(self.report_dir)

    def _save_report_json(self, image_results: list[dict], summary: dict) -> None:
        """Save full results as JSON."""
        path = self.report_dir / "report.json"
        payload = {
            "metadata": {
                "dataset": self.dataset_name,
                "generated_at": datetime.now(UTC).isoformat(),
                "total_images": len(image_results),
            },
            "summary": make_serialisable(summary),
            "image_results": [make_serialisable(r) for r in image_results],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Saved report.json (%d images)", len(image_results))

    def _save_summary_txt(self, summary: dict) -> None:
        """Save a human-readable text summary."""
        sep = "=" * 70
        lines: list[str] = [
            sep,
            f"  Auto-Annotation Report: {self.dataset_name}",
            sep,
            f"  Total images scanned : {summary.get('total_images', 0)}",
            f"  Total annotated      : {summary.get('total_annotated', 0)}",
            f"  Total detections     : {summary.get('total_detections', 0)}",
            f"  Avg detections/image : {summary.get('avg_detections_per_image', 0):.2f}",
            f"  Output format        : {summary.get('output_format', 'bbox')}",
            f"  Mode                 : {summary.get('mode', 'text')}",
            f"  Dry run              : {summary.get('dry_run', False)}",
            "",
        ]

        # Per-class counts
        per_class = summary.get("per_class_counts", {})
        if per_class:
            lines.append("  Per-Class Counts:")
            for cls_name, count in sorted(per_class.items(), key=lambda x: -x[1]):
                lines.append(f"    {cls_name:<20}: {count}")
            lines.append("")

        # Per-split counts
        per_split = summary.get("per_split_counts", {})
        if per_split:
            lines.append("  Per-Split Counts:")
            for split_name, count in sorted(per_split.items()):
                lines.append(f"    {split_name:<20}: {count}")
            lines.append("")

        # Timing
        timing = summary.get("timing", {})
        if timing:
            lines.append("  Timing (per sample):")
            lines.append(f"    Avg annotation : {timing.get('avg_annotate_s', 0.0):.4f}s")
            lines.append(f"    Avg total      : {timing.get('avg_total_per_sample_s', 0.0):.4f}s")
            lines.append(f"    Min total      : {timing.get('min_total_per_sample_s', 0.0):.4f}s")
            lines.append(f"    Max total      : {timing.get('max_total_per_sample_s', 0.0):.4f}s")

        lines.append(sep)

        path = self.report_dir / "summary.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Saved summary.txt")

    def _visualize_previews(self, image_results: list[dict], count: int) -> None:
        """Save visualizations of annotated images.

        Draws generated bboxes on sample images for visual review.
        """
        if cv2 is None:
            logger.warning("cv2 not available — skipping preview visualizations")
            return

        vis_dir = self.report_dir / "preview"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Pick images with the most detections for interesting previews
        sorted_results = sorted(
            image_results,
            key=lambda r: len(r.get("detections", [])),
            reverse=True,
        )

        import supervision as sv

        from utils.viz import annotate_detections, annotate_polygons, classification_banner

        # Color palette for classes (RGB)
        colors = [
            (0, 200, 0), (220, 0, 0), (200, 200, 0), (200, 0, 200),
            (0, 200, 200), (128, 0, 255), (255, 128, 0), (0, 128, 255),
        ]

        for rank, result in enumerate(sorted_results[:count], 1):
            img_path = result.get("image_path", "")
            if not img_path or not Path(img_path).is_file():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            detections = result.get("detections", [])

            # Group detections by class_id so we emit one annotate_detections
            # call per color (helpers accept a single color per call).
            by_cls: dict[int, list[dict]] = {}
            for det in detections:
                by_cls.setdefault(int(det.get("class_id", 0)), []).append(det)

            for cls_id, dets in by_cls.items():
                color_rgb = colors[cls_id % len(colors)]
                sv_color = sv.Color(r=color_rgb[0], g=color_rgb[1], b=color_rgb[2])

                xyxy = np.array([
                    [
                        (d["cx"] - d["w"] / 2) * w,
                        (d["cy"] - d["h"] / 2) * h,
                        (d["cx"] + d["w"] / 2) * w,
                        (d["cy"] + d["h"] / 2) * h,
                    ]
                    for d in dets
                ], dtype=np.float32)
                scores_arr = np.array(
                    [float(d.get("score", 0.0)) for d in dets], dtype=np.float32
                )
                cls_arr = np.full(len(dets), cls_id, dtype=int)
                sv_dets = sv.Detections(xyxy=xyxy, confidence=scores_arr, class_id=cls_arr)
                labels = [f"cls{cls_id} {s:.2f}" for s in scores_arr]
                img_rgb = annotate_detections(img_rgb, sv_dets, labels=labels, color=sv_color)

                # Draw polygons (closed outlines) per class
                polys: list[np.ndarray] = []
                for d in dets:
                    polygon = d.get("polygon")
                    if polygon and len(polygon) >= 6:
                        pts = np.array(
                            [[int(polygon[i] * w), int(polygon[i + 1] * h)]
                             for i in range(0, len(polygon), 2)],
                            dtype=np.int32,
                        )
                        polys.append(pts)
                if polys:
                    img_rgb = annotate_polygons(img_rgb, polys, color=sv_color)

            # Header banner (red text on top bar — was red BGR (0,0,255) ≡ red RGB)
            img_rgb = classification_banner(
                img_rgb,
                f"detections={len(detections)}",
                position="overlay_top",
                text_color_rgb=(255, 0, 0),
            )

            stem = Path(img_path).stem
            out_path = vis_dir / f"{rank:03d}_{stem}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        logger.info(
            "Saved %d preview visualizations",
            min(count, len(sorted_results)),
        )


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
