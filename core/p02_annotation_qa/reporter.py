"""Annotation QA report generation.

Produces machine-readable JSON reports, human-readable text summaries,
worst-image visualizations, and auto-fix suggestion files from QA results.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

import cv2
import matplotlib

matplotlib.use("Agg")

from utils.langgraph_common import make_serialisable
from utils.metrics import xywh_to_xyxy

logger = logging.getLogger(__name__)


class QAReporter:
    """Generate annotation QA reports."""

    def __init__(self, output_dir: str, dataset_name: str, config: dict) -> None:
        """Initialise the reporter.

        Args:
            output_dir: Run directory (e.g., ``"runs/fire_detection/2026-03-26_1430_05_annotation_quality"``).
            dataset_name: Dataset name for report metadata.
            config: QA config dict (the ``reporting`` section).
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.config = config
        self.report_dir = self.output_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Configurable knobs with sensible defaults.
        self.worst_count: int = int(config.get("worst_count", 10))
        self.save_fixes: bool = bool(config.get("save_fixes", True))
        self.save_visualizations: bool = bool(config.get("save_visualizations", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self, image_results: List[Dict], summary: Dict) -> str:
        """Generate all report artefacts.

        Creates:
            1. ``report.json``  -- full machine-readable results
            2. ``summary.txt``  -- human-readable text summary
            3. ``worst_images/`` -- visualizations of worst N images
            4. ``fixes.json``   -- auto-fixable corrections (when enabled)
            5. ``checkpoint.json`` -- processing state for resume

        Args:
            image_results: Per-image QA result dicts.
            summary: Aggregated summary dict.

        Returns:
            Absolute path to the report directory.
        """
        logger.info("Generating QA report in %s", self.report_dir)

        self._save_report_json(image_results, summary)
        self._save_summary_txt(summary)

        if self.save_visualizations:
            self._visualize_worst_images(image_results, self.worst_count)

        if self.save_fixes:
            self._save_fixes_json(image_results)

        # Persist a lightweight checkpoint so runs can be resumed.
        self.save_checkpoint({
            "completed": True,
            "total_checked": summary.get("total_checked", len(image_results)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info("QA report complete: %s", self.report_dir)
        return str(self.report_dir)

    def save_checkpoint(self, state_snapshot: Dict) -> None:
        """Save a processing checkpoint for resume.

        Args:
            state_snapshot: Arbitrary serialisable state dict.
        """
        path = self.report_dir / "checkpoint.json"
        data = {**state_snapshot, "saved_at": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load a previously saved checkpoint.

        Args:
            checkpoint_path: Path to ``checkpoint.json``.

        Returns:
            The deserialised state dict.
        """
        data: Dict = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_report_json(self, image_results: List[Dict], summary: Dict) -> None:
        """Save full results as JSON."""
        path = self.report_dir / "report.json"
        payload = {
            "metadata": {
                "dataset": self.dataset_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_images": len(image_results),
            },
            "summary": make_serialisable(summary),
            "image_results": [make_serialisable(r) for r in image_results],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Saved report.json (%d images)", len(image_results))

    def _save_summary_txt(self, summary: Dict) -> None:
        """Save a human-readable text summary."""
        total = summary.get("total_checked", 0)
        avg_score = summary.get("avg_quality_score", 0.0)
        grades = summary.get("grades", {})
        issues = summary.get("issue_breakdown", {})
        per_class = summary.get("per_class_stats", {})
        worst = summary.get("worst_images", [])

        sep = "=" * 70
        lines: List[str] = [
            sep,
            f"  Annotation QA Report: {self.dataset_name}",
            sep,
            f"  Total images checked: {total}",
            f"  Average quality score: {avg_score:.3f}",
            "",
            "  Grade Distribution:",
        ]
        for grade in ("good", "review", "bad"):
            count = grades.get(grade, 0)
            pct = (count / total * 100) if total else 0.0
            lines.append(f"    {grade:<8}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("  Issue Breakdown:")
        for issue_type, count in sorted(issues.items(), key=lambda x: -x[1]):
            lines.append(f"    {issue_type:<24}: {count}")

        if per_class:
            lines.append("")
            lines.append("  Per-Class Statistics:")
            for cls_name, stats in sorted(per_class.items()):
                cls_count = stats.get("count", 0)
                cls_issues = stats.get("issues", 0)
                lines.append(f"    {cls_name:<16}: count={cls_count}, issues={cls_issues}")

        # --- Timing statistics ---
        timing = summary.get("timing", {})
        if timing:
            lines.append("")
            lines.append("  Timing (per sample):")
            lines.append(f"    Avg validation : {timing.get('avg_validate_s', 0.0):.4f}s")
            lines.append(f"    Avg SAM3 verify: {timing.get('avg_sam3_verify_s', 0.0):.4f}s")
            lines.append(f"    Avg total      : {timing.get('avg_total_per_sample_s', 0.0):.4f}s")
            lines.append(f"    Min total      : {timing.get('min_total_per_sample_s', 0.0):.4f}s")
            lines.append(f"    Max total      : {timing.get('max_total_per_sample_s', 0.0):.4f}s")

        if worst:
            lines.append("")
            n_show = min(len(worst), 10)
            lines.append(f"  Top {n_show} Worst Images:")
            for rank, entry in enumerate(worst[:n_show], 1):
                img_path = entry.get("image_path", "?")
                score = entry.get("quality_score", 0.0)
                issue_count = entry.get("issue_count", 0)
                lines.append(f"    {rank:>2}. {img_path} — score={score:.3f}, issues={issue_count}")

        lines.append(sep)

        path = self.report_dir / "summary.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Saved summary.txt")

    def _visualize_worst_images(self, image_results: List[Dict], count: int) -> None:
        """Save visualizations of the worst-scoring images.

        For each worst image the method:
            1. Loads the image with cv2.
            2. Draws original YOLO annotations in GREEN.
            3. Draws SAM3-suggested bboxes in RED (when available).
            4. Overlays issue text.
            5. Saves to ``worst_images/{rank}_{stem}.png``.

        Uses ``utils.visualization.draw_bboxes`` for drawing.
        """
        if cv2 is None:
            logger.warning("cv2 not available — skipping worst-image visualizations")
            return

        try:
            from core.p08_evaluation.visualization import draw_bboxes
        except ImportError:
            logger.warning("visualization module not available — skipping worst-image visualizations")
            return

        vis_dir = self.report_dir / "worst_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Sort ascending by quality score (worst first).
        sorted_results = sorted(image_results, key=lambda r: r.get("quality_score", 1.0))

        for rank, result in enumerate(sorted_results[:count], 1):
            img_path = result.get("image_path", "")
            if not img_path or not Path(img_path).is_file():
                logger.debug("Image not found, skipping visualization: %s", img_path)
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.debug("Failed to read image: %s", img_path)
                continue

            h, w = img.shape[:2]

            # -- Draw original annotations in GREEN --
            annotations = result.get("annotations", [])
            if annotations:
                ann_array = np.array(annotations, dtype=np.float64)
                cls_ids = ann_array[:, 0].astype(np.int64)
                boxes_pixel = _yolo_to_pixel(ann_array[:, 1:5], w, h)
                green_colors = {int(c): (0, 200, 0) for c in np.unique(cls_ids)}
                img = draw_bboxes(img, boxes_pixel, cls_ids, colors=green_colors)

            # -- Draw SAM3 suggested bboxes in RED (if present) --
            sam3 = result.get("sam3_verification") or {}
            sam3_boxes_raw = sam3.get("suggested_boxes", [])
            if sam3_boxes_raw:
                sam3_arr = np.array(sam3_boxes_raw, dtype=np.float64).reshape(-1, 5)
                sam3_cls = sam3_arr[:, 0].astype(np.int64)
                sam3_pixel = _yolo_to_pixel(sam3_arr[:, 1:5], w, h)
                red_colors = {int(c): (0, 0, 220) for c in np.unique(sam3_cls)}
                img = draw_bboxes(img, sam3_pixel, sam3_cls, colors=red_colors)

            # -- Issue text overlay --
            issues = result.get("validation_issues", [])
            quality = result.get("quality_score", 0.0)
            grade = result.get("grade", "?")
            header = f"score={quality:.3f}  grade={grade}  issues={len(issues)}"
            cv2.putText(
                img, header, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )
            for idx, issue in enumerate(issues[:5]):
                text = f"- {issue.get('type', '?')}: {issue.get('detail', '')}"
                y_pos = 50 + idx * 20
                cv2.putText(
                    img, text[:90], (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA,
                )

            stem = Path(img_path).stem
            out_path = vis_dir / f"{rank:03d}_{stem}.png"
            cv2.imwrite(str(out_path), img)

        logger.info("Saved %d worst-image visualizations", min(count, len(sorted_results)))

    def _save_fixes_json(self, image_results: List[Dict]) -> None:
        """Save auto-fixable corrections to ``fixes.json``.

        Only results that contain ``suggested_fixes`` entries are included.
        """
        fixes: List[Dict] = []
        for result in image_results:
            for fix in result.get("suggested_fixes", []):
                fixes.append({
                    "image_path": result.get("image_path", ""),
                    "label_path": _label_path_from_image(result.get("image_path", "")),
                    "fix_type": fix.get("type", ""),
                    "annotation_idx": fix.get("annotation_idx"),
                    "original": fix.get("original"),
                    "suggested": fix.get("suggested"),
                    "reason": fix.get("reason", ""),
                })

        payload = {
            "total_fixable": len(fixes),
            "fixes": [make_serialisable(f) for f in fixes],
        }

        path = self.report_dir / "fixes.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Saved fixes.json (%d fixable issues)", len(fixes))


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _yolo_to_pixel(boxes_norm: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalised YOLO cxcywh boxes to pixel xyxy coordinates."""
    boxes_pixel = xywh_to_xyxy(boxes_norm)
    boxes_pixel[:, [0, 2]] *= img_w
    boxes_pixel[:, [1, 3]] *= img_h
    return boxes_pixel


def _label_path_from_image(image_path: str) -> str:
    """Derive the YOLO label path from an image path.

    Follows the standard convention: ``images/ -> labels/`` with ``.txt``
    extension.
    """
    p = Path(image_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return str(Path(*parts).with_suffix(".txt")) if parts else ""

