"""Video Analytics Dashboard tab — class distribution, alert timeline, and processing stats.

Reads the last video processing summary stored by the detection tab (or any
other video-processing tab) and renders charts and statistics.
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.tabs.tab_detection import get_video_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_class_distribution_chart(summary: dict) -> Optional[str]:
    """Create a bar chart of class counts and return the image file path."""
    class_counts = summary.get("class_counts", {})
    if not class_counts:
        return None

    classes = list(class_counts.keys())
    counts = [int(class_counts[c]) for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, counts, color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Class")
    ax.set_ylabel("Total Detections")
    ax.set_title("Class Distribution")
    ax.tick_params(axis="x", rotation=45)

    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="analytics_")
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _format_alert_timeline(summary: dict) -> str:
    """Format alerts into a readable text timeline."""
    alerts = summary.get("alerts", [])
    if not alerts:
        return "No alerts recorded."

    lines = []
    for i, alert in enumerate(alerts, 1):
        if isinstance(alert, dict):
            frame = alert.get("frame", "?")
            cls = alert.get("class", alert.get("type", "unknown"))
            conf = alert.get("confidence", alert.get("score", 0))
            msg = alert.get("message", "")
            line = f"[{i}] Frame {frame} | {cls}"
            if conf:
                line += f" (conf: {conf:.2f})"
            if msg:
                line += f" - {msg}"
            lines.append(line)
        else:
            lines.append(f"[{i}] {alert}")

    return "\n".join(lines)


def _format_processing_stats(summary: dict) -> str:
    """Format processing statistics as readable text."""
    if not summary:
        return "No video has been processed yet. Process a video in the Object Detection tab first."

    fps = summary.get("fps", 0)
    total_frames = summary.get("total_frames", 0)
    processed_frames = summary.get("processed_frames", 0)
    total_detections = summary.get("total_detections", 0)

    lines = [
        f"Processing FPS:    {fps:.1f}" if isinstance(fps, (int, float)) else f"Processing FPS:    {fps}",
        f"Total Frames:      {total_frames}",
        f"Processed Frames:  {processed_frames}",
        f"Total Detections:  {total_detections}",
    ]
    return "\n".join(lines)


def _refresh_analytics() -> tuple:
    """Refresh all analytics components from the latest video summary."""
    summary = get_video_summary()

    if not summary:
        empty_msg = "No video has been processed yet. Process a video in the Object Detection tab first."
        return None, empty_msg, empty_msg

    chart_path = _make_class_distribution_chart(summary)
    alert_text = _format_alert_timeline(summary)
    stats_text = _format_processing_stats(summary)

    return chart_path, alert_text, stats_text


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_analytics(manager: Any, config: dict) -> None:
    """Build the Video Analytics Dashboard tab inside a ``gr.Tab`` context.

    Args:
        manager: ``ModelManager`` instance (unused here but kept for uniform API).
        config: Demo config dict.
    """
    with gr.Tab("Video Analytics"):
        refresh_btn = gr.Button("Refresh Analytics", variant="secondary")

        with gr.Row():
            chart_image = gr.Image(
                type="filepath",
                label="Class Distribution",
            )
            alert_textbox = gr.Textbox(
                label="Alert Timeline",
                lines=12,
                interactive=False,
            )

        with gr.Row():
            stats_textbox = gr.Textbox(
                label="Processing Stats",
                lines=6,
                interactive=False,
                value="No video has been processed yet. Process a video in the Object Detection tab first.",
            )

        refresh_btn.click(
            fn=_refresh_analytics,
            inputs=[],
            outputs=[chart_image, alert_textbox, stats_textbox],
        )
