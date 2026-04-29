"""Point & Call (Shisa-Kanko) detection tab.

Rule-based detector for the Japanese 指差呼称 crosswalk gesture sequence:
worker stops at curb -> points right -> points left -> (optional front) ->
crosses. Pretrained-only — uses DWPose ONNX + yolo11n person detector.

This tab has a SOFT dependency on the U3 orchestrator code at
``features.safety_poketenashi_point_and_call.code.orchestrator.PointAndCallOrchestrator``.
The orchestrator is imported lazily; if missing, the tab renders a warning
placeholder so the rest of the app_demo still launches.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import supervision as sv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from app_demo.utils import bgr_to_rgb, create_status_html, rgb_to_bgr

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _use_case_config(config: dict) -> dict:
    """Return the ``point_and_call`` use_case block from app_demo config."""
    return config.get("use_cases", {}).get("point_and_call", {}) or {}


def _pose_choices(config: dict) -> dict[str, str]:
    """Map dropdown label -> pose YAML config path (from app_demo config)."""
    pose_configs = _use_case_config(config).get("pose_configs", {}) or {}
    return {
        "DWPose": pose_configs.get("dwpose", ""),
        "RTMPose": pose_configs.get("rtmpose", ""),
        "MediaPipe": pose_configs.get("mediapipe", ""),
    }


# ---------------------------------------------------------------------------
# Lazy orchestrator import
# ---------------------------------------------------------------------------


def _load_orchestrator_class() -> tuple[Any | None, str | None]:
    """Try to import ``PointAndCallOrchestrator`` from U3 code.

    Returns:
        Tuple ``(class_or_None, error_message_or_None)``. The class is None
        when U3 is not yet on main; ``error_message`` carries the failure
        reason for display.
    """
    try:
        from features.safety_poketenashi_point_and_call.code.orchestrator import (
            PointAndCallOrchestrator,
        )
    except ImportError as exc:  # U3 not yet merged
        return None, str(exc)
    return PointAndCallOrchestrator, None


# ---------------------------------------------------------------------------
# Status badge helper
# ---------------------------------------------------------------------------


def _status_badge(message: str, style: str = "safe") -> str:
    return create_status_html(status=style, message=message)


def _make_alert_html(state: str) -> str:
    """Render a banner reflecting current sequence state.

    States: ``idle``, ``in_progress``, ``done``, ``missing_directions``.
    """
    if state == "done":
        return _status_badge("POINT-AND-CALL COMPLETED", style="safe")
    if state == "missing_directions":
        return _status_badge("MISSING DIRECTIONS", style="alert")
    if state == "in_progress":
        return _status_badge("SEQUENCE IN PROGRESS", style="warning")
    return _status_badge("IDLE", style="safe")


# ---------------------------------------------------------------------------
# Orchestrator construction
# ---------------------------------------------------------------------------


def _build_orchestrator(
    orch_cls: Any,
    pose_label: str,
    pose_choices: dict[str, str],
    config: dict,
    project_root: Path,
) -> tuple[Any | None, str | None]:
    """Instantiate the orchestrator with the requested pose backend.

    Returns ``(orchestrator, None)`` on success or ``(None, error)`` on
    failure (e.g. config missing on disk).
    """
    pose_cfg_rel = pose_choices.get(pose_label, "")
    if not pose_cfg_rel:
        return None, f"No pose config registered for '{pose_label}'."
    pose_cfg_abs = project_root / pose_cfg_rel
    if not pose_cfg_abs.exists():
        return None, f"Pose config not found on disk: {pose_cfg_rel}"

    uc_cfg = _use_case_config(config)
    inference_cfg_rel = uc_cfg.get("inference_config", "")
    inference_cfg_abs = (
        project_root / inference_cfg_rel if inference_cfg_rel else None
    )

    try:
        orchestrator = orch_cls(
            pose_config=str(pose_cfg_abs),
            inference_config=(
                str(inference_cfg_abs) if inference_cfg_abs and inference_cfg_abs.exists() else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 - surface any orchestrator init error
        logger.exception("Failed to build PointAndCallOrchestrator")
        return None, f"Orchestrator init failed: {exc}"

    return orchestrator, None


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------


def _run_image(
    image: np.ndarray | None,
    pose_label: str,
    orch_cls: Any,
    pose_choices: dict[str, str],
    config: dict,
    project_root: Path,
) -> tuple[np.ndarray | None, str, str]:
    """Single-frame inference -> annotated RGB image + alert HTML + JSON."""
    if image is None:
        return None, _make_alert_html("idle"), json.dumps({"error": "No image provided."})

    orchestrator, err = _build_orchestrator(
        orch_cls, pose_label, pose_choices, config, project_root
    )
    if orchestrator is None:
        return None, _make_alert_html("idle"), json.dumps({"error": err})

    bgr = rgb_to_bgr(image)

    try:
        result = orchestrator.process(bgr)
        annotated_bgr = orchestrator.draw(bgr, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Orchestrator failed on image")
        return None, _make_alert_html("idle"), json.dumps({"error": str(exc)})

    annotated_rgb = bgr_to_rgb(annotated_bgr)

    state = "idle"
    if isinstance(result, dict):
        state = str(result.get("sequence_state", "idle"))
    return annotated_rgb, _make_alert_html(state), json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Video inference
# ---------------------------------------------------------------------------


def _run_video(
    video_path: str | None,
    pose_label: str,
    orch_cls: Any,
    pose_choices: dict[str, str],
    config: dict,
    project_root: Path,
) -> tuple[str | None, str]:
    """Run orchestrator frame-by-frame over a video and write annotated mp4."""
    if not video_path:
        return None, "No video provided."

    orchestrator, err = _build_orchestrator(
        orch_cls, pose_label, pose_choices, config, project_root
    )
    if orchestrator is None:
        return None, err or "Failed to build orchestrator."

    try:
        video_info = sv.VideoInfo.from_video_path(video_path)
        generator = sv.get_video_frames_generator(video_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to read video: {exc}"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    counts: dict[str, int] = {}
    final_state = "idle"
    frame_idx = 0

    with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
        for frame in generator:
            try:
                result = orchestrator.process(frame)
                annotated = orchestrator.draw(frame, result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Orchestrator failed on frame %d", frame_idx)
                annotated = frame
                result = {"error": str(exc)}

            if isinstance(result, dict):
                state = str(result.get("sequence_state", "idle"))
                counts[state] = counts.get(state, 0) + 1
                final_state = state

            sink.write_frame(annotated)
            frame_idx += 1

    summary_lines = [
        f"Pose backend: {pose_label}",
        f"Total frames: {frame_idx}",
        f"Resolution: {video_info.width}x{video_info.height}",
        f"FPS: {video_info.fps}",
        f"Final state: {final_state}",
        "",
        "--- State frame counts ---",
    ]
    if counts:
        for state, n in counts.items():
            summary_lines.append(f"  {state}: {n}")
    else:
        summary_lines.append("  (none)")

    return output_path, "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_tab_point_and_call(manager: Any, config: dict) -> None:
    """Build the Point & Call tab (lazy-imports the U3 orchestrator).

    Args:
        manager: ``ModelManager`` instance (kept for API parity; the tab
            does not currently use it because the orchestrator owns its
            own predictors).
        config: Demo config dict; must contain
            ``use_cases.point_and_call`` with ``pose_configs``.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    with gr.Tab("Point & Call"):
        gr.Markdown("## Point & Call (Shisa-Kanko)")
        gr.Markdown(
            "Rule-based detector for the Japanese crosswalk gesture sequence: "
            "stop -> point right -> point left -> (optional front) -> cross."
        )

        orch_cls, import_error = _load_orchestrator_class()
        if orch_cls is None:
            gr.Markdown(
                "**safety-poketenashi_point_and_call code not yet on main; merge U3/U5 first.**\n\n"
                f"Import error: `{import_error}`"
            )
            return

        pose_choices = _pose_choices(config)
        pose_options = list(pose_choices.keys())
        default_pose = pose_options[0] if pose_options else "DWPose"

        with gr.Row():
            pose_dropdown = gr.Dropdown(
                choices=pose_options,
                value=default_pose,
                label="Pose Backend",
                interactive=True,
            )

        with gr.Tabs():
            # ---- Image sub-tab ----
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="numpy", label="Upload Image")
                        image_btn = gr.Button("Run", variant="primary")
                    with gr.Column():
                        image_output = gr.Image(type="numpy", label="Annotated Result")
                        image_alert = gr.HTML(value=_make_alert_html("idle"))
                        image_json = gr.Textbox(
                            label="Per-frame Result (JSON)",
                            lines=12,
                            interactive=False,
                        )

                image_btn.click(
                    fn=lambda img, pose: _run_image(
                        img, pose, orch_cls, pose_choices, config, project_root
                    ),
                    inputs=[image_input, pose_dropdown],
                    outputs=[image_output, image_alert, image_json],
                )

            # ---- Video sub-tab ----
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_btn = gr.Button("Process Video", variant="primary")
                    with gr.Column():
                        video_output = gr.Video(label="Processed Video")
                        video_log = gr.Textbox(
                            label="Sequence Summary",
                            lines=14,
                            interactive=False,
                        )

                video_btn.click(
                    fn=lambda vid, pose: _run_video(
                        vid, pose, orch_cls, pose_choices, config, project_root
                    ),
                    inputs=[video_input, pose_dropdown],
                    outputs=[video_output, video_log],
                )
