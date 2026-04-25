"""Gradio demo for SAM3 image and video segmentation.

Calls the SAM3 REST service (default: http://localhost:18100) — no local
model loading.  Runs inference once per "Run" click, caches raw masks in
``gr.State``, then re-renders instantly when visualization controls change.

Run::

    uv run python demo.py
    uv run python demo.py --url http://localhost:18100 --server-port 7861 --share
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import gradio as gr
import numpy as np
import requests as _req
import supervision as sv
from loguru import logger as _logger
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAM3_URL: str = os.getenv("SAM3_URL", "http://localhost:18100")

_DEFAULT_OPACITY: float = 0.4
_DEFAULT_DET_THR: float = 0.5
_DEFAULT_MASK_THR: float = 0.5
_DEFAULT_AUTO_THR: float = 0.2
_DEFAULT_MAX_FRAMES: int = 100

_AUTO_MASK_PROMPTS = [
    "person. car. truck. dog. cat. bird.",
    "chair. table. laptop. phone. bottle. cup.",
]

_DEMO_DIR = Path(__file__).parent / "demo"

# Stateless annotators — instantiated once, reused across all renders
_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
_LABEL_ANNOTATOR = sv.LabelAnnotator(text_scale=0.5, text_padding=3)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _encode_image(image: Image.Image) -> str:
    """PIL Image → base64-encoded JPEG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _encode_frame(frame_rgb: np.ndarray) -> str:
    return _encode_image(Image.fromarray(frame_rgb))


def _encode_frames(frames_rgb: list[np.ndarray]) -> list[str]:
    """Encode frames to base64 JPEG in parallel using a thread pool."""
    with ThreadPoolExecutor() as ex:
        return list(ex.map(_encode_frame, frames_rgb))


def _get_service_info() -> str:
    try:
        data = _req.get(f"{SAM3_URL}/health", timeout=5).json()
        return (
            f"SAM3 @ `{SAM3_URL}`  |  "
            f"Model: `{data['model']}`  |  "
            f"Device: `{data['device']}`  |  "
            f"Dtype: `{data['dtype']}`"
        )
    except Exception:
        return f"SAM3 @ `{SAM3_URL}`  |  ⚠ service unreachable"


# ---------------------------------------------------------------------------
# Core rendering helpers
# ---------------------------------------------------------------------------


def _decode_mask(b64: str) -> np.ndarray:
    """Base64-encoded PNG → bool (H, W) numpy array."""
    buf = io.BytesIO(base64.b64decode(b64))
    return np.array(Image.open(buf).convert("L")) > 0


def _decode_dets(dets: list[dict]) -> list[dict]:
    """Decode each detection's base64 mask to a numpy array in-place."""
    for d in dets:
        d["mask"] = _decode_mask(d["mask"])
    return dets


def _parse_box_coords(box_text: str) -> list[int]:
    """Parse 'x1,y1,x2,y2' string → list of 4 ints. Raises ValueError on bad input."""
    coords = [int(x.strip()) for x in box_text.strip().split(",")]
    if len(coords) != 4:
        raise ValueError("Expected 4 coordinates")
    return coords


def _build_sv_detections(dets: list[dict], selected: set[int]) -> sv.Detections:
    """Filter detection dicts (with pre-decoded numpy masks) to *selected* indices."""
    xyxy, masks, confidences, class_ids = [], [], [], []
    for i, det in enumerate(dets):
        if i not in selected:
            continue
        masks.append(det["mask"])  # already a numpy bool array after _decode_dets
        confidences.append(float(det.get("score", 0.0)))
        class_ids.append(i)
        b = det["bbox"]
        xyxy.append([b["x1"], b["y1"], b["x2"], b["y2"]])

    if not masks:
        return sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=np.int32),
        )
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.stack(masks),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=np.int32),
    )


def _render_image(
    original_rgb: np.ndarray,
    sv_dets: sv.Detections,
    opacity: float,
    show_boxes: bool,
    show_labels: bool,
) -> np.ndarray:
    """Annotate an RGB frame with supervision annotators. Returns RGB."""
    if len(sv_dets) == 0:
        return original_rgb.copy()
    bgr = original_rgb[:, :, ::-1].copy()
    if sv_dets.mask is not None:
        bgr = sv.MaskAnnotator(opacity=opacity).annotate(bgr, sv_dets)
    if show_boxes:
        bgr = _BOX_ANNOTATOR.annotate(bgr, sv_dets)
    if show_labels:
        labels = [
            f"obj_{sv_dets.class_id[i]} ({sv_dets.confidence[i]:.2f})"
            for i in range(len(sv_dets))
        ]
        bgr = _LABEL_ANNOTATOR.annotate(bgr, sv_dets, labels=labels)
    return bgr[:, :, ::-1]


def _det_choices(dets: list[dict], prefix: str = "obj") -> list[tuple[str, str]]:
    safe = prefix[:12].replace(" ", "_")
    return [
        (f"{safe}_{i}  score={d.get('score', 0.0):.2f}", str(i))
        for i, d in enumerate(dets)
    ]


def _det_info(dets: list[dict]) -> list[dict]:
    return [
        {
            "id": i,
            "bbox": d["bbox"],
            "score": round(d.get("score", 0.0), 4),
            "area": round(d.get("area", 0.0), 4),
        }
        for i, d in enumerate(dets)
    ]


# ---------------------------------------------------------------------------
# Video helpers — PyAV only
# ---------------------------------------------------------------------------


def _extract_frames(video_path: str, max_frames: int = _DEFAULT_MAX_FRAMES) -> list[np.ndarray]:
    """Extract up to *max_frames* RGB frames from *video_path* using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0
    step = max(1, total // max_frames) if total > 0 else 1
    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % step == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    container.close()
    return frames


def _write_video(frames: list[np.ndarray], fps: float = 15.0) -> str:
    """Write RGB frames to a temp MP4 using PyAV. Returns file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    if not frames:
        return tmp.name
    h, w = frames[0].shape[:2]
    container = av.open(tmp.name, "w")
    try:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.options = {"crf": "23", "preset": "fast"}
    except Exception:
        stream = container.add_stream("mpeg4", rate=int(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for rgb in frames:
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return tmp.name


def _unlink_safe(path: str | None) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Image: re-render
# ---------------------------------------------------------------------------


def _rerender_image(
    state: dict,
    selected: list[str],
    opacity: float,
    show_boxes: bool,
    show_labels: bool,
) -> np.ndarray | None:
    if not state:
        return None
    sel = {int(v) for v in selected}
    return _render_image(
        state["original_rgb"],
        _build_sv_detections(state["detections"], sel),
        opacity, show_boxes, show_labels,
    )


# ---------------------------------------------------------------------------
# Image: inference
# ---------------------------------------------------------------------------


def _run_text_segment(
    image: Image.Image | None, text: str, det_thr: float, mask_thr: float,
) -> tuple:
    if image is None or not text.strip():
        return {}, gr.update(choices=[], value=[]), None, []
    try:
        resp = _req.post(
            f"{SAM3_URL}/segment_text",
            json={"image": _encode_image(image), "text": text.strip(),
                  "detection_threshold": det_thr, "mask_threshold": mask_thr},
            timeout=120,
        )
        resp.raise_for_status()
        dets = _decode_dets(resp.json()["detections"])
    except Exception as exc:
        gr.Warning(f"SAM3 service error: {exc}")
        return {}, gr.update(choices=[], value=[]), None, []

    orig = np.array(image.convert("RGB"))
    state = {"detections": dets, "original_rgb": orig}
    vals = [str(i) for i in range(len(dets))]
    output = _render_image(orig, _build_sv_detections(dets, set(range(len(dets)))), _DEFAULT_OPACITY, True, True)
    return state, gr.update(choices=_det_choices(dets, text.strip()), value=vals), output, _det_info(dets)


def _run_box_segment(image: Image.Image | None, box_text: str) -> tuple:
    if image is None or not box_text.strip():
        return {}, gr.update(choices=[], value=[]), None, []
    try:
        coords = _parse_box_coords(box_text)
    except ValueError as exc:
        gr.Warning(str(exc))
        return {}, gr.update(choices=[], value=[]), None, []
    try:
        resp = _req.post(
            f"{SAM3_URL}/segment_box",
            json={"image": _encode_image(image), "box": coords},
            timeout=120,
        )
        resp.raise_for_status()
        dets = _decode_dets([resp.json()["result"]])
    except Exception as exc:
        gr.Warning(f"SAM3 service error: {exc}")
        return {}, gr.update(choices=[], value=[]), None, []

    orig = np.array(image.convert("RGB"))
    state = {"detections": dets, "original_rgb": orig}
    output = _render_image(orig, _build_sv_detections(dets, {0}), _DEFAULT_OPACITY, True, True)
    return state, gr.update(choices=_det_choices(dets, "box"), value=["0"]), output, _det_info(dets)


def _run_auto_segment(image: Image.Image | None, threshold: float) -> tuple:
    if image is None:
        return {}, gr.update(choices=[], value=[]), None, []
    try:
        resp = _req.post(
            f"{SAM3_URL}/auto_mask",
            json={"image": _encode_image(image), "threshold": threshold},
            timeout=180,
        )
        resp.raise_for_status()
        dets = _decode_dets(resp.json()["detections"])
    except Exception as exc:
        gr.Warning(f"SAM3 service error: {exc}")
        return {}, gr.update(choices=[], value=[]), None, []

    orig = np.array(image.convert("RGB"))
    state = {"detections": dets, "original_rgb": orig}
    vals = [str(i) for i in range(len(dets))]
    output = _render_image(orig, _build_sv_detections(dets, set(range(len(dets)))), _DEFAULT_OPACITY, True, True)
    return state, gr.update(choices=_det_choices(dets, "auto"), value=vals), output, _det_info(dets)


# ---------------------------------------------------------------------------
# Video: re-render with render-key caching and temp-file cleanup
# ---------------------------------------------------------------------------


def _rerender_video(
    state: dict,
    selected: list[str],
    opacity: float,
    show_boxes: bool,
    show_labels: bool,
) -> tuple[str | None, dict]:
    """Re-render video. Returns (video_path, updated_state).

    Skips re-encoding if visualization parameters haven't changed.
    Cleans up the previous temp MP4 when a new one is written.
    """
    if not state:
        return None, state

    sel = {int(v) for v in selected}
    cache_key = (tuple(sorted(sel)), opacity, show_boxes, show_labels)

    if state.get("_render_key") == cache_key:
        return state.get("_video_path"), state

    annotated = [
        _render_image(frame_rgb, _build_sv_detections(state["det_map"].get(fi, []), sel), opacity, show_boxes, show_labels)
        for fi, frame_rgb in enumerate(state["frames_rgb"])
    ]

    _unlink_safe(state.get("_video_path"))
    video_path = _write_video(annotated)
    new_state = {**state, "_render_key": cache_key, "_video_path": video_path}
    return video_path, new_state


# ---------------------------------------------------------------------------
# Video: inference
# ---------------------------------------------------------------------------


def _run_video_text(video_path: str | None, text: str, max_frames: int) -> tuple:
    if not video_path or not text.strip():
        gr.Warning("Provide both a video and a text prompt.")
        return {}, gr.update(choices=[], value=[]), None

    frames_rgb = _extract_frames(video_path, max_frames)
    if not frames_rgb:
        gr.Warning("Could not extract frames from video.")
        return {}, gr.update(choices=[], value=[]), None

    session_id = f"demo-{uuid.uuid4().hex[:8]}"
    try:
        resp = _req.post(
            f"{SAM3_URL}/sessions",
            json={"mode": "video", "frames": _encode_frames(frames_rgb), "text": text.strip()},
            timeout=300,
        )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

        prop = _req.post(
            f"{SAM3_URL}/sessions/{session_id}/propagate",
            json={"max_frames": max_frames},
            timeout=300,
        )
        prop.raise_for_status()
        det_map = {
            r["frame_idx"]: _decode_dets(r["detections"])
            for r in prop.json()["frames"]
        }
    except Exception as exc:
        gr.Warning(f"SAM3 service error: {exc}")
        return {}, gr.update(choices=[], value=[]), None
    finally:
        try:
            _req.delete(f"{SAM3_URL}/sessions/{session_id}", timeout=10)
        except Exception as exc:
            _logger.warning("Failed to delete session %s: %s", session_id, exc)

    max_dets = max((len(v) for v in det_map.values()), default=0)
    choices = [(f"obj_{i}", str(i)) for i in range(max_dets)]
    vals = [str(i) for i in range(max_dets)]
    state = {"det_map": det_map, "frames_rgb": frames_rgb}
    video_out, state = _rerender_video(state, vals, _DEFAULT_OPACITY, True, True)
    return state, gr.update(choices=choices, value=vals), video_out


def _run_video_box(video_path: str | None, box_text: str, max_frames: int) -> tuple:
    if not video_path or not box_text.strip():
        gr.Warning("Provide both a video and a box prompt (x1,y1,x2,y2).")
        return {}, gr.update(choices=[], value=[]), None
    try:
        coords = _parse_box_coords(box_text)
    except ValueError as exc:
        gr.Warning(str(exc))
        return {}, gr.update(choices=[], value=[]), None

    frames_rgb = _extract_frames(video_path, max_frames)
    if not frames_rgb:
        gr.Warning("Could not extract frames from video.")
        return {}, gr.update(choices=[], value=[]), None

    session_id = f"demo-{uuid.uuid4().hex[:8]}"
    try:
        resp = _req.post(
            f"{SAM3_URL}/sessions",
            json={"mode": "tracker", "frames": _encode_frames(frames_rgb)},
            timeout=300,
        )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

        _req.post(
            f"{SAM3_URL}/sessions/{session_id}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [coords]},
            timeout=60,
        ).raise_for_status()

        prop = _req.post(
            f"{SAM3_URL}/sessions/{session_id}/propagate",
            json={"max_frames": max_frames},
            timeout=300,
        )
        prop.raise_for_status()
        det_map = {
            r["frame_idx"]: _decode_dets(r["detections"])
            for r in prop.json()["frames"]
        }
    except Exception as exc:
        gr.Warning(f"SAM3 service error: {exc}")
        return {}, gr.update(choices=[], value=[]), None
    finally:
        try:
            _req.delete(f"{SAM3_URL}/sessions/{session_id}", timeout=10)
        except Exception as exc:
            _logger.warning("Failed to delete session %s: %s", session_id, exc)

    choices = [("obj_0  (tracked)", "0")]
    state = {"det_map": det_map, "frames_rgb": frames_rgb}
    video_out, state = _rerender_video(state, ["0"], _DEFAULT_OPACITY, True, True)
    return state, gr.update(choices=choices, value=["0"]), video_out


# ---------------------------------------------------------------------------
# Shared visualization controls (placed in right column, below output)
# ---------------------------------------------------------------------------


def _obj_filter() -> gr.CheckboxGroup:
    return gr.CheckboxGroup(choices=[], value=[], label="Objects", interactive=True)


def _viz_row() -> tuple[gr.Slider, gr.Checkbox, gr.Checkbox]:
    with gr.Row(equal_height=True):
        opacity = gr.Slider(
            minimum=0.1, maximum=0.9, step=0.05, value=_DEFAULT_OPACITY,
            label="Opacity", scale=3,
        )
        show_boxes = gr.Checkbox(value=True, label="Boxes", scale=1, min_width=60)
        show_labels = gr.Checkbox(value=True, label="Labels", scale=1, min_width=60)
    return opacity, show_boxes, show_labels


# ---------------------------------------------------------------------------
# Demo builder
# ---------------------------------------------------------------------------


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="SAM3 Segmentation Demo") as demo:

        gr.Markdown(_get_service_info())

        with gr.Tabs():

            # ── Image: Text Prompt ─────────────────────────────────────────
            with gr.Tab("Text Prompt"):
                img_state_t = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        img_in_t = gr.Image(type="pil", label="Image")
                        txt_prompt = gr.Textbox(placeholder="person. car. dog.", label="Text prompt")
                        with gr.Accordion("Inference settings", open=False):
                            det_thr_t = gr.Slider(0.0, 1.0, step=0.05, value=_DEFAULT_DET_THR, label="Detection threshold")
                            mask_thr_t = gr.Slider(0.0, 1.0, step=0.05, value=_DEFAULT_MASK_THR, label="Mask threshold")
                        run_t = gr.Button("Run", variant="primary")
                    with gr.Column(scale=2):
                        out_t = gr.Image(label="Output", type="numpy")
                        opac_t, boxes_t, lbls_t = _viz_row()
                        info_t = gr.JSON(label="Detections")
                    with gr.Column(scale=1, min_width=180):
                        obj_t = _obj_filter()
                gr.Examples(
                    examples=[
                        [str(_DEMO_DIR / "truck.jpg"), "truck", 0.5, 0.5],
                        [str(_DEMO_DIR / "cars.jpg"), "car", 0.4, 0.5],
                        [str(_DEMO_DIR / "groceries.jpg"), "package. bag. food.", 0.3, 0.5],
                    ],
                    inputs=[img_in_t, txt_prompt, det_thr_t, mask_thr_t],
                    label="Examples",
                )
                run_t.click(
                    _run_text_segment,
                    inputs=[img_in_t, txt_prompt, det_thr_t, mask_thr_t],
                    outputs=[img_state_t, obj_t, out_t, info_t],
                )
                for ctrl in [obj_t, opac_t, boxes_t, lbls_t]:
                    ctrl.change(
                        _rerender_image,
                        inputs=[img_state_t, obj_t, opac_t, boxes_t, lbls_t],
                        outputs=[out_t],
                    )

            # ── Image: Box Prompt ──────────────────────────────────────────
            with gr.Tab("Box Prompt"):
                img_state_b = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        img_in_b = gr.Image(type="pil", label="Image")
                        box_prompt_in = gr.Textbox(placeholder="x1,y1,x2,y2", label="Box prompt (pixels)")
                        run_b = gr.Button("Run", variant="primary")
                    with gr.Column(scale=2):
                        out_b = gr.Image(label="Output", type="numpy")
                        opac_b, boxes_b, lbls_b = _viz_row()
                        info_b = gr.JSON(label="Detections")
                    with gr.Column(scale=1, min_width=180):
                        obj_b = _obj_filter()
                gr.Examples(
                    examples=[
                        [str(_DEMO_DIR / "truck.jpg"), "350,150,1500,800"],
                        [str(_DEMO_DIR / "cars.jpg"), "50,300,600,1100"],
                    ],
                    inputs=[img_in_b, box_prompt_in],
                    label="Examples",
                )
                run_b.click(
                    _run_box_segment,
                    inputs=[img_in_b, box_prompt_in],
                    outputs=[img_state_b, obj_b, out_b, info_b],
                )
                for ctrl in [obj_b, opac_b, boxes_b, lbls_b]:
                    ctrl.change(
                        _rerender_image,
                        inputs=[img_state_b, obj_b, opac_b, boxes_b, lbls_b],
                        outputs=[out_b],
                    )

            # ── Image: Auto Mask ───────────────────────────────────────────
            with gr.Tab("Auto Mask"):
                img_state_a = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        img_in_a = gr.Image(type="pil", label="Image")
                        auto_thr = gr.Slider(0.0, 1.0, step=0.05, value=_DEFAULT_AUTO_THR, label="Detection threshold")
                        gr.Markdown("Prompts: " + " · ".join(f"`{p[:30]}`" for p in _AUTO_MASK_PROMPTS))
                        run_a = gr.Button("Segment Everything", variant="primary")
                    with gr.Column(scale=2):
                        out_a = gr.Image(label="Output", type="numpy")
                        opac_a, boxes_a, lbls_a = _viz_row()
                        info_a = gr.JSON(label="Detections")
                    with gr.Column(scale=1, min_width=180):
                        obj_a = _obj_filter()
                gr.Examples(
                    examples=[
                        [str(_DEMO_DIR / "cars.jpg"), 0.3],
                        [str(_DEMO_DIR / "groceries.jpg"), 0.25],
                        [str(_DEMO_DIR / "truck.jpg"), 0.35],
                    ],
                    inputs=[img_in_a, auto_thr],
                    label="Examples",
                )
                run_a.click(
                    _run_auto_segment,
                    inputs=[img_in_a, auto_thr],
                    outputs=[img_state_a, obj_a, out_a, info_a],
                )
                for ctrl in [obj_a, opac_a, boxes_a, lbls_a]:
                    ctrl.change(
                        _rerender_image,
                        inputs=[img_state_a, obj_a, opac_a, boxes_a, lbls_a],
                        outputs=[out_a],
                    )

            # ── Video: Text Tracking ───────────────────────────────────────
            with gr.Tab("Text Tracking"):
                vid_state_t = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        vid_in_t = gr.Video(label="Video")
                        vid_txt = gr.Textbox(placeholder="person. bicycle. car.", label="Text prompt")
                        with gr.Accordion("Inference settings", open=False):
                            vid_max_t = gr.Slider(10, 300, step=10, value=_DEFAULT_MAX_FRAMES, label="Max frames")
                        run_vt = gr.Button("Run", variant="primary")
                    with gr.Column(scale=2):
                        out_vt = gr.Video(label="Output Video")
                        opac_vt, boxes_vt, lbls_vt = _viz_row()
                    with gr.Column(scale=1, min_width=180):
                        obj_vt = _obj_filter()
                gr.Examples(
                    examples=[
                        [str(_DEMO_DIR / "people-detection.mp4"), "person", 100],
                        [str(_DEMO_DIR / "person-bicycle-car-detection.mp4"), "person. bicycle. car.", 100],
                    ],
                    inputs=[vid_in_t, vid_txt, vid_max_t],
                    label="Examples",
                )
                run_vt.click(
                    _run_video_text,
                    inputs=[vid_in_t, vid_txt, vid_max_t],
                    outputs=[vid_state_t, obj_vt, out_vt],
                )
                for ctrl in [obj_vt, opac_vt, boxes_vt, lbls_vt]:
                    ctrl.change(
                        _rerender_video,
                        inputs=[vid_state_t, obj_vt, opac_vt, boxes_vt, lbls_vt],
                        outputs=[out_vt, vid_state_t],
                    )

            # ── Video: Box Tracker ─────────────────────────────────────────
            with gr.Tab("Box Tracker"):
                vid_state_b = gr.State({})
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        vid_in_b = gr.Video(label="Video")
                        vid_box_in = gr.Textbox(placeholder="x1,y1,x2,y2  (first frame)", label="Box prompt (pixels)")
                        with gr.Accordion("Inference settings", open=False):
                            vid_max_b = gr.Slider(10, 300, step=10, value=_DEFAULT_MAX_FRAMES, label="Max frames")
                        run_vb = gr.Button("Run", variant="primary")
                    with gr.Column(scale=2):
                        out_vb = gr.Video(label="Output Video")
                        opac_vb, boxes_vb, lbls_vb = _viz_row()
                    with gr.Column(scale=1, min_width=180):
                        obj_vb = _obj_filter()
                gr.Examples(
                    examples=[
                        [str(_DEMO_DIR / "people-detection.mp4"), "200,50,450,380", 100],
                        [str(_DEMO_DIR / "person-bicycle-car-detection.mp4"), "100,50,280,350", 100],
                    ],
                    inputs=[vid_in_b, vid_box_in, vid_max_b],
                    label="Examples",
                )
                run_vb.click(
                    _run_video_box,
                    inputs=[vid_in_b, vid_box_in, vid_max_b],
                    outputs=[vid_state_b, obj_vb, out_vb],
                )
                for ctrl in [obj_vb, opac_vb, boxes_vb, lbls_vb]:
                    ctrl.change(
                        _rerender_video,
                        inputs=[vid_state_b, obj_vb, opac_vb, boxes_vb, lbls_vb],
                        outputs=[out_vb, vid_state_b],
                    )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 Segmentation Gradio Demo")
    parser.add_argument("--url", default=None, help="SAM3 service URL (overrides SAM3_URL env)")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    global SAM3_URL
    args = parse_args()
    if args.url:
        SAM3_URL = args.url
    create_demo().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        theme=gr.themes.Citrus(),
    )


if __name__ == "__main__":
    main()
