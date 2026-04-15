"""Video session state management using SAM3.1 native multiplex predictor."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException

import torch

from src.config import SHM_DIR, config
from src.helpers import decode_frames, decode_image, mask_to_detection
from src.models import get_predictor, inference_lock

_sessions_cfg = config.get("sessions", {})
SESSIONS_TTL = _sessions_cfg.get("ttl_seconds", 3600)
SESSIONS_MAX_ACTIVE = _sessions_cfg.get("max_active", 10)


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------


@dataclass
class StoredPrompt:
    """Saved prompt for re-application after session reinit."""
    frame_idx: int
    kwargs: dict  # bounding_boxes, points, point_labels, obj_id, text


@dataclass
class SessionState:
    id: str
    mode: str          # "tracker" or "video"
    temp_dir: str      # temporary directory holding frame JPEGs
    text: Optional[str] = None
    num_frames: int = 0
    prompts: list = field(default_factory=list)   # list[StoredPrompt]
    frame_results: dict = field(default_factory=dict)  # frame_idx -> list[dict]
    created_at: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0
    native_session_valid: bool = False


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def _frame_path(temp_dir: str, idx: int) -> str:
    return os.path.join(temp_dir, f"{idx:06d}.jpg")


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return state


def _reserve_session(session_id: str, state: SessionState):
    """Atomically cleanup expired, check limit, and insert new session."""
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if now - v.created_at > SESSIONS_TTL]
        for k in expired:
            _cleanup_state(_sessions.pop(k))
        if len(_sessions) >= SESSIONS_MAX_ACTIVE:
            raise HTTPException(status_code=429, detail=f"Max {SESSIONS_MAX_ACTIVE} active sessions reached")
        _sessions[session_id] = state


def _cleanup_state(state: SessionState):
    """Close native session and remove temp directory."""
    try:
        pred = get_predictor()
        pred.handle_request(dict(type="close_session", session_id=state.id))
    except Exception:
        pass
    if state.temp_dir and os.path.isdir(state.temp_dir):
        shutil.rmtree(state.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Native session management helpers
# ---------------------------------------------------------------------------


def _init_native_session(state: SessionState):
    """(Re-)start the native predictor session from temp_dir frames."""
    if state.num_frames == 0:
        return
    pred = get_predictor()
    with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
        # Close existing native session first
        try:
            pred.handle_request(dict(type="close_session", session_id=state.id))
        except Exception:
            pass

        resource = state.temp_dir if state.num_frames > 1 else _frame_path(state.temp_dir, 0)
        pred.handle_request(dict(type="start_session", resource_path=resource, session_id=state.id))

        # Re-apply stored prompts.
        # SAM3.1 resets on each add_prompt (semantic path), so merge all box prompts
        # per frame into a single call; point prompts (SAM2 path) are issued separately.
        frames_with_prompts = sorted({p.frame_idx for p in state.prompts})
        for fidx in frames_with_prompts:
            frame_prompts = [p for p in state.prompts if p.frame_idx == fidx]
            # Separate point prompts (SAM2) from box prompts (SAM3)
            point_prompts = [p for p in frame_prompts if "points" in p.kwargs]
            box_prompts = [p for p in frame_prompts if "bounding_boxes" in p.kwargs]

            # Issue box prompts merged
            if box_prompts:
                all_boxes = []
                for p in box_prompts:
                    all_boxes.extend(p.kwargs["bounding_boxes"])
                resp = pred.handle_request(dict(
                    type="add_prompt",
                    session_id=state.id,
                    frame_index=fidx,
                    bounding_boxes=torch.tensor(all_boxes, dtype=torch.float32),
                    bounding_box_labels=torch.tensor([1] * len(all_boxes), dtype=torch.int32),
                ))
                dets = _extract_detections(resp["outputs"], state.width, state.height)
                state.frame_results[fidx] = dets

            # Issue point prompts separately (each has its own obj_id)
            for p in point_prompts:
                kwargs = copy.deepcopy(p.kwargs)
                kwargs.pop("frame_idx", None)
                resp = pred.handle_request(dict(
                    type="add_prompt",
                    session_id=state.id,
                    frame_index=fidx,
                    points=torch.tensor(kwargs["points"], dtype=torch.float32),
                    point_labels=torch.tensor(kwargs["point_labels"], dtype=torch.int32),
                    obj_id=kwargs.get("obj_id"),
                ))
                dets = _extract_detections(resp["outputs"], state.width, state.height)
                state.frame_results[fidx] = dets


def _extract_detections(outputs: dict, width: int, height: int) -> list[dict]:
    """Convert native predictor outputs to detection dicts."""
    masks = outputs.get("out_binary_masks", [])
    probs = outputs.get("out_probs", [])
    obj_ids = outputs.get("out_obj_ids", [])
    detections = []
    for i in range(len(probs)):
        mask = masks[i].astype(bool) if i < len(masks) else None
        if mask is None or not mask.any():
            continue
        score = float(probs[i])
        area = float(mask.sum()) / mask.size
        det = mask_to_detection(mask, score, area)
        det["obj_id"] = int(obj_ids[i]) if i < len(obj_ids) else i
        detections.append(det)
    return detections


# ---------------------------------------------------------------------------
# High-level session operations
# ---------------------------------------------------------------------------


def create_session_sync(mode: str, frames_b64: list[str] | None, text: str | None) -> dict:
    """Create a new session. Returns dict with session info."""
    if mode == "video" and not text:
        raise HTTPException(status_code=400, detail="video mode requires 'text' prompt")

    session_id = uuid.uuid4().hex[:12]
    temp_dir = tempfile.mkdtemp(prefix=f"sam31_{session_id}_", dir=SHM_DIR)
    frames = decode_frames(frames_b64) if frames_b64 else []

    w = frames[0].size[0] if frames else 0
    h = frames[0].size[1] if frames else 0

    state = SessionState(
        id=session_id, mode=mode, temp_dir=temp_dir,
        text=text, num_frames=len(frames), width=w, height=h,
    )

    # Save frames to temp dir
    for idx, frame in enumerate(frames):
        frame.save(_frame_path(temp_dir, idx), format="JPEG")

    _reserve_session(session_id, state)

    if frames:
        pred = get_predictor()
        resource = temp_dir if len(frames) > 1 else _frame_path(temp_dir, 0)
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            pred.handle_request(dict(type="start_session", resource_path=resource, session_id=session_id))
            if mode == "video" and text:
                pred.handle_request(dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text,
                ))
        state.native_session_valid = True

    return {
        "session_id": session_id, "mode": mode,
        "num_frames": len(frames), "width": w, "height": h,
    }


def add_frame_sync(session_id: str, frame_b64: str) -> dict:
    """Add a frame to a streaming tracker session."""
    state = get_session(session_id)
    if state.mode == "video":
        raise HTTPException(status_code=400, detail="Video sessions don't support streaming frames")

    frame = decode_image(frame_b64)
    frame_idx = state.num_frames
    frame.save(_frame_path(state.temp_dir, frame_idx), format="JPEG")
    state.num_frames += 1
    if not state.width:
        state.width, state.height = frame.size

    # Mark native session as invalid — will reinit lazily before next inference
    state.native_session_valid = False

    # Return detections for this new frame (no prompts yet → empty)
    dets = state.frame_results.get(frame_idx, [])
    return {"frame_idx": frame_idx, "detections": dets}


def add_prompts_sync(
    session_id: str, frame_idx: int, obj_ids: list[int],
    points=None, labels=None, boxes=None, masks=None,
) -> dict:
    """Add interactive prompts to a tracker session."""
    state = get_session(session_id)
    if state.mode != "tracker":
        raise HTTPException(status_code=400, detail="Prompts only supported for tracker sessions")
    if state.num_frames == 0:
        raise HTTPException(status_code=400, detail="No frames in session — add frames first")
    if frame_idx >= state.num_frames:
        raise HTTPException(
            status_code=400,
            detail=f"frame_idx {frame_idx} out of range (have {state.num_frames} frames)",
        )

    # Build this prompt's kwargs
    new_kwargs: dict = {}
    if boxes is not None:
        # boxes from API may be HF format [[[x1,y1,x2,y2]], ...] or flat [[x1,y1,x2,y2], ...]
        flat = []
        for b in boxes:
            inner = b[0] if isinstance(b[0], (list, tuple)) else b
            flat.append(inner)
        norm = [
            [c[0] / state.width, c[1] / state.height,
             (c[2] - c[0]) / state.width, (c[3] - c[1]) / state.height]
            for c in flat
        ]
        new_kwargs["bounding_boxes"] = norm
        new_kwargs["bounding_box_labels"] = [1] * len(norm)
    if points is not None:
        # points from API may be HF format [[[[x,y]]]] or flat [[x,y], ...]
        flat_pts = []
        for p in points:
            while isinstance(p, (list, tuple)) and len(p) == 1 and isinstance(p[0], (list, tuple)):
                p = p[0]
            flat_pts.append(p)
        new_kwargs["points"] = [[p[0] / state.width, p[1] / state.height] for p in flat_pts]
        if labels is not None:
            flat_labels = []
            for lbl in labels:
                while isinstance(lbl, (list, tuple)):
                    lbl = lbl[0]
                flat_labels.append(int(lbl))
            new_kwargs["point_labels"] = flat_labels
        else:
            new_kwargs["point_labels"] = [1] * len(flat_pts)
    if masks is not None:
        new_kwargs["masks"] = masks

    if not new_kwargs:
        raise HTTPException(status_code=400, detail="At least one of points/boxes/masks required")

    # Store the prompt (SAM2-style point prompts keep per-object obj_id)
    new_kwargs["obj_id"] = obj_ids[0] if obj_ids else 1
    new_kwargs["frame_idx"] = frame_idx
    stored = StoredPrompt(frame_idx=frame_idx, kwargs=copy.deepcopy(new_kwargs))
    state.prompts.append(stored)

    pred = get_predictor()

    # Ensure native session is initialized with current frames before adding prompts
    if not state.native_session_valid:
        _init_native_session(state)
        state.native_session_valid = True

    # SAM3.1 native add_prompt resets state on each call (semantic prompt model).
    # For multi-object box tracking, combine ALL accumulated box prompts on this frame
    # into one add_prompt call so the model sees all objects simultaneously.
    is_point_prompt = "points" in new_kwargs
    if is_point_prompt:
        # SAM2-style: each point prompt keeps its obj_id and is issued separately
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            resp = pred.handle_request(dict(
                type="add_prompt",
                session_id=state.id,
                frame_index=frame_idx,
                points=torch.tensor(new_kwargs["points"], dtype=torch.float32),
                point_labels=torch.tensor(new_kwargs["point_labels"], dtype=torch.int32),
                obj_id=new_kwargs.get("obj_id"),
            ))
    else:
        # SAM3-style box/text: merge all box prompts for this frame into one call
        all_boxes_for_frame = []
        for p in state.prompts:
            if p.frame_idx == frame_idx and "bounding_boxes" in p.kwargs:
                all_boxes_for_frame.extend(p.kwargs["bounding_boxes"])
        with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
            resp = pred.handle_request(dict(
                type="add_prompt",
                session_id=state.id,
                frame_index=frame_idx,
                bounding_boxes=torch.tensor(all_boxes_for_frame, dtype=torch.float32),
                bounding_box_labels=torch.tensor([1] * len(all_boxes_for_frame), dtype=torch.int32),
            ))

    dets = _extract_detections(resp["outputs"], state.width, state.height)
    state.frame_results[frame_idx] = dets

    return {"frame_idx": frame_idx, "detections": dets}


def propagate_sync(session_id: str, max_frames: Optional[int] = None) -> list[dict]:
    """Propagate tracked objects through all frames."""
    state = get_session(session_id)
    if state.num_frames == 0:
        raise HTTPException(status_code=400, detail="No frames in session")

    pred = get_predictor()

    # Ensure native session is initialized
    if not state.native_session_valid:
        _init_native_session(state)
        state.native_session_valid = True

    # SAM3.1: if only point prompts (SAM2-style) were added, previous_stages_out is all None.
    # Passing start_frame_index=0 bypasses the "No prompts received" check.
    has_only_point_prompts = (
        state.prompts
        and all("points" in p.kwargs for p in state.prompts)
        and all("bounding_boxes" not in p.kwargs for p in state.prompts)
        and state.mode == "tracker"
    )

    propagate_req: dict = dict(type="propagate_in_video", session_id=session_id)
    if max_frames is not None:
        propagate_req["max_frame_num_to_track"] = max_frames
    if has_only_point_prompts:
        propagate_req["start_frame_index"] = 0

    results = []
    with inference_lock, torch.autocast("cuda", dtype=torch.bfloat16):
        gen = list(pred.handle_stream_request(propagate_req))
    for item in gen:
        fidx = item.get("frame_index", item.get("frame_idx", 0))
        outputs = item.get("outputs", {})
        # Use propagation outputs; fall back to stored prompt results for the prompted frame
        dets = _extract_detections(outputs, state.width, state.height)
        if not dets and fidx in state.frame_results:
            dets = state.frame_results[fidx]
        results.append({"frame_idx": fidx, "detections": dets})

    return results


def delete_session_sync(session_id: str) -> dict:
    """Delete a session and free GPU memory."""
    with _sessions_lock:
        state = _sessions.pop(session_id, None)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    _cleanup_state(state)
    return {"deleted": True}


def get_sessions_info() -> dict:
    """Return active session count for health endpoint."""
    return {"active": len(_sessions), "max": SESSIONS_MAX_ACTIVE}
