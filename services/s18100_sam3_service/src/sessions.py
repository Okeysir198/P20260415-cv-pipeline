"""Video session state management and operations."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch
from fastapi import HTTPException

from src.config import config
from src.helpers import decode_frames, decode_image, mask_post_kwargs, mask_to_detection
from src.models import device, dtype, get_tracker, get_video

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
    obj_ids: list
    kwargs: dict  # input_boxes, input_points, input_labels, input_masks


@dataclass
class SessionState:
    id: str
    mode: str  # "tracker" or "video"
    frames: list  # list of PIL Images
    text: Optional[str] = None
    inference_session: Optional[object] = None
    prompts: list = field(default_factory=list)  # list of StoredPrompt
    created_at: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return state


def cleanup_expired():
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if now - v.created_at > SESSIONS_TTL]
        for k in expired:
            del _sessions[k]


def check_session_limit():
    with _sessions_lock:
        if len(_sessions) >= SESSIONS_MAX_ACTIVE:
            raise HTTPException(status_code=429, detail=f"Max {SESSIONS_MAX_ACTIVE} active sessions reached")


def _reserve_session(session_id: str, state: SessionState):
    """Atomically cleanup expired, check limit, and insert a new session."""
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if now - v.created_at > SESSIONS_TTL]
        for k in expired:
            del _sessions[k]
        if len(_sessions) >= SESSIONS_MAX_ACTIVE:
            raise HTTPException(status_code=429, detail=f"Max {SESSIONS_MAX_ACTIVE} active sessions reached")
        _sessions[session_id] = state


# ---------------------------------------------------------------------------
# Session init helpers
# ---------------------------------------------------------------------------


def init_tracker_session(state: SessionState):
    """(Re-)initialize the tracker inference session from stored frames."""
    if not state.frames:
        return
    _, processor = get_tracker()
    state.inference_session = processor.init_video_session(
        video=state.frames, inference_device=device(), dtype=dtype(),
    )
    for p in state.prompts:
        processor.add_inputs_to_inference_session(
            inference_session=state.inference_session,
            frame_idx=p.frame_idx, obj_ids=list(p.obj_ids), **copy.deepcopy(p.kwargs),
        )


def init_video_session(state: SessionState):
    """(Re-)initialize the video inference session from stored frames + text."""
    if not state.frames or not state.text:
        return
    _, processor = get_video()
    state.inference_session = processor.init_video_session(
        video=state.frames, inference_device=device(), dtype=dtype(),
    )
    processor.add_text_prompt(inference_session=state.inference_session, text=state.text)


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------


def run_tracker_on_frame(state: SessionState, frame_idx: int) -> list[dict]:
    """Run tracker model on a single frame."""
    if state.inference_session is None or not state.prompts:
        return []
    model, processor = get_tracker()
    with torch.no_grad():
        outputs = model(inference_session=state.inference_session, frame_idx=frame_idx)
        obj_ids = outputs.object_ids if outputs.object_ids is not None else []
        if not obj_ids:
            return []
        masks = processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[[state.height, state.width]],
            **mask_post_kwargs(),
        )[0]
        dets = []
        iou_scores = outputs.object_score_logits.cpu() if outputs.object_score_logits is not None else None
        for i, oid in enumerate(obj_ids):
            if i >= masks.shape[0]:
                break
            m = masks[i].squeeze()
            mask_np = m.cpu().numpy().astype(bool) if isinstance(m, torch.Tensor) else m.astype(bool)
            score = float(torch.sigmoid(iou_scores[i]).item()) if iou_scores is not None and i < len(iou_scores) else 1.0
            det = mask_to_detection(mask_np, score)
            det["obj_id"] = oid
            dets.append(det)
    return dets


def propagate_tracker(state: SessionState, max_frames: Optional[int] = None) -> list[dict]:
    """Propagate tracker through all frames."""
    if state.inference_session is None:
        return []
    model, processor = get_tracker()
    results = []
    with torch.no_grad():
        for output in model.propagate_in_video_iterator(
            inference_session=state.inference_session,
            max_frame_num_to_track=max_frames,
        ):
            obj_ids = output.object_ids if output.object_ids is not None else []
            if not obj_ids:
                results.append({"frame_idx": output.frame_idx, "detections": []})
                continue
            masks_post = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[state.height, state.width]],
                **mask_post_kwargs(),
            )[0]
            dets = []
            for i, oid in enumerate(obj_ids):
                if i >= masks_post.shape[0]:
                    break
                m = masks_post[i].squeeze()
                mask_np = m.cpu().numpy().astype(bool) if isinstance(m, torch.Tensor) else m.astype(bool)
                score_logit = output.object_score_logits[i] if output.object_score_logits is not None and i < len(output.object_score_logits) else None
                score = float(torch.sigmoid(score_logit).item()) if score_logit is not None else 1.0
                dets.append({**mask_to_detection(mask_np, score), "obj_id": oid})
            results.append({"frame_idx": output.frame_idx, "detections": dets})
    return results


def propagate_video(state: SessionState, max_frames: Optional[int] = None) -> list[dict]:
    """Propagate video model (text-driven) through all frames."""
    if state.inference_session is None:
        return []
    model, processor = get_video()
    results = []
    with torch.no_grad():
        for output in model.propagate_in_video_iterator(
            inference_session=state.inference_session,
            max_frame_num_to_track=max_frames,
        ):
            post = processor.postprocess_outputs(
                inference_session=state.inference_session,
                model_outputs=output,
                original_sizes=[[state.height, state.width]],
            )
            obj_ids = post["object_ids"].tolist() if hasattr(post.get("object_ids", None), "tolist") else []
            masks = post.get("masks")
            scores = post.get("scores")
            dets = []
            if masks is not None:
                for i, oid in enumerate(obj_ids):
                    if i >= masks.shape[0]:
                        break
                    mask_np = masks[i].cpu().numpy().astype(bool) if isinstance(masks[i], torch.Tensor) else masks[i].astype(bool)
                    score = float(scores[i].item()) if scores is not None and i < len(scores) else 1.0
                    dets.append({**mask_to_detection(mask_np, score), "obj_id": oid})
            results.append({"frame_idx": output.frame_idx, "detections": dets})
    return results


# ---------------------------------------------------------------------------
# High-level session operations (used by routes)
# ---------------------------------------------------------------------------


def create_session_sync(mode: str, frames_b64: list[str] | None, text: str | None) -> dict:
    """Create a new session. Returns dict with session info."""
    if mode == "video" and not text:
        raise HTTPException(status_code=400, detail="video mode requires 'text' prompt")

    session_id = uuid.uuid4().hex[:12]
    frames = decode_frames(frames_b64) if frames_b64 else []
    w = frames[0].size[0] if frames else 0
    h = frames[0].size[1] if frames else 0

    state = SessionState(
        id=session_id, mode=mode, frames=frames,
        text=text, width=w, height=h,
    )

    # Reserve slot atomically (cleanup + limit check + insert under one lock)
    _reserve_session(session_id, state)

    # Init inference session outside lock (GPU work)
    if frames:
        if mode == "tracker":
            init_tracker_session(state)
        else:
            init_video_session(state)

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
    state.frames.append(frame)
    if not state.width:
        state.width, state.height = frame.size

    init_tracker_session(state)

    frame_idx = len(state.frames) - 1
    dets = run_tracker_on_frame(state, frame_idx)
    return {"frame_idx": frame_idx, "detections": dets}


def add_prompts_sync(session_id: str, frame_idx: int, obj_ids: list[int],
                     points=None, labels=None, boxes=None, masks=None) -> dict:
    """Add interactive prompts to a tracker session."""
    state = get_session(session_id)
    if state.mode != "tracker":
        raise HTTPException(status_code=400, detail="Prompts only supported for tracker sessions")
    if not state.frames:
        raise HTTPException(status_code=400, detail="No frames in session — add frames first")
    if frame_idx >= len(state.frames):
        raise HTTPException(status_code=400, detail=f"frame_idx {frame_idx} out of range (have {len(state.frames)} frames)")

    kwargs = {}
    if boxes is not None:
        # HF SAM3 tracker processor requires 3-level nesting: [image, box, coords]
        # The client sends [[x1,y1,x2,y2], ...] (2 levels) — wrap to 3 levels.
        if boxes and not isinstance(boxes[0][0], (list, tuple)):
            boxes = [boxes]
        kwargs["input_boxes"] = boxes
    if points is not None:
        kwargs["input_points"] = points
    if labels is not None:
        kwargs["input_labels"] = labels
    if masks is not None:
        kwargs["input_masks"] = masks

    if not kwargs:
        raise HTTPException(status_code=400, detail="At least one of points/boxes/masks required")

    stored = StoredPrompt(frame_idx=frame_idx, obj_ids=list(obj_ids), kwargs=copy.deepcopy(kwargs))
    state.prompts.append(stored)

    if state.inference_session is None:
        init_tracker_session(state)
    else:
        _, processor = get_tracker()
        processor.add_inputs_to_inference_session(
            inference_session=state.inference_session,
            frame_idx=frame_idx, obj_ids=list(obj_ids), **copy.deepcopy(kwargs),
        )

    dets = run_tracker_on_frame(state, frame_idx)
    return {"frame_idx": frame_idx, "detections": dets}


def propagate_sync(session_id: str, max_frames: Optional[int] = None) -> list[dict]:
    """Propagate tracked objects through all frames."""
    state = get_session(session_id)
    if not state.frames:
        raise HTTPException(status_code=400, detail="No frames in session")

    if state.inference_session is None:
        if state.mode == "tracker":
            init_tracker_session(state)
        else:
            init_video_session(state)

    if state.inference_session is None:
        raise HTTPException(status_code=400, detail="Could not initialize inference session")

    if state.mode == "tracker":
        return propagate_tracker(state, max_frames)
    else:
        return propagate_video(state, max_frames)


def delete_session_sync(session_id: str) -> dict:
    """Delete a session and free GPU memory."""
    with _sessions_lock:
        state = _sessions.pop(session_id, None)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    state.inference_session = None
    state.frames.clear()
    state.prompts.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"deleted": True}


def get_sessions_info() -> dict:
    """Return active session count for health endpoint."""
    return {
        "active": len(_sessions),
        "max": SESSIONS_MAX_ACTIVE,
    }
