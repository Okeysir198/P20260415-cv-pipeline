"""Tests for video session endpoints: POST/DELETE /video/sessions, POST frames."""

from __future__ import annotations

import base64
import io
import json

import cv2
import numpy as np
import requests
import supervision as sv
from PIL import Image

from conftest import DATA_DIR, OUTPUT_DIR, SERVICE_URL, skip_no_service, write_vscode_video

NUM_FRAMES = 5
FRAME_SIZE = (640, 480)


def _extract_frames(video_path, n: int) -> tuple[list[str], list[np.ndarray]]:
    """Extract n evenly-spaced frames from video. Returns (base64 list, BGR numpy list)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        cap.release()
        raise ValueError(f"Video has only {total} frames, need at least {n}")

    indices = [int(i * total / n) for i in range(n)]
    frames_b64 = []
    frames_bgr = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, FRAME_SIZE)
        frames_bgr.append(frame_resized)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frames_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    cap.release()
    return frames_b64, frames_bgr


def _detections_to_sv_video(detections: list[dict], img_w: int, img_h: int) -> sv.Detections:
    """Convert video frame detections (with obj_id) to sv.Detections."""
    if not detections:
        return sv.Detections.empty()

    xyxy, scores, class_ids, tracker_ids = [], [], [], []
    for det in detections:
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        xyxy.append(bbox)
        scores.append(det.get("score", 0.0))
        class_ids.append(max(0, det.get("class_id", 0)))
        tracker_ids.append(max(0, det.get("obj_id", 0)))

    if not xyxy:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(scores, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


@skip_no_service
class TestVideoSessions:
    def test_create_add_frames_and_close(self):
        """Full video session lifecycle with text prompt: create → propagate → close."""
        video_path = DATA_DIR / "indoor_fire.mp4"
        if not video_path.exists():
            import pytest
            pytest.skip("Video file not found")

        frames_b64, frames_bgr = _extract_frames(video_path, NUM_FRAMES)

        # POST /video/sessions — create with text prompt and all frames upfront
        resp = requests.post(
            f"{SERVICE_URL}/video/sessions",
            json={"mode": "video", "text": "fire", "classes": {"0": "fire"}, "frames": frames_b64},
            timeout=60,
        )
        assert resp.status_code == 200
        session = resp.json()
        assert "session_id" in session
        assert "sam3_session_id" in session
        assert session["mode"] == "video"
        session_id = session["session_id"]

        # POST /video/sessions/{id}/propagate — get all frame results
        resp = requests.post(
            f"{SERVICE_URL}/video/sessions/{session_id}/propagate",
            timeout=60,
        )
        assert resp.status_code == 200
        propagate_result = resp.json()
        assert "frames" in propagate_result
        frame_results = [f["detections"] for f in propagate_result["frames"]]

        # DELETE /video/sessions/{id} — close
        resp = requests.delete(f"{SERVICE_URL}/video/sessions/{session_id}", timeout=30)
        assert resp.status_code == 200
        close = resp.json()
        assert close["deleted"] is True
        assert close["session_id"] == session_id

        # Save outputs
        json_path = OUTPUT_DIR / "test03_video_session.json"
        with open(json_path, "w") as f:
            json.dump({"session_id": session_id, "frames": frame_results}, f, indent=2)

        # Save overlay frames + video
        class_names = {0: "fire"}
        annotated_frames = []
        for i, (bgr_frame, dets) in enumerate(zip(frames_bgr, frame_results)):
            img_h, img_w = bgr_frame.shape[:2]
            sv_dets = _detections_to_sv_video(dets, img_w, img_h)

            labels = []
            for j in range(len(sv_dets)):
                tid = int(sv_dets.tracker_id[j]) if sv_dets.tracker_id is not None else -1
                labels.append(f"fire #{tid}")

            annotated = bgr_frame.copy()
            box_ann = sv.BoxAnnotator(thickness=3, color_lookup=sv.ColorLookup.CLASS)
            label_ann = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, text_padding=6, color_lookup=sv.ColorLookup.CLASS)
            annotated = box_ann.annotate(scene=annotated, detections=sv_dets)
            annotated = label_ann.annotate(scene=annotated, detections=sv_dets, labels=labels)
            cv2.putText(annotated, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            annotated_frames.append(annotated)
            cv2.imwrite(str(OUTPUT_DIR / f"test03_video_frame_{i}.png"), annotated)

        # Write annotated video (H.264 for VS Code compatibility)
        video_out = OUTPUT_DIR / "test03_video_overlay.mp4"
        write_vscode_video(annotated_frames, video_out, fps=1.0)

    def test_delete_nonexistent_session(self):
        """DELETE /video/sessions/{id} returns 404 for unknown session."""
        resp = requests.delete(f"{SERVICE_URL}/video/sessions/nonexistent_123", timeout=10)
        assert resp.status_code == 404

    def test_add_frame_nonexistent_session(self):
        """POST frames to nonexistent session returns 404."""
        resp = requests.post(
            f"{SERVICE_URL}/video/sessions/nonexistent_456/frames",
            json={"frame": "dGVzdA=="},
            timeout=10,
        )
        assert resp.status_code == 404
