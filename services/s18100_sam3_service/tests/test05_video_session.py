"""Tests for text-driven video session lifecycle (create, propagate, delete)."""

from __future__ import annotations

import cv2
import requests
import supervision as sv

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    annotate_image,
    detections_from_masks,
    load_all_video_frames,
    load_video_frames_b64,
    skip_no_service,
    write_vscode_video,
)


@skip_no_service
class TestVideoSession:
    def test_create_video_session(self):
        """Create a video session with text prompt."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "video", "frames": frames_b64, "text": "bed"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "video"
        assert data["num_frames"] == 3
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{data['session_id']}", timeout=10)

    def test_video_requires_text(self):
        """Video mode without text returns 400."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=2)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "video", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400

    def test_propagate_video(self):
        """Propagate text-driven video detection."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=5)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "video", "frames": frames_b64, "text": "bed"},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "frames" in data
        assert len(data["frames"]) > 0
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_saves_propagation_video(self):
        """Full video lifecycle — annotate all frames and write output .mp4."""
        frames_bgr, frames_b64, video_info = load_all_video_frames("bedroom.mp4")

        # Create session with all frames + text
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "video", "frames": frames_b64, "text": "bed"},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        # Propagate
        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        prop_result = resp.json()
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

        # Write annotated video (H.264 for VS Code compatibility)
        out_path = OUTPUT_DIR / "test05_video_session.mp4"
        annotated_frames = []
        for fr in prop_result["frames"]:
            idx = fr["frame_idx"]
            if idx >= len(frames_bgr):
                break
            img = frames_bgr[idx].copy()
            if fr["detections"]:
                dets = detections_from_masks(fr["detections"])
                labels = [f"obj {d.get('obj_id', '?')}" for d in fr["detections"]]
                img = annotate_image(img, dets, labels)
            annotated_frames.append(img)
        write_vscode_video(annotated_frames, out_path, fps=video_info.fps)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
