"""Tests for tracker video session lifecycle (create, prompts, frames, propagate, delete)."""

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
class TestTrackerSession:
    def test_create_tracker_session(self):
        """Create a tracker session with frames."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["mode"] == "tracker"
        assert data["num_frames"] == 3
        assert data["width"] > 0
        assert data["height"] > 0
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{data['session_id']}", timeout=10)

    def test_add_prompts(self):
        """Add box prompt to tracker session returns detections."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_idx"] == 0
        assert "detections" in data
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_propagate(self):
        """Propagate tracked objects through all frames."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=5)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "frames" in data
        assert len(data["frames"]) > 0
        for fr in data["frames"]:
            assert "frame_idx" in fr
            assert "detections" in fr
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_delete_session(self):
        """Delete session returns success, second delete returns 404."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=2)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        resp = requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp = requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)
        assert resp.status_code == 404

    def test_streaming_add_frame(self):
        """Create empty session, add frames one by one."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        assert resp.json()["num_frames"] == 0

        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        for i, frame_b64 in enumerate(frames_b64):
            resp = requests.post(
                f"{SERVICE_URL}/sessions/{sid}/frames",
                json={"frame": frame_b64},
                timeout=REQUEST_TIMEOUT,
            )
            assert resp.status_code == 200
            assert resp.json()["frame_idx"] == i
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_saves_propagation_video(self):
        """Full tracker lifecycle — annotate all frames and write output .mp4."""
        frames_bgr, frames_b64, video_info = load_all_video_frames("bedroom.mp4")

        # Create session with all frames
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        # Prompt box on frame 0
        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )

        # Propagate
        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        prop_result = resp.json()
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

        # Write annotated video (H.264 for VS Code compatibility)
        out_path = OUTPUT_DIR / "test04_tracker_session.mp4"
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
