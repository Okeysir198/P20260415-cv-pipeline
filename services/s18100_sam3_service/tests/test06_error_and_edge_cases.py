"""Tests for error handling, point prompts, multi-object tracking, and edge cases."""

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
    load_image_b64,
    load_image_cv2,
    load_video_frames_b64,
    skip_no_service,
    write_vscode_video,
)


@skip_no_service
class TestErrorCases:
    def test_segment_box_invalid_base64(self):
        """Invalid base64 image data returns 422 or 500."""
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": "not-valid-base64!!!", "box": [0, 0, 100, 100]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code in (400, 422, 500)

    def test_segment_box_missing_box(self):
        """Missing box field returns 422."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_segment_text_missing_text(self):
        """Missing text field returns 422."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_session_invalid_mode(self):
        """Invalid session mode returns 422 (Pydantic Literal validation)."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "invalid"},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_session_video_no_text(self):
        """Video mode without text returns 400."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=2)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "video", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400

    def test_session_not_found(self):
        """DELETE on a nonexistent session returns 404."""
        resp = requests.delete(f"{SERVICE_URL}/sessions/nonexistent_id", timeout=10)
        assert resp.status_code == 404

    def test_add_frame_nonexistent_session(self):
        """POST frame to nonexistent session returns 404."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/sessions/nonexistent_id/frames",
            json={"frame": image_b64},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 404

    def test_add_prompts_nonexistent_session(self):
        """POST prompts to nonexistent session returns 404."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions/nonexistent_id/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 404

    def test_propagate_nonexistent_session(self):
        """POST propagate on nonexistent session returns 404."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions/nonexistent_id/propagate",
            json={},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 404


@skip_no_service
class TestPointPrompts:
    def test_point_prompt_single_object(self):
        """Point prompt on tracker session returns detections."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "points": [[[[500, 375]]]], "labels": [[[1]]]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_idx"] == 0
        assert "detections" in data
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_point_prompt_propagate(self):
        """Point prompt on tracker session propagates through frames."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "points": [[[[500, 375]]]], "labels": [[[1]]]},
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

    def test_saves_point_prompt_video(self):
        """Full point prompt lifecycle — annotate all frames and write output .mp4."""
        frames_bgr, frames_b64, video_info = load_all_video_frames("bedroom.mp4")

        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "points": [[[[500, 375]]]], "labels": [[[1]]]},
            timeout=REQUEST_TIMEOUT,
        )

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        prop_result = resp.json()
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

        out_path = OUTPUT_DIR / "test06_point_prompt.mp4"
        annotated_frames = []
        for fr in prop_result["frames"]:
            idx = fr["frame_idx"]
            if idx >= len(frames_bgr):
                break
            img = frames_bgr[idx].copy()
            if fr["detections"]:
                dets = detections_from_masks(fr["detections"])
                labels = [f"obj {d.get('obj_id', '?')} (point)" for d in fr["detections"]]
                img = annotate_image(img, dets, labels)
            annotated_frames.append(img)
        write_vscode_video(annotated_frames, out_path, fps=video_info.fps)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


@skip_no_service
class TestMultiObject:
    def test_multi_object_box_prompts(self):
        """Two separate box prompts return detections with distinct obj_ids."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        # Add first object
        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )
        # Add second object
        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [2], "boxes": [[[50, 50, 300, 200]]]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        obj_ids = {d.get("obj_id") for d in data["detections"]}
        assert len(obj_ids) > 1
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_multi_object_propagate(self):
        """Propagating two box prompts yields multiple obj_ids across frames."""
        frames_b64 = load_video_frames_b64("bedroom.mp4", num_frames=3)
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker", "frames": frames_b64},
            timeout=REQUEST_TIMEOUT,
        )
        sid = resp.json()["session_id"]

        # Add two objects separately
        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )
        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [2], "boxes": [[[50, 50, 300, 200]]]},
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
        all_obj_ids = {
            d.get("obj_id")
            for fr in data["frames"]
            for d in fr["detections"]
        }
        assert len(all_obj_ids) > 1
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_saves_multi_object_video(self):
        """Full multi-object lifecycle — annotate all frames and write output .mp4."""
        frames_bgr, frames_b64, video_info = load_all_video_frames("bedroom.mp4")

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
        requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [2], "boxes": [[[50, 50, 300, 200]]]},
            timeout=REQUEST_TIMEOUT,
        )

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT * 2,
        )
        prop_result = resp.json()
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

        out_path = OUTPUT_DIR / "test06_multi_object.mp4"
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


@skip_no_service
class TestEdgeCases:
    def test_segment_text_no_match(self):
        """Text with no visual match returns 200 with empty or near-empty detections."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_text",
            json={"image": image_b64, "text": "purple dinosaur wearing a hat"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_segment_box_full_image(self):
        """Box covering most of the image returns a valid result and saves overlay."""
        image_b64 = load_image_b64("truck.jpg")
        resp = requests.post(
            f"{SERVICE_URL}/segment_box",
            json={"image": image_b64, "box": [0, 0, 900, 600]},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert "mask" in result
        assert "bbox" in result

        img = load_image_cv2("truck.jpg")
        dets = detections_from_masks([result])
        labels = [f"full-image box score={result['score']:.3f}"]
        annotated = annotate_image(img, dets, labels)
        out_path = OUTPUT_DIR / "test06_full_image_box.png"
        cv2.imwrite(str(out_path), annotated)
        assert out_path.exists()

    def test_empty_tracker_session_propagate(self):
        """Propagate on a tracker session with no frames returns 400."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/propagate",
            json={},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)

    def test_prompts_no_frames(self):
        """Adding prompts to a session with no frames returns 400."""
        resp = requests.post(
            f"{SERVICE_URL}/sessions",
            json={"mode": "tracker"},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]

        resp = requests.post(
            f"{SERVICE_URL}/sessions/{sid}/prompts",
            json={"frame_idx": 0, "obj_ids": [1], "boxes": [[[350, 280, 550, 470]]]},  # person on bed (tighter 200x190)
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 400
        # Cleanup
        requests.delete(f"{SERVICE_URL}/sessions/{sid}", timeout=10)
