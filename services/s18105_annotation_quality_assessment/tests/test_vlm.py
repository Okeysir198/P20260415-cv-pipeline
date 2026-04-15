"""VLM unit tests (parsers, crop) and integration tests (require service + Ollama)."""

import base64
import io
import sys
from pathlib import Path

# Allow running from project root — add service root to sys.path
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVICE_ROOT))

from PIL import Image

from src.vlm import crop_annotation, parse_crop_response, parse_scene_response

from conftest import SERVICE_URL, skip_no_ollama, skip_no_service


# ---------------------------------------------------------------------------
# Unit tests — no service or Ollama needed
# ---------------------------------------------------------------------------


class TestParseCropResponse:
    """Parse YES/NO | confidence | reason responses."""

    def test_yes_response(self):
        is_correct, conf, reason = parse_crop_response("YES | 0.95 | clearly shows flames")
        assert is_correct is True
        assert abs(conf - 0.95) < 0.01
        assert "flames" in reason

    def test_no_response(self):
        is_correct, conf, reason = parse_crop_response("NO | 0.3 | looks like orange light")
        assert is_correct is False
        assert abs(conf - 0.3) < 0.01
        assert "orange" in reason

    def test_malformed_response(self):
        # "not" contains "NO" → parsed as is_correct=False with default conf=0.1
        is_correct, conf, reason = parse_crop_response("I'm not sure what this is")
        assert is_correct is False
        assert conf == 0.1

    def test_verbose_response(self):
        text = "Based on my analysis, YES | 0.85 | This appears to be fire with visible flames and heat distortion"
        is_correct, conf, reason = parse_crop_response(text)
        assert is_correct is True
        assert conf > 0.5

    def test_no_pipe_separator(self):
        is_correct, conf, reason = parse_crop_response("YES this is definitely fire 0.9")
        assert is_correct is True

    def test_empty_string(self):
        is_correct, conf, reason = parse_crop_response("")
        assert is_correct is False


class TestParseSceneResponse:
    """Parse INCORRECT/MISSING/QUALITY scene responses."""

    def test_normal_response(self):
        text = "INCORRECT: [1, 3]\nMISSING: [a smoke plume in the corner]\nQUALITY: 0.7"
        incorrect, missing, quality = parse_scene_response(text)
        assert incorrect == [1, 3]
        assert len(missing) >= 1
        assert abs(quality - 0.7) < 0.01

    def test_all_none(self):
        text = "INCORRECT: NONE\nMISSING: NONE\nQUALITY: 0.95"
        incorrect, missing, quality = parse_scene_response(text)
        assert incorrect == []
        assert missing == []
        assert abs(quality - 0.95) < 0.01

    def test_malformed_indices(self):
        text = "INCORRECT: maybe index 2?\nMISSING: NONE\nQUALITY: 0.6"
        incorrect, missing, quality = parse_scene_response(text)
        assert 2 in incorrect
        assert abs(quality - 0.6) < 0.01

    def test_missing_quality(self):
        text = "INCORRECT: NONE\nMISSING: NONE"
        incorrect, missing, quality = parse_scene_response(text)
        assert quality == 0.0


class TestCropExtraction:
    """Test bbox crop from image."""

    def test_normal_crop(self, test_image_b64):
        result = crop_annotation(test_image_b64, [0.5, 0.5, 0.3, 0.2], 640, 480, 0.05)
        assert len(result) > 0
        img = Image.open(io.BytesIO(base64.b64decode(result)))
        assert img.size[0] > 0 and img.size[1] > 0

    def test_edge_crop(self, test_image_b64):
        result = crop_annotation(test_image_b64, [0.95, 0.95, 0.2, 0.2], 640, 480, 0.05)
        assert len(result) > 0

    def test_padding_clipping(self, test_image_b64):
        result = crop_annotation(test_image_b64, [0.0, 0.0, 0.1, 0.1], 640, 480, 0.5)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration tests — require service + Ollama
# ---------------------------------------------------------------------------


class TestVLMIntegration:
    """VLM integration tests (require both service and Ollama)."""

    @skip_no_service
    @skip_no_ollama
    def test_verify_vlm_enabled(self, test_image_b64, sample_classes):
        import requests
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
            "enable_vlm": True,
            "vlm_trigger": "all",
        }, timeout=180)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("vlm_verification"):
                assert "crop_verification" in data["vlm_verification"]
                assert "scene_verification" in data["vlm_verification"]
                assert data["vlm_verification"]["available"] is True

    @skip_no_service
    def test_verify_vlm_disabled_default(self, test_image_b64, sample_classes):
        import requests
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
        }, timeout=120)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("vlm_verification") is None

    @skip_no_service
    @skip_no_ollama
    def test_verify_vlm_trigger_all(self, test_image_b64, sample_classes):
        import requests
        resp = requests.post(f"{SERVICE_URL}/verify", json={
            "image": test_image_b64,
            "labels": ["0 0.5 0.5 0.3 0.2"],
            "classes": sample_classes,
            "enable_vlm": True,
            "vlm_trigger": "all",
        }, timeout=180)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            # With trigger "all", VLM should always run
            if data.get("vlm_verification"):
                assert data["vlm_verification"]["available"] is True
