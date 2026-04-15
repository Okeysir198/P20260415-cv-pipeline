"""Flux NIM service tests — POST /v1/infer (text-to-image)."""

import matplotlib.pyplot as plt
import requests

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    base64_to_image,
    skip_no_service,
)


@skip_no_service
class TestText2Img:
    def test_basic_generation(self):
        """Text-to-image returns an artifact with base64 image."""
        prompt = "a red fire truck parked in front of a building, photorealistic"
        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={"prompt": prompt, "seed": 42, "steps": 4},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "artifacts" in data
        assert len(data["artifacts"]) > 0
        artifact = data["artifacts"][0]
        assert "base64" in artifact
        assert artifact.get("finishReason") == "SUCCESS"

    def test_image_dimensions(self):
        """Generated image has expected dimensions (1024x1024)."""
        prompt = "a simple blue sky with white clouds"
        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={"prompt": prompt, "seed": 0, "steps": 4},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        image = base64_to_image(resp.json()["artifacts"][0]["base64"])
        assert image.size[0] == 1024
        assert image.size[1] == 1024

    def test_seed_determinism(self):
        """Same seed + prompt produces same image."""
        prompt = "a green tree in a park"
        payload = {"prompt": prompt, "seed": 123, "steps": 4}
        resp1 = requests.post(f"{SERVICE_URL}/v1/infer", json=payload, timeout=REQUEST_TIMEOUT)
        resp2 = requests.post(f"{SERVICE_URL}/v1/infer", json=payload, timeout=REQUEST_TIMEOUT)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        b64_1 = resp1.json()["artifacts"][0]["base64"]
        b64_2 = resp2.json()["artifacts"][0]["base64"]
        assert b64_1 == b64_2, "Same seed should produce identical output"

    def test_different_seeds(self):
        """Different seeds produce different images."""
        prompt = "a mountain landscape at dawn"
        resp1 = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={"prompt": prompt, "seed": 1, "steps": 4},
            timeout=REQUEST_TIMEOUT,
        )
        resp2 = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={"prompt": prompt, "seed": 999, "steps": 4},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        b64_1 = resp1.json()["artifacts"][0]["base64"]
        b64_2 = resp2.json()["artifacts"][0]["base64"]
        assert b64_1 != b64_2, "Different seeds should produce different output"

    def test_saves_visualization(self):
        """Generate image and save visualization to outputs/.

        Note: This NIM has a max of 4 steps — images will be rough/low quality.
        For production quality, use a different model or host the model yourself.
        """
        prompt = "a red fire truck parked in front of a building"
        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={"prompt": prompt, "seed": 42, "steps": 4},
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        image = base64_to_image(resp.json()["artifacts"][0]["base64"])

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.set_title("Text2Img: 'a red fire truck...'", fontsize=10)
        ax.axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / "test01_text2img.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        assert out_path.exists()
