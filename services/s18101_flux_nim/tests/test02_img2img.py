"""Flux NIM service tests — POST /v1/infer (image-to-image)."""

import matplotlib.pyplot as plt
import requests
from PIL import Image

from conftest import (
    DATA_DIR,
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    base64_to_image,
    image_to_base64,
    skip_no_service,
)


def _generate_source_image() -> Image.Image:
    """Generate a safe source image via text2img for img2img tests."""
    resp = requests.post(
        f"{SERVICE_URL}/v1/infer",
        json={"prompt": "a cozy living room with a sofa and bookshelf, photorealistic", "seed": 0, "steps": 4},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return base64_to_image(resp.json()["artifacts"][0]["base64"])


@skip_no_service
class TestImg2Img:
    def test_basic_img2img(self):
        """Image-to-image returns an artifact from source image + prompt."""
        source = _generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        source_b64 = image_to_base64(source)

        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={
                "prompt": "same scene but during a beautiful sunset with orange sky",
                "image": [f"data:image/png;base64,{source_b64}"],
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "artifacts" in data
        assert len(data["artifacts"]) > 0
        artifact = data["artifacts"][0]
        # May be CONTENT_FILTERED or SUCCESS
        if artifact.get("finishReason") == "SUCCESS":
            assert artifact.get("base64"), "SUCCESS artifact should have base64 data"

    def test_img2img_from_file(self):
        """Image-to-image works with an image loaded from disk."""
        source = Image.open(DATA_DIR / "test_scene.jpg").convert("RGB")
        source_b64 = image_to_base64(source)

        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={
                "prompt": "a serene lake surrounded by mountains at sunset, photorealistic",
                "image": [f"data:image/png;base64,{source_b64}"],
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        artifacts = resp.json().get("artifacts", [])
        assert len(artifacts) > 0

    def test_img2img_output_dimensions(self):
        """Image-to-image output has expected dimensions."""
        source = _generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        source_b64 = image_to_base64(source)

        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={
                "prompt": "a winter scene with snow covering everything",
                "image": [f"data:image/png;base64,{source_b64}"],
                "seed": 7,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        artifact = resp.json()["artifacts"][0]
        if artifact.get("finishReason") == "SUCCESS":
            image = base64_to_image(artifact["base64"])
            assert image.size[0] == 1024
            assert image.size[1] == 1024

    def test_saves_visualization(self):
        """Generate img2img result and save visualization to outputs/."""
        source = _generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        source_b64 = image_to_base64(source)

        prompt = "same scene but during a beautiful sunset with orange sky"
        resp = requests.post(
            f"{SERVICE_URL}/v1/infer",
            json={
                "prompt": prompt,
                "image": [f"data:image/png;base64,{source_b64}"],
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        artifact = resp.json()["artifacts"][0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(source)
        axes[0].set_title("Source")
        axes[0].axis("off")

        if artifact.get("finishReason") == "SUCCESS" and artifact.get("base64"):
            result = base64_to_image(artifact["base64"])
            axes[1].imshow(result)
            axes[1].set_title(f"Img2Img: '{prompt[:40]}...'")
        else:
            axes[1].text(0.5, 0.5, "CONTENT_FILTERED", ha="center", va="center", fontsize=14)
            axes[1].set_title("Img2Img: Content filtered")
        axes[1].axis("off")

        plt.tight_layout()
        out_path = OUTPUT_DIR / "test02_img2img.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        assert out_path.exists()
