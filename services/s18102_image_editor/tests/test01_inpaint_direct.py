"""Image Editor service tests — POST /inpaint (direct edit, no mask)."""

import matplotlib.pyplot as plt
import requests
from PIL import Image

from conftest import (
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    SERVICE_URL,
    base64_to_image,
    generate_source_image,
    image_to_base64,
    skip_no_service,
)


@skip_no_service
class TestInpaintDirect:
    def test_direct_edit(self):
        """Direct edit (no mask) returns result images."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "same room but during sunset with warm orange light",
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "images" in data
        assert len(data["images"]) >= 1
        assert data["mask_used"] is None, "Direct edit should not use a mask"

    def test_direct_edit_multiple_variants(self):
        """Direct edit with num_variants returns multiple results."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "same room in a futuristic style",
                "num_variants": 2,
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) == 2

    def test_saves_visualization(self):
        """Generate direct edit and save visualization to outputs/."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "same room but during sunset with warm orange light",
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        result = base64_to_image(resp.json()["images"][0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(source)
        axes[0].set_title("Source")
        axes[0].axis("off")
        axes[1].imshow(result)
        axes[1].set_title("Direct Edit (no mask)")
        axes[1].axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / "test01_inpaint_direct.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        assert out_path.exists()
