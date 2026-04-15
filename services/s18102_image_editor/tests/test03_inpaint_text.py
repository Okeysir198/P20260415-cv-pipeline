"""Image Editor service tests — POST /inpaint (text_prompt -> SAM3 segmentation)."""

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
class TestInpaintText:
    def test_text_inpaint(self):
        """Text-prompted inpaint returns result with mask."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "a golden trophy",
                "text_prompt": "sofa",
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) >= 1
        assert data["mask_used"] is not None, "Text inpaint should produce a mask"

    def test_text_inpaint_result_composited(self):
        """Text inpaint result is composited (differs from source)."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        source_b64 = image_to_base64(source)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": source_b64,
                "prompt": "a wooden barrel",
                "text_prompt": "bookshelf",
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        result_b64 = resp.json()["images"][0]
        assert result_b64 != source_b64, "Result should differ from source"

    def test_saves_visualization(self):
        """Generate text inpaint and save visualization to outputs/."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "a golden trophy",
                "text_prompt": "sofa",
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        result = base64_to_image(resp.json()["images"][0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(source)
        axes[0].set_title("Source (segment: 'sofa')")
        axes[0].axis("off")
        axes[1].imshow(result)
        axes[1].set_title("Text Inpaint: 'a golden trophy'")
        axes[1].axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / "test03_inpaint_text.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        assert out_path.exists()
