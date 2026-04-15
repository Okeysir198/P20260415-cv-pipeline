"""Image Editor service tests — POST /inpaint (bbox -> SAM3 segmentation)."""

import matplotlib.pyplot as plt
import requests
from matplotlib.patches import Rectangle
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
class TestInpaintBbox:
    def test_bbox_inpaint(self):
        """Bbox inpaint returns result with mask."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "a cute orange cat sitting comfortably",
                "bbox": [200, 200, 500, 500],
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["images"]) >= 1
        assert data["mask_used"] is not None, "Bbox inpaint should produce a mask"

    def test_bbox_result_is_composited(self):
        """Bbox inpaint result differs from source (compositing happened)."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        source_b64 = image_to_base64(source)

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": source_b64,
                "prompt": "a bright yellow sunflower",
                "bbox": [100, 100, 400, 400],
                "seed": 7,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        result_b64 = resp.json()["images"][0]
        assert result_b64 != source_b64, "Result should differ from source"

    def test_saves_visualization(self):
        """Generate bbox inpaint and save visualization to outputs/."""
        source = generate_source_image()
        source.thumbnail((768, 768), Image.Resampling.LANCZOS)
        bbox = [200, 200, 500, 500]

        resp = requests.post(
            f"{SERVICE_URL}/inpaint",
            json={
                "image": image_to_base64(source),
                "prompt": "a cute orange cat sitting comfortably",
                "bbox": bbox,
                "seed": 42,
                "steps": 4,
            },
            timeout=REQUEST_TIMEOUT,
        )
        assert resp.status_code == 200
        result = base64_to_image(resp.json()["images"][0])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(source)
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].set_title(f"Source + BBox {bbox}")
        axes[0].axis("off")
        axes[1].imshow(result)
        axes[1].set_title("BBox Inpaint Result")
        axes[1].axis("off")
        plt.tight_layout()

        out_path = OUTPUT_DIR / "test02_inpaint_bbox.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        assert out_path.exists()
