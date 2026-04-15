"""Run pretrained SOTA detectors on fire_detection sample images.

Top 2 models per `safety-fire_detection-sota.md` §3 (D-FINE-M / D-FINE-S).
COCO-pretrained checkpoints have no fire/smoke classes, so detections are
reported as the COCO labels they trigger; this still measures localisation
quality and false-positive behaviour on the 10 sample set.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

ROOT = Path("/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai")
SAMPLES = ROOT / "features/safety-fire_detection/samples"
PREDICT = ROOT / "features/safety-fire_detection/predict"
PRETRAINED = ROOT / "pretrained/safety-fire_detection"

MODELS = {
    "dfine_medium_coco": PRETRAINED / "ustc-community_dfine-medium-coco",
    "dfine_small_coco": PRETRAINED / "ustc-community_dfine-small-coco",
}
THRESHOLD = 0.30


def _font() -> ImageFont.ImageFont:
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, 16)
    return ImageFont.load_default()


def _draw(img: Image.Image, dets: list[dict]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = f"{d['label']} {d['score']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 18)), label, fill="yellow", font=font)
    return out


def _run(model_name: str, model_path: Path, samples: list[Path]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForObjectDetection.from_pretrained(model_path).to(device).eval()
    out_dir = PREDICT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for sp in samples:
        img = Image.open(sp).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target = torch.tensor([img.size[::-1]], device=device)
        post = processor.post_process_object_detection(
            outputs, target_sizes=target, threshold=THRESHOLD
        )[0]
        dets = [
            {
                "label": model.config.id2label[int(lbl)],
                "score": float(score),
                "box": [float(v) for v in box.tolist()],
            }
            for score, lbl, box in zip(post["scores"], post["labels"], post["boxes"])
        ]
        results[sp.name] = dets
        _draw(img, dets).save(out_dir / sp.name)
    (out_dir / "_results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    samples = sorted(SAMPLES.glob("*.jpg"))
    print(f"Samples: {len(samples)}")
    summary: dict[str, dict] = {}
    for name, path in MODELS.items():
        print(f"\n[{name}] loading from {path}")
        summary[name] = _run(name, path, samples)
        print(f"[{name}] wrote {len(samples)} visualisations -> predict/{name}/")
    (PREDICT / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
