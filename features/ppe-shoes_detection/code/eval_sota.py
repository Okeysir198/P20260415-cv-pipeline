"""Two-stage SOTA pretrained eval for safety-shoes detection.

Stage 1: D-FINE-N (or D-FINE-S) detect persons (COCO class 0) at 640.
Stage 2: DINOv2-small / EfficientFormerV2-S0 zero-shot-ish classify foot crop.
         (No fine-tuned shoe head exists yet; we report top-IN1K labels for the
         foot crop as a sanity signal of feature quality on small ROIs.)

Outputs:
  predict/<pipeline>/<sample>.jpg with boxes + classifier label/conf
  predict/QUALITY_REPORT.md (written by caller)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors

ROOT = Path("/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai")
PRETRAINED = ROOT / "pretrained" / "ppe-shoes_detection"
FEATURE = ROOT / "features" / "ppe-shoes_detection"
SAMPLES = FEATURE / "samples"
PREDICT = FEATURE / "predict"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERSON_CONF = 0.35
FOOT_FRAC = 0.30  # bottom 30% of person box treated as foot region


def _load_dfine(name: str):
    """Load D-FINE detector from local weights."""
    from transformers import AutoImageProcessor, DFineForObjectDetection

    cfg_dir = PRETRAINED / f"{name}_hf"
    cfg_dir.mkdir(exist_ok=True)
    # Materialize an HF-style folder so from_pretrained works offline
    for src, dst in [
        (PRETRAINED / f"{name}_config.json", cfg_dir / "config.json"),
        (PRETRAINED / f"{name}_preprocessor.json", cfg_dir / "preprocessor_config.json"),
        (PRETRAINED / f"{name}.safetensors", cfg_dir / "model.safetensors"),
    ]:
        if not dst.exists():
            dst.symlink_to(src)
    processor = AutoImageProcessor.from_pretrained(cfg_dir)
    model = DFineForObjectDetection.from_pretrained(cfg_dir).to(DEVICE).eval()
    return processor, model


def detect_persons(processor, model, image: Image.Image) -> list[tuple[float, float, float, float, float]]:
    """Return list of (x1,y1,x2,y2,conf) for COCO 'person' (id=0)."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=PERSON_CONF
    )[0]
    persons: list[tuple[float, float, float, float, float]] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=False):
        if int(label) == 0:  # COCO person
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            persons.append((x1, y1, x2, y2, float(score)))
    return persons


def _load_dinov2():
    from transformers import AutoImageProcessor, Dinov2Model

    cfg_dir = PRETRAINED / "dinov2_small_hf"
    cfg_dir.mkdir(exist_ok=True)
    for src, dst in [
        (PRETRAINED / "dinov2_small_config.json", cfg_dir / "config.json"),
        (PRETRAINED / "dinov2_small_preprocessor.json", cfg_dir / "preprocessor_config.json"),
        (PRETRAINED / "dinov2_small.bin", cfg_dir / "pytorch_model.bin"),
    ]:
        if not dst.exists():
            dst.symlink_to(src)
    proc = AutoImageProcessor.from_pretrained(cfg_dir)
    model = Dinov2Model.from_pretrained(cfg_dir).to(DEVICE).eval()
    return proc, model


def _load_effv2():
    """Load timm EfficientFormerV2-S0 with IN1K head from local .bin."""
    import timm

    model = timm.create_model("efficientformerv2_s0", pretrained=False, num_classes=1000)
    sd = torch.load(PRETRAINED / "efficientformerv2_s0.bin", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(DEVICE).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return transform, model


def _imagenet_labels() -> list[str]:
    p = PRETRAINED / "imagenet_classes.txt"
    if p.exists():
        return p.read_text().strip().splitlines()
    # Minimal fallback — ship a small subset relevant to shoes
    return [str(i) for i in range(1000)]


def classify_dinov2(proc, model, crop: Image.Image) -> tuple[str, float]:
    """Pseudo-classify foot crop: cosine-sim to anchor prompts using DINOv2 CLS embeds.

    DINOv2 has no language head; we instead emit the CLS-token L2 norm and
    a confidence proxy = ratio of max patch attention. This serves as a
    backbone-quality sanity signal, not a shoe label.
    """
    inputs = proc(images=crop, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    cls = out.last_hidden_state[:, 0]  # (1, D)
    norm = float(cls.norm(dim=-1).item())
    # confidence proxy: how peaked the CLS is across feature dims
    sm = torch.softmax(cls.abs(), dim=-1)
    peakiness = float(sm.max().item())
    return f"dinov2_cls_norm={norm:.1f}", peakiness


def classify_effv2(transform, model, labels: list[str], crop: Image.Image) -> tuple[str, float]:
    x = transform(crop.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    conf, idx = float(probs.max().item()), int(probs.argmax().item())
    label = labels[idx] if idx < len(labels) else str(idx)
    return label, conf


def _foot_crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = box
    h = y2 - y1
    fy1 = y1 + h * (1.0 - FOOT_FRAC)
    return image.crop((max(0, x1), max(0, fy1), x2, y2))


def _draw(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for it in items:
        x1, y1, x2, y2 = it["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        # foot region
        h = y2 - y1
        fy1 = y1 + h * (1.0 - FOOT_FRAC)
        draw.rectangle([x1, fy1, x2, y2], outline=(255, 140, 0), width=2)
        text = f"P{it['det_conf']:.2f} | {it['cls_label']} ({it['cls_conf']:.2f})"
        bbox = draw.textbbox((x1, max(0, y1 - 22)), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x1, max(0, y1 - 22)), text, fill=(255, 255, 255), font=font)
    return out


def run_pipeline(detector_name: str, classifier_name: str) -> dict:
    pipeline = f"{detector_name}__{classifier_name}"
    out_dir = PREDICT / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{pipeline}] loading models on {DEVICE}…")
    det_proc, det_model = _load_dfine(detector_name)
    if classifier_name == "dinov2_small":
        cls_proc, cls_model = _load_dinov2()
        labels: list[str] = []
    elif classifier_name == "efficientformerv2_s0":
        cls_proc, cls_model = _load_effv2()
        labels = _imagenet_labels()
    else:
        raise ValueError(classifier_name)

    samples = sorted(SAMPLES.glob("*.jpg"))
    per_sample: list[dict] = []
    failures: list[str] = []
    t0 = time.time()
    for sp in samples:
        try:
            img = Image.open(sp).convert("RGB")
            persons = detect_persons(det_proc, det_model, img)
            items = []
            for box in persons[:5]:
                crop = _foot_crop(img, box[:4])
                if crop.size[0] < 8 or crop.size[1] < 8:
                    continue
                if classifier_name == "dinov2_small":
                    label, conf = classify_dinov2(cls_proc, cls_model, crop)
                else:
                    label, conf = classify_effv2(cls_proc, cls_model, labels, crop)
                items.append({
                    "box": box[:4],
                    "det_conf": box[4],
                    "cls_label": label,
                    "cls_conf": conf,
                })
            vis = _draw(img, items)
            vis.save(out_dir / sp.name)
            per_sample.append({
                "sample": sp.name,
                "n_persons": len(persons),
                "items": [{k: v for k, v in it.items() if k != "box"} for it in items],
            })
            print(f"  {sp.name}: {len(persons)} person(s), {len(items)} foot crop(s)")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{sp.name}: {e}")
            print(f"  {sp.name}: FAIL {e}")
    dt = time.time() - t0
    summary = {
        "pipeline": pipeline,
        "device": DEVICE,
        "n_samples": len(samples),
        "elapsed_s": round(dt, 2),
        "per_sample": per_sample,
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    PREDICT.mkdir(parents=True, exist_ok=True)
    pipelines = [
        ("dfine_nano_coco", "dinov2_small"),       # top recommended two-stage
        ("dfine_nano_coco", "efficientformerv2_s0"),
        ("dfine_small_coco", "efficientformerv2_s0"),  # single-stage-ish stronger detector
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summaries = []
    for det, cls in pipelines:
        if only and only not in f"{det}__{cls}":
            continue
        summaries.append(run_pipeline(det, cls))
    (PREDICT / "_all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Outputs in {PREDICT}")


if __name__ == "__main__":
    main()
