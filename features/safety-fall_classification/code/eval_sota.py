"""Run SOTA pretrained models on 10 fall-classification samples.

Models (no fine-tune, ImageNet/Kinetics only):
  1. EfficientNetV2-S (timm, Apache-2.0) — single-frame, ImageNet-1k top-5.
  2. VideoMAE-Small K400 (CC-BY-NC-4.0, research only) — temporal, 16-frame
     clip from a single image (replicated). Top-3 Kinetics-400 actions.

Outputs annotated images under predict/<model>/ and a JSON per-sample log.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/nthanhtrung/Documents/05_Team/02_Vietsol/01_Projects/edge_ai/ai")
WEIGHTS = ROOT / "pretrained/safety-fall_classification"
SAMPLES = ROOT / "features/safety-fall_classification/samples"
PREDICT = ROOT / "features/safety-fall_classification/predict"


# ---------- helpers ---------- #
def _font(sz: int = 18) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _annotate(img: Image.Image, lines: list[str], out: Path) -> None:
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    f = _font(18)
    pad = 6
    h = sum(f.getbbox(t)[3] + 4 for t in lines) + 2 * pad
    w = max(f.getbbox(t)[2] for t in lines) + 2 * pad
    d.rectangle([0, 0, w, h], fill=(0, 0, 0, 200))
    y = pad
    for t in lines:
        d.text((pad, y), t, fill=(255, 255, 0), font=f)
        y += f.getbbox(t)[3] + 4
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def _list_samples() -> list[Path]:
    return sorted(SAMPLES.glob("*.jpg"))


# ---------- ImageNet labels (need them for human-readable top-k) ---------- #
def _imagenet_labels() -> list[str]:
    cache = WEIGHTS / "_imagenet_classes.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import urllib.request

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url, timeout=30) as r:
        labels = [ln.strip() for ln in r.read().decode().splitlines() if ln.strip()]
    cache.write_text(json.dumps(labels))
    return labels


# ---------- Model 1: EfficientNetV2-S ---------- #
def run_efficientnetv2s(samples: list[Path]) -> dict:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnetv2_rw_s", pretrained=False, num_classes=1000)
    sd = torch.load(WEIGHTS / "efficientnetv2_rw_s.ra2_in1k.bin", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    tfm = create_transform(**cfg)
    labels = _imagenet_labels()
    out_dir = PREDICT / "efficientnetv2s_in1k"
    log: dict = {"model": "efficientnetv2_rw_s.ra2_in1k", "license": "Apache-2.0",
                 "task": "ImageNet-1k top-5 (zero-shot — no fall fine-tune)", "samples": {}}

    # tokens hinting at fallen / lying / standing postures (heuristic only)
    fallen_hints = {"stretcher", "crutch", "sleeping bag", "quilt", "mat", "tub",
                     "bath towel", "pillow", "cradle"}

    with torch.inference_mode():
        for p in samples:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            logits = model(x)[0]
            probs = torch.softmax(logits, dim=0)
            top = torch.topk(probs, 5)
            picks = [(labels[i], float(probs[i])) for i in top.indices.tolist()]
            top1, conf = picks[0]
            heuristic = "fallen-like" if any(h in t for t, _ in picks for h in fallen_hints) else "upright/other"
            lines = [f"EffNetV2-S | {heuristic}", f"top1: {top1} ({conf:.2f})"]
            _annotate(img, lines, out_dir / p.name)
            log["samples"][p.name] = {"top1": top1, "top1_conf": round(conf, 4),
                                       "top5": [(t, round(c, 4)) for t, c in picks],
                                       "heuristic": heuristic}
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    return log


# ---------- Model 2: VideoMAE-Small K400 ---------- #
def run_videomae_small(samples: list[Path]) -> dict:
    from transformers import AutoImageProcessor, VideoMAEForVideoClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = "MCG-NJU/videomae-small-finetuned-kinetics"
    proc = AutoImageProcessor.from_pretrained(repo)
    model = VideoMAEForVideoClassification.from_pretrained(repo).eval().to(device)
    out_dir = PREDICT / "videomae_small_k400"
    log: dict = {"model": repo, "license": "CC-BY-NC-4.0 (research-only)",
                 "task": "Kinetics-400 top-3 (clip = single frame x16)", "samples": {}}

    fall_hints = {"falling", "fall", "lying", "yoga", "stretching", "sleeping",
                   "crawling", "wrestling"}

    with torch.inference_mode():
        for p in samples:
            img = Image.open(p).convert("RGB")
            frames = [img] * 16  # static-frame surrogate; VideoMAE tokenises 16 frames
            inputs = proc(frames, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=0)
            top = torch.topk(probs, 3)
            picks = [(model.config.id2label[i], float(probs[i])) for i in top.indices.tolist()]
            top1, conf = picks[0]
            heuristic = "fallen-like" if any(h in t.lower() for t, _ in picks for h in fall_hints) else "upright/other"
            lines = [f"VideoMAE-S K400 | {heuristic}", f"top1: {top1} ({conf:.2f})"]
            _annotate(img, lines, out_dir / p.name)
            log["samples"][p.name] = {"top1": top1, "top1_conf": round(conf, 4),
                                       "top3": [(t, round(c, 4)) for t, c in picks],
                                       "heuristic": heuristic}
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    return log


def main() -> None:
    samples = _list_samples()
    if len(samples) != 10:
        print(f"WARN: expected 10 samples, found {len(samples)}")
    print(f"running on {len(samples)} samples, device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    eff = run_efficientnetv2s(samples)
    print(f"EffNetV2-S done -> {PREDICT / 'efficientnetv2s_in1k'}")

    vm = run_videomae_small(samples)
    print(f"VideoMAE-S done -> {PREDICT / 'videomae_small_k400'}")

    summary = {"efficientnetv2s": {n: v["top1"] for n, v in eff["samples"].items()},
               "videomae_small": {n: v["top1"] for n, v in vm["samples"].items()}}
    (PREDICT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
