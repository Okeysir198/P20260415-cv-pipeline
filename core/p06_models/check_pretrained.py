"""Verify pretrained weights load correctly by running COCO-mode inference
across YOLOX-M, D-FINE-S, and RT-DETRv2-R18 on the same image.

All three models are loaded with their original COCO 80-class head so
real detections are expected. This is a sanity check — if detections look
reasonable the backbone transfer is confirmed.

Usage:
    uv run core/p06_models/check_pretrained.py --image path/to/image.jpg
    uv run core/p06_models/check_pretrained.py --image path/to/image.jpg \
        --out eval/pretrained_check.png --conf 0.3
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_SIZE = 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO 80-class names (index matches yolox_m.pth label order)
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_PALETTE = [
    (255, 80,  0), (0, 200,  0), (0, 100, 255), (200,   0, 200),
    (0, 200, 200), (255, 200, 0), (100,  0, 255), (255,   0, 100),
    (0, 255, 100), (100, 255,  0), (180, 180,  0), (0, 180, 180),
]


# ---------------------------------------------------------------------------
# YOLOX — 80-class COCO pretrained
# ---------------------------------------------------------------------------

def _load_yolox(device: torch.device) -> torch.nn.Module:
    from core.p06_models.yolox import YOLOXModel

    model = YOLOXModel(num_classes=80, depth=0.67, width=0.75)
    pth = ROOT / "pretrained" / "yolox_m.pth"
    if not pth.exists():
        raise FileNotFoundError(f"YOLOX pretrained not found: {pth}")

    state = torch.load(pth, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("YOLOX-M: %d missing, %d unexpected keys", len(missing), len(unexpected))
    return model.to(device).eval()


def _preprocess_yolox(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    # YOLOX was trained on raw pixel values [0, 255] — no /255 normalization.
    resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE))
    rgb = resized[:, :, ::-1].copy().astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def _postprocess_yolox(
    raw: torch.Tensor,
    orig_h: int,
    orig_w: int,
    conf: float,
    nms_thresh: float,
) -> dict:
    """raw: (1, N, 5+80) — cx,cy,w,h are decoded pixels; obj+cls are raw logits."""
    pred = raw[0]  # (N, 85) — obj+cls already sigmoid'd by _DecoupledHead in eval mode
    obj = pred[:, 4]
    cls_scores, cls_ids = pred[:, 5:].max(dim=1)
    scores = obj * cls_scores

    mask = scores >= conf
    pred, scores, cls_ids = pred[mask], scores[mask], cls_ids[mask]

    if pred.shape[0] == 0:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)}

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w / 2) * orig_w / INPUT_SIZE
    y1 = (cy - h / 2) * orig_h / INPUT_SIZE
    x2 = (cx + w / 2) * orig_w / INPUT_SIZE
    y2 = (cy + h / 2) * orig_h / INPUT_SIZE
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep = batched_nms(boxes, scores, cls_ids, nms_thresh)
    return {
        "boxes":  boxes[keep].cpu().numpy(),
        "scores": scores[keep].cpu().numpy(),
        "labels": cls_ids[keep].cpu().numpy().astype(np.int64),
    }


def infer_yolox(model, image_bgr: np.ndarray, device: torch.device, conf: float) -> dict:
    h, w = image_bgr.shape[:2]
    tensor = _preprocess_yolox(image_bgr, device)
    with torch.no_grad():
        raw = model(tensor)
    return _postprocess_yolox(raw, h, w, conf, nms_thresh=0.45)


# ---------------------------------------------------------------------------
# HF models (D-FINE-S, RT-DETRv2-R18) — original COCO head, no num_labels override
# ---------------------------------------------------------------------------

def _load_hf(model_id: str, device: torch.device):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    logger.info("Loading HF model: %s", model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id)
    return model.to(device).eval(), processor


def infer_hf(model, processor, image_bgr: np.ndarray, device: torch.device, conf: float) -> tuple[dict, list[str]]:
    from PIL import Image as PILImage

    h, w = image_bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = {k: v.to(device) for k, v in processor(images=pil, return_tensors="pt").items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=[(h, w)])[0]
    id2label = model.config.id2label
    class_names = [id2label.get(i, str(i)) for i in range(max(id2label) + 1)]

    return {
        "boxes":  results["boxes"].cpu().numpy(),
        "scores": results["scores"].cpu().numpy(),
        "labels": results["labels"].cpu().numpy().astype(np.int64),
    }, class_names


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_panel(image_bgr: np.ndarray, result: dict, class_names: list[str], title: str) -> np.ndarray:
    img = image_bgr.copy()
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"], strict=True):
        x1, y1, x2, y2 = map(int, box)
        color = _PALETTE[int(label) % len(_PALETTE)]
        name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw + 3, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Title banner
    n = len(result["boxes"])
    banner_text = f"{title}  |  {n} detection{'s' if n != 1 else ''}"
    banner = np.full((38, img.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(banner, banner_text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    return np.vstack([banner, img])


def _hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    max_h = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, max_h - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    # Thin separator between columns
    sep = np.full((max_h, 4, 3), 80, dtype=np.uint8)
    out = padded[0]
    for p in padded[1:]:
        out = np.hstack([out, sep, p])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained weight sanity check — COCO inference on one image")
    parser.add_argument("--image",  required=True, help="Input image path")
    parser.add_argument("--out",    default="pretrained_check.png", help="Output grid path")
    parser.add_argument("--conf",   type=float, default=0.3, help="Confidence threshold (default 0.3)")
    args = parser.parse_args()

    from utils.device import auto_select_gpu
    auto_select_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Cannot read image: {args.image}")

    panels = []

    # --- YOLOX-M ---
    logger.info("=== YOLOX-M (COCO 80) ===")
    yolox = _load_yolox(device)
    r = infer_yolox(yolox, image, device, args.conf)
    panels.append(_draw_panel(image, r, COCO80, "YOLOX-M (COCO)"))
    logger.info("YOLOX-M: %d detections", len(r["boxes"]))
    del yolox; torch.cuda.empty_cache()

    # --- D-FINE-S ---
    logger.info("=== D-FINE-S (COCO 80) ===")
    dfine, dfine_proc = _load_hf("ustc-community/dfine_s_coco", device)
    r, names = infer_hf(dfine, dfine_proc, image, device, args.conf)
    panels.append(_draw_panel(image, r, names, "D-FINE-S (COCO)"))
    logger.info("D-FINE-S: %d detections", len(r["boxes"]))
    del dfine; torch.cuda.empty_cache()

    # --- RT-DETRv2-R18 ---
    logger.info("=== RT-DETRv2-R18 (COCO 80) ===")
    rtdetr, rtdetr_proc = _load_hf("PekingU/rtdetr_v2_r18vd", device)
    r, names = infer_hf(rtdetr, rtdetr_proc, image, device, args.conf)
    panels.append(_draw_panel(image, r, names, "RT-DETRv2-R18 (COCO)"))
    logger.info("RT-DETRv2-R18: %d detections", len(r["boxes"]))
    del rtdetr; torch.cuda.empty_cache()

    grid = _hstack_panels(panels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    logger.info("Saved: %s  (%dx%d)", out_path, grid.shape[1], grid.shape[0])


if __name__ == "__main__":
    main()
