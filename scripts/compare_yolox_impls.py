"""Parity check: custom vs official YOLOX on COCO pretrained weights.

Loads ``pretrained/yolox_m.pth`` (Megvii YOLOX-M, COCO 80-class) into BOTH:

* ``YOLOXModel`` — self-contained custom reimplementation
* ``_OfficialYOLOXAdapter`` wrapping the upstream ``yolox`` package

Runs identical preprocessing on one image, compares raw decoded outputs
element-wise, then applies the same NMS + threshold and compares the
top-K detections.

If the two implementations are functionally equivalent, the per-anchor
output tensors should match to within fp32 rounding noise (max abs diff
< 1e-3 after sigmoid).

Usage (requires the official yolox venv — the main .venv/ has no yolox pkg):
    ./.venv-yolox-official/bin/python scripts/compare_yolox_impls.py \\
        --image services/s18100_sam3_service/demo/cars.jpg \\
        --out eval/yolox_parity.png
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

INPUT_SIZE = 640
PRETRAINED = ROOT / "pretrained" / "yolox_m.pth"

# Reuse COCO80 + palette from check_pretrained
from core.p06_models.check_pretrained import COCO80, _draw_panel, _hstack_panels


def _preprocess(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """YOLOX-M expects [0, 255] raw pixels, BGR→RGB, (1, 3, H, W)."""
    resized = cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE))
    rgb = resized[:, :, ::-1].copy().astype(np.float32)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    return state


def _build_custom(device: torch.device, state: dict) -> torch.nn.Module:
    from core.p06_models.yolox import YOLOXModel
    model = YOLOXModel(num_classes=80, depth=0.67, width=0.75)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("custom YOLOXModel: %d missing, %d unexpected keys", len(missing), len(unexpected))
    return model.to(device).eval()


def _build_official(device: torch.device, state: dict) -> torch.nn.Module:
    from core.p06_models.yolox import _OfficialYOLOXAdapter
    adapter = _OfficialYOLOXAdapter(num_classes=80, depth=0.67, width=0.75)
    # Official state_dict has the YOLOX(backbone, head) layout that
    # adapter._model matches exactly — no key remapping needed.
    missing, unexpected = adapter._model.load_state_dict(state, strict=False)
    logger.info(
        "official _OfficialYOLOXAdapter: %d missing, %d unexpected keys",
        len(missing), len(unexpected),
    )
    return adapter.to(device).eval()


def _postprocess(raw: torch.Tensor, orig_h: int, orig_w: int, conf: float, nms: float) -> dict:
    """Accepts (1, N, 5+80) with obj+cls already sigmoid'd. Returns NMS'd detections."""
    pred = raw[0]
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
    keep = batched_nms(boxes, scores, cls_ids, nms)
    return {
        "boxes":  boxes[keep].cpu().numpy(),
        "scores": scores[keep].cpu().numpy(),
        "labels": cls_ids[keep].cpu().numpy().astype(np.int64),
    }


def _tensor_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a - b).abs()
    logger.info(
        "  %s  shape=%s  max|Δ|=%.3e  mean|Δ|=%.3e  cosine=%.6f",
        name, tuple(a.shape), diff.max().item(), diff.mean().item(),
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    )


def _match_detections(
    r_a: dict, r_b: dict, iou_thresh: float = 0.5
) -> tuple[int, float, float]:
    """Greedy 1-to-1 match by class + IoU. Returns (matched_count, mean_iou, mean_score_diff)."""
    if len(r_a["boxes"]) == 0 or len(r_b["boxes"]) == 0:
        return 0, 0.0, 0.0
    ba = torch.from_numpy(r_a["boxes"]).float()
    bb = torch.from_numpy(r_b["boxes"]).float()
    from torchvision.ops import box_iou
    ious = box_iou(ba, bb)
    matches = 0
    iou_sum = 0.0
    score_diff_sum = 0.0
    used_b = set()
    for i in range(len(ba)):
        best_iou, best_j = -1.0, -1
        for j in range(len(bb)):
            if j in used_b or r_a["labels"][i] != r_b["labels"][j]:
                continue
            iou = ious[i, j].item()
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            matches += 1
            iou_sum += best_iou
            score_diff_sum += abs(r_a["scores"][i] - r_b["scores"][best_j])
            used_b.add(best_j)
    if matches == 0:
        return 0, 0.0, 0.0
    return matches, iou_sum / matches, score_diff_sum / matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare custom vs official YOLOX on pretrained COCO weights")
    parser.add_argument("--image", default=str(ROOT / "services/s18100_sam3_service/demo/cars.jpg"))
    parser.add_argument("--out",   default=str(ROOT / "eval/yolox_parity.png"))
    parser.add_argument("--conf",  type=float, default=0.3)
    parser.add_argument("--nms",   type=float, default=0.45)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Image: %s", args.image)

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Cannot read image: {args.image}")
    h, w = image.shape[:2]

    if not PRETRAINED.exists():
        sys.exit(f"Pretrained weights not found: {PRETRAINED}")
    state = _load_state(PRETRAINED)
    logger.info("Loaded %d keys from %s", len(state), PRETRAINED.name)

    tensor = _preprocess(image, device)

    # ---- Custom YOLOXModel ----
    logger.info("=== Custom YOLOXModel ===")
    m_custom = _build_custom(device, state)
    with torch.no_grad():
        raw_custom = m_custom(tensor)
    r_custom = _postprocess(raw_custom, h, w, args.conf, args.nms)
    logger.info("custom: %d detections after NMS", len(r_custom["boxes"]))

    # ---- Official via adapter ----
    logger.info("=== Official _OfficialYOLOXAdapter ===")
    m_official = _build_official(device, state)
    with torch.no_grad():
        raw_official = m_official(tensor)
    r_official = _postprocess(raw_official, h, w, args.conf, args.nms)
    logger.info("official: %d detections after NMS", len(r_official["boxes"]))

    # ---- Parity report ----
    logger.info("=== Raw output parity (1, N=%d, %d) ===", raw_custom.shape[1], raw_custom.shape[2])
    _tensor_stats("full tensor", raw_custom, raw_official)
    _tensor_stats("  xy (cx,cy)", raw_custom[..., :2], raw_official[..., :2])
    _tensor_stats("  wh (w,h)  ", raw_custom[..., 2:4], raw_official[..., 2:4])
    _tensor_stats("  obj       ", raw_custom[..., 4], raw_official[..., 4])
    _tensor_stats("  cls logits", raw_custom[..., 5:], raw_official[..., 5:])

    matched, miou, dscore = _match_detections(r_custom, r_official)
    logger.info(
        "=== Detection parity ===  custom=%d  official=%d  matched=%d  mean_IoU=%.4f  mean_|Δscore|=%.4f",
        len(r_custom["boxes"]), len(r_official["boxes"]), matched, miou, dscore,
    )
    top_k = min(5, len(r_custom["boxes"]), len(r_official["boxes"]))
    if top_k > 0:
        logger.info("Top-%d by score:", top_k)
        # sort each by score desc
        a_idx = np.argsort(-r_custom["scores"])[:top_k]
        b_idx = np.argsort(-r_official["scores"])[:top_k]
        for i in range(top_k):
            la, lb = r_custom["labels"][a_idx[i]], r_official["labels"][b_idx[i]]
            sa, sb = r_custom["scores"][a_idx[i]], r_official["scores"][b_idx[i]]
            ba, bb = r_custom["boxes"][a_idx[i]], r_official["boxes"][b_idx[i]]
            logger.info(
                "  #%d  custom=%s(%.3f) box=%s   official=%s(%.3f) box=%s",
                i + 1, COCO80[la], sa, np.round(ba, 1).tolist(),
                COCO80[lb], sb, np.round(bb, 1).tolist(),
            )

    # ---- Visualization ----
    panels = [
        _draw_panel(image, r_custom,   COCO80, "Custom YOLOXModel (COCO)"),
        _draw_panel(image, r_official, COCO80, "Official Megvii YOLOX (COCO)"),
    ]
    grid = _hstack_panels(panels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    logger.info("Saved grid: %s  (%dx%d)", out_path, grid.shape[1], grid.shape[0])


if __name__ == "__main__":
    main()
