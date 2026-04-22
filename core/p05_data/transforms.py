"""Core data augmentation transforms for object detection training.

Built on torchvision.transforms.v2 with tv_tensors for joint image+bbox transforms.

External API: build_transforms(config, is_train, input_size, mean, std) returns a
callable that accepts (HWC uint8 BGR ndarray, (N,5) ndarray) and returns
(CHW float32 tensor, (N,5) tensor).

Internal representation uses v2 dict format:
    {"image": tv_tensors.Image, "boxes": tv_tensors.BoundingBoxes, "labels": Tensor}
"""

import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def _to_v2_sample(
    image_bgr_np: np.ndarray,
    targets_np: np.ndarray,
    canvas_size: tuple[int, int] | None = None,
) -> dict:
    """BGR numpy + (N,5) normalized CXCYWH -> v2 sample dict.

    Args:
        image_bgr_np: HWC uint8 BGR numpy array.
        targets_np: (N, 5) array with [class_id, cx, cy, w, h] normalized 0-1.
        canvas_size: (H, W) of the image. Defaults to image_bgr_np.shape[:2].

    Returns:
        Dict with "image" (tv_tensors.Image), "boxes" (tv_tensors.BoundingBoxes),
        and "labels" (Tensor).
    """
    if canvas_size is None:
        canvas_size = (image_bgr_np.shape[0], image_bgr_np.shape[1])
    h, w = canvas_size

    # BGR -> RGB via cv2 (C-optimized, faster than numpy strided `.copy()`
    # or tensor `.flip()`), then HWC uint8 -> CHW via `.permute`. Both
    # alternatives measured slower on CPPE-5 at bs=16, 2 workers.
    rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
    image = tv_tensors.Image(image_tensor)

    if len(targets_np) == 0:
        boxes = tv_tensors.BoundingBoxes(
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        labels = torch.zeros(0, dtype=torch.int64)
    else:
        class_ids = torch.from_numpy(targets_np[:, 0].astype(np.int64))
        cx = targets_np[:, 1] * w
        cy = targets_np[:, 2] * h
        bw = targets_np[:, 3] * w
        bh = targets_np[:, 4] * h
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        xyxy = torch.from_numpy(
            np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        )
        boxes = tv_tensors.BoundingBoxes(
            xyxy,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        labels = class_ids

    return {"image": image, "boxes": boxes, "labels": labels}


def _from_v2_sample(sample: dict, canvas_size: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """v2 sample dict -> (CHW float32 tensor, (N,5) tensor).

    Args:
        sample: Dict with "image", "boxes", "labels".
        canvas_size: (H, W) for normalizing box coordinates. Defaults to image shape.

    Returns:
        (image_tensor, targets_tensor) where targets has [class_id, cx, cy, w, h]
        with normalized coordinates.
    """
    image = sample["image"]
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    h, w = canvas_size if canvas_size is not None else image.shape[-2:]
    boxes = sample["boxes"]
    labels = sample["labels"]

    if len(boxes) == 0:
        targets = torch.zeros((0, 5), dtype=torch.float32)
    else:
        # boxes are pixel XYXY; convert to normalized CXCYWH
        xyxy = boxes.float()
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        class_ids = labels.float()
        targets = torch.stack([class_ids, cx, cy, bw, bh], dim=1)

    return image, targets


class DetectionTransform:
    """Wraps a v2.Compose pipeline with entry/exit coordinate conversion.

    Accepts (HWC uint8 BGR ndarray, (N,5) ndarray) and returns
    (CHW float32 tensor, (N,5) tensor).
    """

    def __init__(
        self,
        v2_transforms: list[Any],
        canvas_size: tuple[int, int],
    ) -> None:
        self.pipeline = v2.Compose(v2_transforms)
        self.transforms = v2_transforms
        self.canvas_size = canvas_size

    def __call__(
        self, image: np.ndarray, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_size = (image.shape[0], image.shape[1])
        sample = _to_v2_sample(image, targets, original_size)
        sample = self.pipeline(sample)
        return _from_v2_sample(sample, self.canvas_size)


class AlbumentationsDetectionTransform:
    """Albumentations-backed alternative to :class:`DetectionTransform`.

    Same callable interface (HWC uint8 BGR ndarray, (N,5) ndarray) →
    (CHW float32 tensor, (N,5) tensor with normalized cxcywh targets), but
    uses Albumentations pipelines under the hood. ~2x faster than
    torchvision v2 on CPU for the detection augment set we run for DETR
    training (RandomPerspective + RandomApply + ColorJitter + Flip + Resize),
    matching qubvel's reference notebook recipe.
    """

    def __init__(
        self,
        albu_pipeline: Any,
        canvas_size: tuple[int, int],
        normalize: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.pipeline = albu_pipeline
        self.canvas_size = canvas_size
        self.normalize = normalize
        self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(
        self, image: np.ndarray, targets: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = image.shape[:2]

        # YOLO normalized cxcywh → COCO pixel xywh that Albumentations
        # expects with `bbox_params=A.BboxParams(format="coco", ...)`.
        if len(targets) > 0:
            cx, cy, bw, bh = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
            x = (cx - bw / 2) * W
            y = (cy - bh / 2) * H
            w_px = bw * W
            h_px = bh * H
            bboxes = [[float(x[i]), float(y[i]), float(w_px[i]), float(h_px[i])]
                      for i in range(len(targets))]
            categories = targets[:, 0].astype(np.int64).tolist()
        else:
            bboxes, categories = [], []

        # cv2.imread returns BGR — convert to RGB for Albumentations.
        img_rgb = image[:, :, ::-1].copy()
        out = self.pipeline(image=img_rgb, bboxes=bboxes, category=categories)
        aug_img = out["image"]               # HWC RGB uint8 or float32
        aug_boxes = out["bboxes"]            # list of [x, y, w, h] pixel
        aug_cats = out["category"]           # list of int

        # Convert image → float [0, 1] tensor (C, H, W), optional ImageNet norm.
        if aug_img.dtype != np.float32:
            aug_img = aug_img.astype(np.float32) / 255.0
        elif aug_img.max() > 1.5:  # rare: some transforms return [0, 255] float
            aug_img = aug_img / 255.0
        if self.normalize:
            aug_img = (aug_img - self._mean) / self._std
        aug_img = np.ascontiguousarray(aug_img.transpose(2, 0, 1))  # CHW
        image_tensor = torch.from_numpy(aug_img)

        # COCO pixel xywh → YOLO normalized cxcywh targets.
        out_H, out_W = image_tensor.shape[1], image_tensor.shape[2]
        if aug_boxes:
            boxes_arr = np.asarray(aug_boxes, dtype=np.float32)
            cats_arr = np.asarray(aug_cats, dtype=np.float32).reshape(-1, 1)
            cx = (boxes_arr[:, 0] + boxes_arr[:, 2] / 2) / out_W
            cy = (boxes_arr[:, 1] + boxes_arr[:, 3] / 2) / out_H
            bw = boxes_arr[:, 2] / out_W
            bh = boxes_arr[:, 3] / out_H
            targets_out = np.stack([cats_arr.ravel(), cx, cy, bw, bh], axis=1).astype(np.float32)
        else:
            targets_out = np.zeros((0, 5), dtype=np.float32)
        return image_tensor, torch.from_numpy(targets_out)

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "DetectionTransform([\n" + "\n".join(lines) + "\n])"


class AlbumentationsWithProcessorTransform:
    """Albumentations aug + HF image_processor (qubvel pattern).

    1. Augment via Albumentations (COCO pixel xywh bboxes, NO resize)
    2. Format as COCO annotations
    3. Call image_processor(images=..., annotations=...) -> resize + rescale + bbox convert

    Returns (pixel_values_tensor, hf_labels_dict) -- the image_processor handles
    resize, uint8->[0,1] rescaling, and COCO->norm-cxcywh bbox conversion atomically.
    """

    def __init__(self, albu_pipeline, image_processor):
        self.pipeline = albu_pipeline
        self.image_processor = image_processor

    def __call__(self, image, targets):
        H, W = image.shape[:2]

        # YOLO normalized cxcywh -> COCO pixel xywh
        if len(targets) > 0:
            cx, cy, bw, bh = targets[:, 1], targets[:, 2], targets[:, 3], targets[:, 4]
            x = (cx - bw / 2) * W
            y = (cy - bh / 2) * H
            w_px = bw * W
            h_px = bh * H
            bboxes = [[float(x[i]), float(y[i]), float(w_px[i]), float(h_px[i])]
                      for i in range(len(targets))]
            categories = targets[:, 0].astype(np.int64).tolist()
        else:
            bboxes, categories = [], []

        # BGR -> RGB
        img_rgb = image[:, :, ::-1].copy()
        out = self.pipeline(image=img_rgb, bboxes=bboxes, category=categories)
        aug_img = out["image"]     # HWC RGB uint8
        aug_boxes = out["bboxes"]  # list of [x, y, w, h] COCO pixel
        aug_cats = out["category"] # list of int

        # Format as COCO annotations dict for image_processor
        annotations = []
        for bbox, cat_id in zip(aug_boxes, aug_cats):
            annotations.append({
                "category_id": int(cat_id),
                "bbox": bbox,  # COCO [x, y, w, h] pixel
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })

        formatted = {"image_id": 0, "annotations": annotations}
        result = self.image_processor(
            images=aug_img, annotations=formatted, return_tensors="pt",
        )

        pixel_values = result["pixel_values"][0]  # squeeze batch dim
        labels = result["labels"][0]  # full dict: class_labels, boxes, size, image_id, area, iscrowd, orig_size
        return pixel_values, labels


class TorchvisionWithProcessorTransform:
    """torchvision v2 aug + HF image_processor.

    1. Convert to v2 sample, run v2 pipeline (NO Normalize -- processor handles rescaling)
    2. Extract from v2 sample: uint8 RGB numpy + COCO pixel bboxes
    3. Call image_processor(images=..., annotations=...) -> resize + rescale + bbox convert

    Returns (pixel_values_tensor, hf_labels_dict).
    """

    def __init__(self, v2_pipeline, image_processor, canvas_size):
        self.pipeline = v2_pipeline
        self.image_processor = image_processor
        self.canvas_size = canvas_size

    def __call__(self, image, targets):
        original_size = (image.shape[0], image.shape[1])

        # Convert to v2 sample (BGR->RGB, YOLO norm cxcywh->pixel XYXY)
        sample = _to_v2_sample(image, targets, original_size)
        sample = self.pipeline(sample)

        # Extract from v2 sample
        img_tensor = sample["image"]  # (3, H, W) float32 [0,1]
        boxes_v2 = sample["boxes"]    # tv_tensors BoundingBoxes XYXY
        labels_v2 = sample["labels"]  # int64 class IDs

        # Convert image to uint8 RGB numpy
        img_uint8 = (img_tensor * 255).to(torch.uint8)
        img_rgb = img_uint8.permute(1, 2, 0).numpy()  # HWC RGB

        # Convert v2 XYXY boxes to COCO pixel xywh
        if len(boxes_v2) > 0:
            xyxy = boxes_v2.float()
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            coco_boxes = np.stack([
                x1.numpy(), y1.numpy(),
                (x2 - x1).numpy(), (y2 - y1).numpy(),
            ], axis=1).tolist()
            cats = labels_v2.numpy().tolist()
        else:
            coco_boxes, cats = [], []

        # Format as COCO annotations dict
        annotations = []
        for bbox, cat_id in zip(coco_boxes, cats):
            annotations.append({
                "category_id": int(cat_id),
                "bbox": [float(b) for b in bbox],
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
            })

        formatted = {"image_id": 0, "annotations": annotations}
        result = self.image_processor(
            images=img_rgb, annotations=formatted, return_tensors="pt",
        )

        pixel_values = result["pixel_values"][0]
        labels = result["labels"][0]  # full dict with class_labels, boxes, size, etc.
        return pixel_values, labels


# ---------------------------------------------------------------------------
# Custom v2.Transform subclasses
# ---------------------------------------------------------------------------

class Mosaic(v2.Transform):
    """4-image mosaic augmentation as a v2.Transform.

    Combines four images into a 2x2 grid with a random centre point.
    Requires a dataset reference set via set_dataset().
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (640, 640),
        dataset: Any = None,
        border_value: int = 114,
    ) -> None:
        super().__init__()
        self.h, self.w = input_size
        self.dataset = dataset
        self.border_value = border_value
        self._clamp = v2.ClampBoundingBoxes()
        self._sanitize = v2.SanitizeBoundingBoxes(min_area=1.0)

    def set_dataset(self, dataset: Any) -> None:
        self.dataset = dataset

    def _get_image_tensor(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get image as CHW RGB uint8 tensor + boxes + labels from dataset."""
        raw = self.dataset.get_raw_item(idx)
        img_bgr, tgt_np = raw["image"], raw["targets"]
        ih, iw = img_bgr.shape[:2]
        img_rgb = torch.from_numpy(img_bgr[:, :, ::-1].copy()).permute(2, 0, 1)

        if len(tgt_np) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            cx = tgt_np[:, 1] * iw
            cy = tgt_np[:, 2] * ih
            bw = tgt_np[:, 3] * iw
            bh = tgt_np[:, 4] * ih
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes = torch.from_numpy(
                np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            )
            labels = torch.from_numpy(tgt_np[:, 0].astype(np.int64))

        return img_rgb, boxes, labels

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, dict):
            return sample
        return self._do_mosaic(sample)

    def _do_mosaic(self, sample):
        if self.dataset is None:
            return sample

        img0 = sample["image"]
        boxes0 = sample["boxes"]
        labels0 = sample["labels"]

        yc = int(random.uniform(self.h * 0.5, self.h * 1.5))
        xc = int(random.uniform(self.w * 0.5, self.w * 1.5))

        mosaic_h, mosaic_w = self.h * 2, self.w * 2
        canvas = torch.full((3, mosaic_h, mosaic_w), self.border_value, dtype=torch.uint8)

        all_boxes = []
        all_labels = []

        images_data = [(img0, boxes0, labels0)]
        for _ in range(3):
            idx = int(random.randint(0, len(self.dataset) - 1))
            images_data.append(self._get_image_tensor(idx))

        for i, (img_i, boxes_i, labels_i) in enumerate(images_data):
            _, ih, iw = img_i.shape

            if i == 0:
                x1d = max(xc - iw, 0)
                y1d = max(yc - ih, 0)
                x2d = xc
                y2d = yc
            elif i == 1:
                x1d = xc
                y1d = max(yc - ih, 0)
                x2d = min(xc + iw, mosaic_w)
                y2d = yc
            elif i == 2:
                x1d = max(xc - iw, 0)
                y1d = yc
                x2d = xc
                y2d = min(yc + ih, mosaic_h)
            else:
                x1d = xc
                y1d = yc
                x2d = min(xc + iw, mosaic_w)
                y2d = min(yc + ih, mosaic_h)

            pw = x2d - x1d
            ph = y2d - y1d
            if pw <= 0 or ph <= 0:
                continue

            if i == 0:
                x1s, y1s = iw - pw, ih - ph
            elif i == 1:
                x1s, y1s = 0, ih - ph
            elif i == 2:
                x1s, y1s = iw - pw, 0
            else:
                x1s, y1s = 0, 0

            canvas[:, y1d:y2d, x1d:x2d] = img_i[:, y1s:y1s + ph, x1s:x1s + pw]

            if len(boxes_i) > 0:
                b = boxes_i.clone().float()
                b[:, 0] = b[:, 0] - x1s + x1d
                b[:, 1] = b[:, 1] - y1s + y1d
                b[:, 2] = b[:, 2] - x1s + x1d
                b[:, 3] = b[:, 3] - y1s + y1d
                all_boxes.append(b)
                all_labels.append(labels_i)

        # Crop to output size
        crop_x1 = max(xc - self.w // 2, 0)
        crop_y1 = max(yc - self.h // 2, 0)
        crop_x1 = min(crop_x1, mosaic_w - self.w)
        crop_y1 = min(crop_y1, mosaic_h - self.h)
        crop_x2 = crop_x1 + self.w
        crop_y2 = crop_y1 + self.h

        out_img = canvas[:, crop_y1:crop_y2, crop_x1:crop_x2].clone()

        if all_boxes:
            merged_boxes = torch.cat(all_boxes, dim=0)
            merged_labels = torch.cat(all_labels, dim=0)
            merged_boxes[:, 0] -= crop_x1
            merged_boxes[:, 1] -= crop_y1
            merged_boxes[:, 2] -= crop_x1
            merged_boxes[:, 3] -= crop_y1

            out_boxes = tv_tensors.BoundingBoxes(
                merged_boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(self.h, self.w),
            )
            out_labels = merged_labels
        else:
            out_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(self.h, self.w),
            )
            out_labels = torch.zeros(0, dtype=torch.int64)

        result = {
            "image": tv_tensors.Image(out_img),
            "boxes": out_boxes,
            "labels": out_labels,
        }

        # Sanitize after mosaic
        result = self._clamp(result)
        result = self._sanitize(result)

        return result

    def __repr__(self) -> str:
        return f"Mosaic(input_size=({self.h}, {self.w}))"


class MixUp(v2.Transform):
    """MixUp augmentation: alpha-blend two images and concatenate labels."""

    def __init__(self, dataset: Any = None, alpha: float = 1.5) -> None:
        super().__init__()
        self.dataset = dataset
        self.alpha = alpha

    def set_dataset(self, dataset: Any) -> None:
        self.dataset = dataset

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, dict):
            return sample

        if self.dataset is None:
            return sample

        img1 = sample["image"]
        boxes1 = sample["boxes"]
        labels1 = sample["labels"]

        idx2 = int(random.randint(0, len(self.dataset) - 1))
        raw2 = self.dataset.get_raw_item(idx2)
        img2_bgr, tgt2_np = raw2["image"], raw2["targets"]
        img2 = torch.from_numpy(img2_bgr[:, :, ::-1].copy()).permute(2, 0, 1)

        _, h1, w1 = img1.shape
        if img2.shape[1] != h1 or img2.shape[2] != w1:
            img2 = F.resize(img2, [h1, w1], antialias=True)

        lam = float(np.random.beta(self.alpha, self.alpha))
        lam = max(lam, 1 - lam)

        blended = (img1.float() * lam + img2.float() * (1 - lam)).to(torch.uint8)

        if len(tgt2_np) == 0:
            boxes2 = torch.zeros((0, 4), dtype=torch.float32)
            labels2 = torch.zeros(0, dtype=torch.int64)
        else:
            cx = tgt2_np[:, 1] * w1
            cy = tgt2_np[:, 2] * h1
            bw = tgt2_np[:, 3] * w1
            bh = tgt2_np[:, 4] * h1
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes2 = torch.from_numpy(
                np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            )
            labels2 = torch.from_numpy(tgt2_np[:, 0].astype(np.int64))

        if len(boxes1) > 0 and len(boxes2) > 0:
            merged_boxes = torch.cat([boxes1.float(), boxes2], dim=0)
            merged_labels = torch.cat([labels1, labels2], dim=0)
        elif len(boxes2) > 0:
            merged_boxes = boxes2
            merged_labels = labels2
        else:
            merged_boxes = boxes1.float() if len(boxes1) > 0 else torch.zeros((0, 4), dtype=torch.float32)
            merged_labels = labels1 if len(labels1) > 0 else torch.zeros(0, dtype=torch.int64)

        out_boxes = tv_tensors.BoundingBoxes(
            merged_boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h1, w1),
        )

        return {
            "image": tv_tensors.Image(blended),
            "boxes": out_boxes,
            "labels": merged_labels,
        }

    def __repr__(self) -> str:
        return f"MixUp(alpha={self.alpha})"


class CopyPaste(v2.Transform):
    """Copy-paste augmentation: crops objects from source and pastes via tensor slicing."""

    def __init__(self, p: float = 0.5, max_objects: int = 3) -> None:
        super().__init__()
        self.p = p
        self.max_objects = max_objects
        self._dataset = None

    def set_dataset(self, dataset: Any) -> None:
        self._dataset = dataset

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs
        if not isinstance(sample, dict):
            return sample

        if self._dataset is None or random.random() > self.p:
            return sample

        img = sample["image"].clone()
        boxes = sample["boxes"]
        labels = sample["labels"]
        _, h, w = img.shape

        idx = int(random.randint(0, len(self._dataset) - 1))
        raw = self._dataset.get_raw_item(idx)
        src_bgr, src_tgt = raw["image"], raw["targets"]
        if len(src_tgt) == 0:
            return sample

        src_h, src_w = src_bgr.shape[:2]
        src_img = torch.from_numpy(src_bgr[:, :, ::-1].copy()).permute(2, 0, 1)

        n_paste = min(self.max_objects, len(src_tgt))
        chosen = np.random.choice(len(src_tgt), size=n_paste, replace=False)

        new_boxes = []
        new_labels = []

        for i in chosen:
            cls_id, cx, cy, bw, bh = src_tgt[i]
            sx1 = max(int((cx - bw / 2) * src_w), 0)
            sy1 = max(int((cy - bh / 2) * src_h), 0)
            sx2 = min(int((cx + bw / 2) * src_w), src_w)
            sy2 = min(int((cy + bh / 2) * src_h), src_h)

            crop_w = sx2 - sx1
            crop_h = sy2 - sy1
            if crop_w <= 0 or crop_h <= 0:
                continue

            crop = src_img[:, sy1:sy2, sx1:sx2]

            dx1 = int(random.randint(0, max(w - crop_w, 1) - 1)) if w > crop_w else 0
            dy1 = int(random.randint(0, max(h - crop_h, 1) - 1)) if h > crop_h else 0
            dx2 = min(dx1 + crop_w, w)
            dy2 = min(dy1 + crop_h, h)

            paste_w = dx2 - dx1
            paste_h = dy2 - dy1
            if paste_w <= 0 or paste_h <= 0:
                continue

            img[:, dy1:dy2, dx1:dx2] = crop[:, :paste_h, :paste_w]

            new_boxes.append(torch.tensor(
                [float(dx1), float(dy1), float(dx2), float(dy2)],
                dtype=torch.float32,
            ))
            new_labels.append(int(cls_id))

        if new_boxes:
            extra_boxes = torch.stack(new_boxes, dim=0)
            extra_labels = torch.tensor(new_labels, dtype=torch.int64)
            if len(boxes) > 0:
                all_boxes = torch.cat([boxes.float(), extra_boxes], dim=0)
                all_labels = torch.cat([labels, extra_labels], dim=0)
            else:
                all_boxes = extra_boxes
                all_labels = extra_labels
        else:
            all_boxes = boxes.float() if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
            all_labels = labels if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)

        out_boxes = tv_tensors.BoundingBoxes(
            all_boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )

        return {
            "image": tv_tensors.Image(img),
            "boxes": out_boxes,
            "labels": all_labels,
        }

    def __repr__(self) -> str:
        return f"CopyPaste(p={self.p}, max_objects={self.max_objects})"


class IRSimulation(v2.Transform):
    """Simulate IR/B&W camera mode and low-light conditions using pure PyTorch ops."""

    def __init__(
        self,
        ir_prob: float = 0.3,
        low_light_prob: float = 0.2,
        noise_sigma: float = 15,
    ) -> None:
        super().__init__()
        self.ir_prob = ir_prob
        self.low_light_prob = low_light_prob
        self.noise_sigma = noise_sigma

    def forward(self, *inputs):
        sample = inputs[0] if len(inputs) == 1 else inputs

        if isinstance(sample, dict):
            img = sample["image"]
            # Handle both uint8 [0,255] (legacy) and float32 [0,1]
            # (resize-first pipeline) — constants (ir offset `+ 10`,
            # `noise_sigma=15`) are uint8-scale; scale down by 255 on float.
            is_float = img.is_floating_point()
            max_v = 1.0 if is_float else 255.0
            unit = 1.0 / 255.0 if is_float else 1.0

            if random.random() < self.ir_prob:
                gray = F.rgb_to_grayscale(img, num_output_channels=1)
                img = gray.repeat(3, 1, 1) if img.ndim == 3 else gray.repeat(1, 3, 1, 1)
                img = torch.clamp(img.float() * 1.3 + 10 * unit, 0, max_v).to(img.dtype)

            if random.random() < self.low_light_prob:
                factor = random.uniform(0.1, 0.4)
                img_f = img.float() * factor
                if self.noise_sigma > 0:
                    noise = torch.randn_like(img_f) * (self.noise_sigma * unit)
                    img_f = img_f + noise
                img = torch.clamp(img_f, 0, max_v).to(sample["image"].dtype)

            sample = dict(sample)
            sample["image"] = tv_tensors.Image(img)
            return sample

        return sample

    def __repr__(self) -> str:
        return (
            f"IRSimulation(ir_prob={self.ir_prob}, "
            f"low_light_prob={self.low_light_prob}, "
            f"noise_sigma={self.noise_sigma})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_transforms(
    config: dict,
    is_train: bool = True,
    input_size: tuple[int, int] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    image_processor: Any = None,
):
    """Build a transform pipeline from an augmentation config dict.

    Args:
        config: The ``augmentation`` section of a training YAML config.
            Expected keys (all optional, with sensible defaults):
            ``mosaic``, ``mixup``, ``hsv_h``, ``hsv_s``, ``hsv_v``,
            ``fliplr``, ``flipud``, ``scale``, ``degrees``, ``translate``,
            ``shear``, ``copypaste``, ``ir_simulation``.
            If ``library: albumentations`` is set, dispatches to
            :func:`build_albumentations_transforms` instead — faster CPU path
            that matches qubvel's reference notebook semantically.
        is_train: If True build training pipeline (with augmentations);
            if False build a minimal val/test pipeline (resize + normalise).
        input_size: ``(height, width)``. Required.
        mean: Normalisation mean. Defaults to ImageNet values.
        std: Normalisation std. Defaults to ImageNet values.

    Returns:
        A callable: either :class:`DetectionTransform` (torchvision v2) or
        :class:`AlbumentationsDetectionTransform`.
    """
    if input_size is None:
        raise ValueError("input_size is required for build_transforms")

    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    # Dispatch to the Albumentations backend when explicitly requested.
    if config.get("library", "").lower() == "albumentations":
        return build_albumentations_transforms(
            config=config, is_train=is_train,
            input_size=input_size, mean=mean, std=std,
            image_processor=image_processor,
        )

    transforms: list[Any] = []

    # Resize + ToDtype are always present; in train mode they run BEFORE
    # the per-sample augmentations so color jitter + perspective operate
    # on the smaller float32 480² tensor instead of the original uint8
    # image. Measured ~4× speedup on ColorJitter(HSV) and Perspective
    # (see `notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/
    # README.md`). Mosaic/MixUp/CopyPaste are dataset-level ops that
    # produce input_size uint8 output, so Resize after them is a cheap
    # no-op; they must stay at the head of the list.
    from torchvision.transforms import InterpolationMode
    _resize = v2.Resize(
        size=list(input_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=config.get("resize_antialias", False),
    )
    _to_float = v2.ToDtype(torch.float32, scale=True)

    if is_train:
        # Mosaic
        if config.get("mosaic", False):
            transforms.append(Mosaic(input_size=input_size))

        # MixUp
        if config.get("mixup", False):
            transforms.append(MixUp())

        # CopyPaste
        if config.get("copypaste", False):
            transforms.append(CopyPaste(
                p=config.get("copypaste_p", 0.5),
                max_objects=config.get("copypaste_max", 3),
            ))

        # Resize + ToDtype before the expensive per-sample augmentations.
        # After this point the image is float32 in [0, 1] at input_size.
        transforms.append(_resize)
        transforms.append(_to_float)

        # RandomAffine. Skipped entirely when all params are identity
        # (avoids per-image interpolation + tv_tensor dispatch for a no-op).
        # Otherwise always applied by default (legacy behaviour); set
        # `affine_p < 1.0` in the config to gate it probabilistically like
        # Albumentations' `A.Affine(p=...)`.
        scale = config.get("scale", [0.5, 1.5])
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            scale_range = tuple(scale)
        else:
            scale_range = (0.5, 1.5)

        degrees = config.get("degrees", 0.0)
        translate_val = config.get("translate", 0.0)
        shear_val = config.get("shear", 0.0)

        affine_is_identity = (
            scale_range == (1.0, 1.0)
            and degrees == 0.0
            and translate_val == 0.0
            and shear_val == 0.0
        )
        # fill is now in [0, 1] float space (114/255 ≈ 0.447) to match the
        # post-ToDtype image range.
        _fill_f = 114.0 / 255.0
        if not affine_is_identity:
            affine_tfm = v2.RandomAffine(
                degrees=degrees,
                translate=(translate_val, translate_val) if translate_val > 0 else None,
                scale=scale_range,
                shear=(-shear_val, shear_val, -shear_val, shear_val) if shear_val > 0 else None,
                fill=_fill_f,
            )
            affine_p = config.get("affine_p", 1.0)
            if affine_p < 1.0:
                transforms.append(v2.RandomApply([affine_tfm], p=affine_p))
            else:
                transforms.append(affine_tfm)

        # Perspective — matches Albumentations' `A.Perspective(p=perspective_p)`.
        # Opt-in via `perspective_p > 0` (default off preserves legacy behavior).
        perspective_p = config.get("perspective_p", 0.0)
        if perspective_p > 0:
            transforms.append(v2.RandomPerspective(
                distortion_scale=config.get("perspective_distortion", 0.2),
                p=perspective_p,
                fill=_fill_f,
            ))

        # Clamp + sanitize after spatial transforms.
        # `min_bbox_area` (default 25 pixels squared) matches qubvel's
        # reference Albumentations recipe (`A.BboxParams(min_area=25, ...)`);
        # the threshold now evaluates against the resized canvas (same as
        # Albumentations, where `A.Resize` runs before `BboxParams` clips).
        min_bbox_area = float(config.get("min_bbox_area", 25.0))
        transforms.append(v2.ClampBoundingBoxes())
        transforms.append(v2.SanitizeBoundingBoxes(min_area=min_bbox_area))

        # Colour augmentation has two schemas:
        #   (A) *Legacy* (default when no `_p` keys set): a single
        #       `v2.ColorJitter(hue=hsv_h, saturation=hsv_s, brightness=hsv_v)`
        #       applied *every step*. Used by fire/helmet/shoes/etc configs.
        #   (B) *Albumentations-compatible* (opt-in via `brightness_contrast_p`
        #       or `hsv_p`): split into two `v2.RandomApply` gates matching
        #       qubvel's reference recipe —
        #       `A.RandomBrightnessContrast(p=0.5)` + `A.HueSaturationValue(p=0.1)`.
        #       Magnitudes and probabilities are independent, so both gates
        #       can fire on the same image (same as Albumentations chaining).
        bc_p = config.get("brightness_contrast_p", 0.0)
        hsv_p = config.get("hsv_p", 0.0)

        if bc_p > 0 or hsv_p > 0:
            # Defaults below mirror Albumentations' standard limits so
            # configs can enable a gate (e.g. `brightness_contrast_p: 0.5`)
            # and rely on sensible magnitudes without listing each one.
            if bc_p > 0:
                transforms.append(v2.RandomApply(
                    [v2.ColorJitter(
                        brightness=config.get("brightness", 0.2),
                        contrast=config.get("contrast", 0.2),
                    )],
                    p=bc_p,
                ))
            if hsv_p > 0:
                transforms.append(v2.RandomApply(
                    [v2.ColorJitter(
                        hue=config.get("hsv_h", 0.015),
                        saturation=config.get("hsv_s", 0.2),
                        brightness=config.get("hsv_v", 0.1),
                    )],
                    p=hsv_p,
                ))
        else:
            # Legacy: single always-applied ColorJitter. Back-compat for
            # every existing feature config that uses hsv_* as magnitudes.
            hsv_h = config.get("hsv_h", 0.0)
            hsv_s = config.get("hsv_s", 0.0)
            hsv_v = config.get("hsv_v", 0.0)
            if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
                transforms.append(v2.ColorJitter(
                    hue=hsv_h,
                    saturation=hsv_s,
                    brightness=hsv_v,
                ))

        # IR simulation — dtype-aware (handles both uint8 and float inputs).
        if config.get("ir_simulation", False):
            transforms.append(IRSimulation(
                ir_prob=config.get("ir_prob", 0.3),
                low_light_prob=config.get("low_light_prob", 0.2),
                noise_sigma=config.get("noise_sigma", 15),
            ))

        # Flips
        fliplr = config.get("fliplr", 0.0)
        if fliplr > 0:
            transforms.append(v2.RandomHorizontalFlip(p=fliplr))

        flipud = config.get("flipud", 0.0)
        if flipud > 0:
            transforms.append(v2.RandomVerticalFlip(p=flipud))
    else:
        # Eval: no augmentations — just resize + ToDtype + Normalize.
        transforms.append(_resize)
        transforms.append(_to_float)

    # HF detection processors (D-FINE, RT-DETRv2) expect [0, 1] rescaled
    # inputs with do_normalize=False — ImageNet normalize degrades their
    # pretrained-feature distribution. Also YOLOX Megvii weights expect
    # [0, 255] raw. Respect `augmentation.normalize` to opt out.
    if config.get("normalize", True) and image_processor is None:
        transforms.append(v2.Normalize(mean=mean, std=std))

    # When an HF image_processor is provided, wrap the v2 pipeline so
    # resize/rescale/bbox-convert go through the processor (qubvel pattern).
    # The Normalize step is skipped — the processor handles rescaling.
    if image_processor is not None:
        return TorchvisionWithProcessorTransform(
            v2_pipeline=v2.Compose(transforms),
            image_processor=image_processor,
            canvas_size=input_size,
        )

    return DetectionTransform(transforms, canvas_size=input_size)


def build_albumentations_transforms(
    config: dict,
    is_train: bool = True,
    input_size: tuple[int, int] = (640, 640),
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    image_processor: Any = None,
):
    """Albumentations-backed pipeline matching our torchvision v2 semantics.

    Same config keys as :func:`build_transforms` (with the same defaults when
    only the `_p` gates are set). Selected via `augmentation.library: albumentations`.
    Mosaic/MixUp/CopyPaste/IRSimulation are **not** supported here — they're
    dataset-level ops tied to our YOLOXDataset and the reference recipe
    doesn't use them. Bounding boxes flow through in COCO pixel xywh format
    inside the pipeline (Albumentations native format).

    Intended primarily for DETR-family reproduction runs where qubvel's
    reference uses Albumentations — gives us speed parity with the reference
    (~14s/epoch on CPPE-5 vs ~28s for v2 with same aug set).
    """
    import albumentations as A

    H_in, W_in = int(input_size[0]), int(input_size[1])
    normalize = bool(config.get("normalize", True))

    tfms: list[Any] = []
    if is_train:
        perspective_p = config.get("perspective_p", 0.0)
        if perspective_p > 0:
            tfms.append(A.Perspective(
                scale=(0.05, config.get("perspective_distortion", 0.2)),
                pad_val=114, p=perspective_p,
            ))

        # Affine — the torchvision-v2 branch of `build_transforms` always
        # applies RandomAffine when called. Mirror that here (skip if all
        # magnitudes are identity, which is our CPPE-5 case).
        degrees = config.get("degrees", 0.0)
        translate_val = config.get("translate", 0.0)
        shear_val = config.get("shear", 0.0)
        scale_cfg = config.get("scale", [1.0, 1.0])
        scale_range = tuple(scale_cfg) if isinstance(scale_cfg, (list, tuple)) and len(scale_cfg) == 2 else (1.0, 1.0)
        affine_active = (
            degrees > 0 or translate_val > 0 or shear_val > 0
            or scale_range != (1.0, 1.0)
        )
        if affine_active:
            tfms.append(A.Affine(
                rotate=(-degrees, degrees) if degrees > 0 else 0,
                translate_percent={"x": (-translate_val, translate_val), "y": (-translate_val, translate_val)} if translate_val > 0 else None,
                scale=scale_range if scale_range != (1.0, 1.0) else 1.0,
                shear={"x": (-shear_val, shear_val), "y": (-shear_val, shear_val)} if shear_val > 0 else 0,
                fill=114,
                p=config.get("affine_p", 1.0),
            ))

        fliplr = config.get("fliplr", 0.0)
        if fliplr > 0:
            tfms.append(A.HorizontalFlip(p=fliplr))
        flipud = config.get("flipud", 0.0)
        if flipud > 0:
            tfms.append(A.VerticalFlip(p=flipud))

        bc_p = config.get("brightness_contrast_p", 0.0)
        if bc_p > 0:
            tfms.append(A.RandomBrightnessContrast(
                brightness_limit=config.get("brightness", 0.2),
                contrast_limit=config.get("contrast", 0.2),
                p=bc_p,
            ))
        hsv_p = config.get("hsv_p", 0.0)
        if hsv_p > 0:
            # Albumentations uses additive shifts in [0, 180] for hue and
            # [0, 255] for sat/val. Map our torchvision-style magnitudes
            # (fraction of full range) to those scales.
            hue = int(round(config.get("hsv_h", 0.015) * 180))      # ~3 → ±3°
            sat = int(round(config.get("hsv_s", 0.2) * 255))        # ~51
            val = int(round(config.get("hsv_v", 0.1) * 255))        # ~25
            tfms.append(A.HueSaturationValue(
                hue_shift_limit=hue,
                sat_shift_limit=sat,
                val_shift_limit=val,
                p=hsv_p,
            ))

        # Legacy ColorJitter — applied always, magnitudes only. Kept for
        # configs that don't set any `_p` gate but do set hsv_*.
        if bc_p <= 0 and hsv_p <= 0:
            hsv_h = config.get("hsv_h", 0.0)
            hsv_s = config.get("hsv_s", 0.0)
            hsv_v = config.get("hsv_v", 0.0)
            if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
                hue = int(round(hsv_h * 180))
                sat = int(round(hsv_s * 255))
                val = int(round(hsv_v * 255))
                tfms.append(A.HueSaturationValue(
                    hue_shift_limit=hue, sat_shift_limit=sat, val_shift_limit=val, p=1.0,
                ))

    # When image_processor is provided, skip A.Resize — the processor
    # handles resize + rescale + bbox conversion atomically (qubvel pattern).
    # `min_bbox_area` default 25 matches qubvel's reference (drops sub-5x5
    # boxes after augmentation — noise for the matcher).
    min_bbox_area = float(config.get("min_bbox_area", 25.0))

    if image_processor is not None:
        # Albumentations-only pipeline (no resize — processor does it)
        pipeline = A.Compose(
            tfms,
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category"],
                clip=True,
                min_area=min_bbox_area,
                min_width=1,
                min_height=1,
            ),
        )
        return AlbumentationsWithProcessorTransform(
            albu_pipeline=pipeline,
            image_processor=image_processor,
        )

    # Always applied (train + val): resize to input_size.
    tfms.append(A.Resize(height=H_in, width=W_in, p=1.0))

    # Note: normalization is applied in AlbumentationsDetectionTransform.__call__
    # instead of via A.Normalize, so we can preserve the same `normalize: false`
    # semantics (feed [0, 1] raw pixels, let HF image processor do per-batch norm).
    pipeline = A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category"],
            clip=True,
            min_area=min_bbox_area,
            min_width=1,
            min_height=1,
        ),
    )
    return AlbumentationsDetectionTransform(
        albu_pipeline=pipeline,
        canvas_size=(H_in, W_in),
        normalize=normalize,
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------------------
# GPU-side augmentation helpers
# ---------------------------------------------------------------------------

def _rgb_to_hsv(images: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB [0, 1] to HSV [0, 1]."""
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    maxc, maxc_idx = images.max(dim=1)
    minc = images.min(dim=1).values
    delta = (maxc - minc).clamp(min=1e-8)
    s = torch.where(maxc > 1e-8, (maxc - minc) / maxc, torch.zeros_like(maxc))
    rc = (g - b) / delta
    gc = 2.0 + (b - r) / delta
    bc = 4.0 + (r - g) / delta
    h = torch.where(maxc_idx == 0, rc, torch.where(maxc_idx == 1, gc, bc))
    h = (h / 6.0) % 1.0
    h = torch.where((maxc - minc) > 1e-8, h, torch.zeros_like(h))
    return torch.stack([h, s, maxc], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) HSV [0, 1] to RGB [0, 1]."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h6 = h * 6.0
    i = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    # Stack 6 sector values per channel: indices 0-5 correspond to hue sectors
    r_vals = torch.stack([v, q, p, p, t, v], dim=1)  # (B, 6, H, W)
    g_vals = torch.stack([t, v, v, q, p, p], dim=1)
    b_vals = torch.stack([p, p, t, v, v, q], dim=1)
    idx = i.unsqueeze(1)  # (B, 1, H, W)
    r = torch.gather(r_vals, 1, idx).squeeze(1)
    g = torch.gather(g_vals, 1, idx).squeeze(1)
    b = torch.gather(b_vals, 1, idx).squeeze(1)
    return torch.stack([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# GPU-side augmentation
# ---------------------------------------------------------------------------


class GpuDetectionTransform:
    """Fully-vectorised detection augmentations on GPU batches.

    All B images are warped in a single CUDA call via F.affine_grid +
    F.grid_sample (one kernel launch vs B sequential launches). ColorJitter
    (brightness / saturation / hue) runs as batched tensor math with no
    per-image Python overhead. Flips are one CUDA call each.

    Affine box transformation: M_fwd = inv(M_inv) maps 4 input corners to
    output space; new bbox is the axis-aligned envelope of the transformed
    corners, clamped and filtered by area >= 1 px.

    Hue/saturation use a vectorised RGB↔HSV pipeline (torch.gather-based
    sector lookup — no loops).
    """

    def __init__(
        self,
        degrees: float,
        translate: tuple[float, float] | None,
        scale_range: tuple[float, float],
        shear: tuple[float, float, float, float] | None,
        fill: float,
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        contrast: float,
        flip_h_p: float,
        flip_v_p: float,
        mean: Sequence[float],
        std: Sequence[float],
        input_size: tuple[int, int],
        normalize: bool = True,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear
        self.fill = fill
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.contrast = contrast
        self.flip_h_p = flip_h_p
        self.flip_v_p = flip_v_p
        self.normalize = normalize
        self._mean = torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1)
        self._std = torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1)
        self.input_size = input_size  # (H, W)

    def __call__(
        self,
        images: torch.Tensor,
        targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply GPU augmentations to a collated batch.

        Args:
            images: (B, C, H, W) float32 [0, 1] on GPU.
            targets: List of B tensors, each (N_i, 5) normalized CXCYWH on GPU.

        Returns:
            (images, targets) augmented, with Normalize applied to images.
        """
        H, W = self.input_size  # target spatial dims from config
        B, _C, img_H, img_W = images.shape
        device = images.device

        # GPU letterbox resize — scale uniformly to fit (H, W), pad remainder with fill
        # All images in the batch share the same size (torch.stack guarantee), so one
        # scale/padding applies to the whole batch.
        if img_H != H or img_W != W:
            scale = min(H / img_H, W / img_W)
            new_H, new_W = round(img_H * scale), round(img_W * scale)
            images = torch.nn.functional.interpolate(
                images, size=(new_H, new_W), mode="bilinear", antialias=True
            )
            # Centre the scaled image inside the target canvas
            pad_top = (H - new_H) // 2
            pad_left = (W - new_W) // 2
            images = torch.nn.functional.pad(
                images,
                (pad_left, W - new_W - pad_left, pad_top, H - new_H - pad_top),
                value=self.fill,
            )
            # Adjust boxes: normalized coords shift with scale + padding
            scale_w, scale_h = new_W / W, new_H / H
            off_x, off_y = pad_left / W, pad_top / H
            new_targets: list[torch.Tensor] = []
            for tgt in targets:
                if len(tgt) == 0:
                    new_targets.append(tgt)
                    continue
                t = tgt.clone()
                t[:, 1] = t[:, 1] * scale_w + off_x  # cx
                t[:, 2] = t[:, 2] * scale_h + off_y  # cy
                t[:, 3] = t[:, 3] * scale_w           # bw
                t[:, 4] = t[:, 4] * scale_h           # bh
                new_targets.append(t)
            targets = new_targets

        # --- Affine: one CUDA warp for all B images ---
        angles = torch.empty(B, device=device).uniform_(-self.degrees, self.degrees)
        if self.translate is not None:
            tx = torch.empty(B, device=device).uniform_(-self.translate[0] * W, self.translate[0] * W)
            ty = torch.empty(B, device=device).uniform_(-self.translate[1] * H, self.translate[1] * H)
        else:
            tx = torch.zeros(B, device=device)
            ty = torch.zeros(B, device=device)
        scales = torch.empty(B, device=device).uniform_(self.scale_range[0], self.scale_range[1])
        if self.shear is not None:
            shear_x = torch.empty(B, device=device).uniform_(self.shear[0], self.shear[1])
            shear_y = torch.empty(B, device=device).uniform_(self.shear[2], self.shear[3])
        else:
            shear_x = torch.zeros(B, device=device)
            shear_y = torch.zeros(B, device=device)

        M_inv, theta = _build_affine_theta(angles, tx, ty, scales, shear_x, shear_y, H, W)

        # Shift by fill so zero-padding outside the warp becomes fill
        shifted = images - self.fill
        grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=False)
        warped = torch.nn.functional.grid_sample(
            shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        images = (warped + self.fill).clamp(0.0, 1.0)

        # Transform boxes via forward map M_fwd = inv(M_inv)
        M_fwd = torch.linalg.inv(M_inv)  # (B, 3, 3)
        targets = _transform_boxes(targets, M_fwd, H, W)

        # --- ColorJitter: randomized order, fully vectorized ---
        color_ops: list[str] = []
        if self.hsv_v > 0:
            color_ops.append("brightness")
        if self.contrast > 0:
            color_ops.append("contrast")
        if self.hsv_s > 0 or self.hsv_h > 0:
            color_ops.append("sat_hue")
        random.shuffle(color_ops)

        for op in color_ops:
            if op == "brightness":
                factors = torch.empty(B, device=device).uniform_(
                    max(0.0, 1.0 - self.hsv_v), 1.0 + self.hsv_v
                )
                images = (images * factors.view(B, 1, 1, 1)).clamp(0.0, 1.0)
            elif op == "contrast":
                luma = 0.2126 * images[:, 0] + 0.7152 * images[:, 1] + 0.0722 * images[:, 2]
                mean_luma = luma.mean(dim=(-2, -1)).view(B, 1, 1, 1)
                factors = torch.empty(B, device=device).uniform_(
                    max(0.0, 1.0 - self.contrast), 1.0 + self.contrast
                )
                fv = factors.view(B, 1, 1, 1)
                images = (images * fv + mean_luma * (1.0 - fv)).clamp(0.0, 1.0)
            elif op == "sat_hue":
                hsv = _rgb_to_hsv(images)
                if self.hsv_s > 0:
                    s_factors = torch.empty(B, device=device).uniform_(
                        max(0.0, 1.0 - self.hsv_s), 1.0 + self.hsv_s
                    )
                    hsv[:, 1] = (hsv[:, 1] * s_factors.view(B, 1, 1)).clamp(0.0, 1.0)
                if self.hsv_h > 0:
                    h_shifts = torch.empty(B, device=device).uniform_(-self.hsv_h, self.hsv_h)
                    hsv[:, 0] = (hsv[:, 0] + h_shifts.view(B, 1, 1)) % 1.0
                images = _hsv_to_rgb(hsv).clamp(0.0, 1.0)

        # --- Flips: one CUDA call per axis ---
        if self.flip_h_p > 0:
            mask = torch.rand(B, device=device) < self.flip_h_p
            if mask.any():
                images[mask] = torch.flip(images[mask], dims=[-1])
                for i in mask.nonzero(as_tuple=True)[0]:
                    t = targets[i]
                    if len(t) > 0:
                        t = t.clone()
                        t[:, 1] = 1.0 - t[:, 1]  # cx_new = 1 - cx_old
                        targets[i] = t

        if self.flip_v_p > 0:
            mask = torch.rand(B, device=device) < self.flip_v_p
            if mask.any():
                images[mask] = torch.flip(images[mask], dims=[-2])
                for i in mask.nonzero(as_tuple=True)[0]:
                    t = targets[i]
                    if len(t) > 0:
                        t = t.clone()
                        t[:, 2] = 1.0 - t[:, 2]  # cy_new = 1 - cy_old
                        targets[i] = t

        if self.normalize:
            mean = self._mean.to(device)
            std = self._std.to(device)
            images = (images - mean) / std

        return images, targets

    def __repr__(self) -> str:
        return (
            f"GpuDetectionTransform(degrees={self.degrees}, scale={self.scale_range}, "
            f"hsv=({self.hsv_h},{self.hsv_s},{self.hsv_v}), "
            f"flip_h={self.flip_h_p}, flip_v={self.flip_v_p})"
        )


def _build_affine_theta(
    angles: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    scales: torch.Tensor,
    shear_x: torch.Tensor,
    shear_y: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B,3,3) pixel-space M_inv and (B,2,3) normalized theta.

    Uses torchvision's exact affine formula so box transforms match image warps.
    theta is for F.affine_grid(align_corners=False).
    """
    B = angles.shape[0]
    device = angles.device
    dtype = angles.dtype

    rot = torch.deg2rad(angles)
    sx = torch.deg2rad(shear_x)
    sy = torch.deg2rad(shear_y)
    cos_sy = torch.cos(sy).clamp(min=1e-7)

    a = torch.cos(rot - sy) / cos_sy
    b = -torch.cos(rot - sy) * torch.tan(sx) / cos_sy - torch.sin(rot)
    c = torch.sin(rot - sy) / cos_sy
    d = -torch.sin(rot - sy) * torch.tan(sx) / cos_sy + torch.cos(rot)
    a, b, c, d = scales * a, scales * b, scales * c, scales * d

    cx = W / 2.0
    cy = H / 2.0
    t03 = cx - a * cx - b * cy + tx
    t13 = cy - c * cx - d * cy + ty

    zeros = torch.zeros(B, device=device, dtype=dtype)
    ones = torch.ones(B, device=device, dtype=dtype)
    M_inv = torch.stack([
        torch.stack([a, b, t03], dim=1),
        torch.stack([c, d, t13], dim=1),
        torch.stack([zeros, zeros, ones], dim=1),
    ], dim=1)  # (B, 3, 3)

    # Convert pixel-space M_inv to normalized theta for affine_grid(align_corners=False)
    # theta = D_inv @ M_inv @ D
    # D: normalized→pixel: x_pixel = W/2*x_norm + W/2 - 0.5
    # D_inv: pixel→normalized: x_norm = 2/W*x_pixel - (W-1)/W
    D = torch.tensor([
        [W / 2.0, 0.0, W / 2.0 - 0.5],
        [0.0, H / 2.0, H / 2.0 - 0.5],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    D_inv = torch.tensor([
        [2.0 / W, 0.0, -(W - 1.0) / W],
        [0.0, 2.0 / H, -(H - 1.0) / H],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype).unsqueeze(0)  # (1, 3, 3)

    theta_3x3 = D_inv @ M_inv @ D  # (B, 3, 3)
    return M_inv, theta_3x3[:, :2, :]  # (B, 3, 3), (B, 2, 3)


def _transform_boxes(
    targets: list[torch.Tensor],
    M_fwd: torch.Tensor,
    H: int,
    W: int,
) -> list[torch.Tensor]:
    """Transform normalized CXCYWH boxes using forward affine matrices.

    Each box's 4 corners are mapped through M_fwd[i], the axis-aligned
    envelope is taken, clamped to [0,W]×[0,H], and boxes with area < 1px
    are dropped.
    """
    new_targets: list[torch.Tensor] = []
    for i, tgt in enumerate(targets):
        if len(tgt) == 0:
            new_targets.append(tgt)
            continue

        device = tgt.device
        cls_ids = tgt[:, 0]
        cx_px = tgt[:, 1] * W
        cy_px = tgt[:, 2] * H
        bw_px = tgt[:, 3] * W
        bh_px = tgt[:, 4] * H

        x1 = cx_px - bw_px / 2
        y1 = cy_px - bh_px / 2
        x2 = cx_px + bw_px / 2
        y2 = cy_px + bh_px / 2

        N = len(tgt)
        # (N, 4, 3): 4 corners (TL,TR,BL,BR) in homogeneous pixel coords
        corners = torch.stack([
            torch.stack([x1, x2, x1, x2], dim=1),
            torch.stack([y1, y1, y2, y2], dim=1),
            torch.ones(N, 4, device=device),
        ], dim=2)  # (N, 4, 3)

        # m: (2,3) — apply forward map to all corners at once
        m = M_fwd[i, :2, :]  # (2, 3)
        out_xy = (m @ corners.permute(0, 2, 1).reshape(3, N * 4)).reshape(2, N, 4).permute(1, 2, 0)

        new_x1 = out_xy[:, :, 0].min(dim=1).values.clamp(0.0, float(W))
        new_y1 = out_xy[:, :, 1].min(dim=1).values.clamp(0.0, float(H))
        new_x2 = out_xy[:, :, 0].max(dim=1).values.clamp(0.0, float(W))
        new_y2 = out_xy[:, :, 1].max(dim=1).values.clamp(0.0, float(H))

        valid = ((new_x2 - new_x1) * (new_y2 - new_y1)) >= 1.0
        if valid.any():
            x1v, y1v, x2v, y2v = new_x1[valid], new_y1[valid], new_x2[valid], new_y2[valid]
            new_tgt = torch.stack([
                cls_ids[valid],
                (x1v + x2v) / 2.0 / W,
                (y1v + y2v) / 2.0 / H,
                (x2v - x1v) / W,
                (y2v - y1v) / H,
            ], dim=1)
        else:
            new_tgt = torch.zeros((0, 5), dtype=tgt.dtype, device=device)

        new_targets.append(new_tgt)

    return new_targets


def build_cpu_transforms(
    config: dict,
    is_train: bool = True,
    input_size: tuple[int, int] | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> DetectionTransform:
    """CPU-only transform pipeline when GPU augmentation is enabled.

    Includes Mosaic/MixUp/CopyPaste (require disk I/O, must stay on CPU) and
    IRSimulation (uint8-specific ops), plus Resize and ToDtype. Excludes
    RandomAffine, ColorJitter, Flips, and Normalize — handled on GPU.

    Args:
        config: The ``augmentation`` section of a training YAML config.
        is_train: If True, include composite augmentations.
        input_size: ``(height, width)``. Required.
        mean: Unused — present for API compatibility.
        std: Unused — present for API compatibility.

    Returns:
        A DetectionTransform with only CPU-mandatory transforms.
    """
    if input_size is None:
        raise ValueError("input_size is required for build_cpu_transforms")

    transforms: list[Any] = []

    if is_train:
        if config.get("mosaic", False):
            transforms.append(Mosaic(input_size=input_size))
        if config.get("mixup", False):
            transforms.append(MixUp())
        if config.get("copypaste", False):
            transforms.append(CopyPaste(
                p=config.get("copypaste_p", 0.5),
                max_objects=config.get("copypaste_max", 3),
            ))
        if config.get("ir_simulation", False):
            transforms.append(IRSimulation(
                ir_prob=config.get("ir_prob", 0.3),
                low_light_prob=config.get("low_light_prob", 0.2),
                noise_sigma=config.get("noise_sigma", 15),
            ))

    # Mosaic always outputs input_size — skip the no-op CPU resize
    if not (is_train and config.get("mosaic", False)):
        transforms.append(v2.Resize(size=list(input_size), antialias=True))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    # Normalize is deferred to GpuDetectionTransform

    return DetectionTransform(transforms, canvas_size=input_size)


def build_gpu_transforms(
    config: dict,
    input_size: tuple[int, int],
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> GpuDetectionTransform:
    """Stateless augmentations to run on GPU after batch transfer.

    The pipeline is split so that box-coupled spatial transforms (RandomAffine,
    Clamp, Sanitize) run per-sample, while box-independent transforms (ColorJitter,
    Flips, Normalize) run once on the stacked (B,C,H,W) batch to eliminate
    per-sample Python overhead for those ops.

    Args:
        config: The ``augmentation`` section of a training YAML config.
        input_size: ``(height, width)`` — must match the collated batch spatial dims.
        mean: Normalisation mean. Defaults to ImageNet values.
        std: Normalisation std. Defaults to ImageNet values.

    Returns:
        A GpuDetectionTransform callable: (images, targets) -> (images, targets).
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    scale = config.get("scale", [0.5, 1.5])
    scale_range = tuple(scale) if isinstance(scale, (list, tuple)) and len(scale) == 2 else (0.5, 1.5)
    degrees = config.get("degrees", 0.0)
    translate_val = config.get("translate", 0.0)
    shear_val = config.get("shear", 0.0)

    return GpuDetectionTransform(
        degrees=degrees,
        translate=(translate_val, translate_val) if translate_val > 0 else None,
        scale_range=scale_range,
        shear=(-shear_val, shear_val, -shear_val, shear_val) if shear_val > 0 else None,
        fill=114 / 255.0,
        hsv_h=config.get("hsv_h", 0.0),
        hsv_s=config.get("hsv_s", 0.0),
        hsv_v=config.get("hsv_v", 0.0),
        contrast=config.get("contrast", 0.0),
        flip_h_p=config.get("fliplr", 0.0),
        flip_v_p=config.get("flipud", 0.0),
        mean=mean,
        std=std,
        input_size=input_size,
        normalize=config.get("normalize", True),
    )
