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
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as v2

from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def _to_v2_sample(
    image_bgr_np: np.ndarray,
    targets_np: np.ndarray,
    canvas_size: Optional[Tuple[int, int]] = None,
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

    # BGR -> RGB, HWC uint8 -> CHW tensor wrapped in tv_tensors.Image
    rgb_np = image_bgr_np[:, :, ::-1].copy()
    image_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # CHW uint8
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


def _from_v2_sample(sample: dict, canvas_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        v2_transforms: List[Any],
        canvas_size: Tuple[int, int],
    ) -> None:
        self.pipeline = v2.Compose(v2_transforms)
        self.transforms = v2_transforms
        self.canvas_size = canvas_size

    def __call__(
        self, image: np.ndarray, targets: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_size = (image.shape[0], image.shape[1])
        sample = _to_v2_sample(image, targets, original_size)
        sample = self.pipeline(sample)
        return _from_v2_sample(sample, self.canvas_size)

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "DetectionTransform([\n" + "\n".join(lines) + "\n])"


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
        input_size: Tuple[int, int] = (640, 640),
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

    def _get_image_tensor(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            if random.random() < self.ir_prob:
                gray = F.rgb_to_grayscale(img, num_output_channels=1)
                img = gray.repeat(3, 1, 1) if img.ndim == 3 else gray.repeat(1, 3, 1, 1)
                img = torch.clamp(img.float() * 1.3 + 10, 0, 255).to(img.dtype)

            if random.random() < self.low_light_prob:
                factor = random.uniform(0.1, 0.4)
                img_f = img.float() * factor
                if self.noise_sigma > 0:
                    noise = torch.randn_like(img_f) * self.noise_sigma
                    img_f = img_f + noise
                img = torch.clamp(img_f, 0, 255).to(sample["image"].dtype)

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
    input_size: Optional[Tuple[int, int]] = None,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> DetectionTransform:
    """Build a transform pipeline from an augmentation config dict.

    Args:
        config: The ``augmentation`` section of a training YAML config.
            Expected keys (all optional, with sensible defaults):
            ``mosaic``, ``mixup``, ``hsv_h``, ``hsv_s``, ``hsv_v``,
            ``fliplr``, ``flipud``, ``scale``, ``degrees``, ``translate``,
            ``shear``, ``copypaste``, ``ir_simulation``.
        is_train: If True build training pipeline (with augmentations);
            if False build a minimal val/test pipeline (resize + normalise).
        input_size: ``(height, width)``. Required.
        mean: Normalisation mean. Defaults to ImageNet values.
        std: Normalisation std. Defaults to ImageNet values.

    Returns:
        A DetectionTransform wrapping a v2.Compose pipeline.
    """
    if input_size is None:
        raise ValueError("input_size is required for build_transforms")

    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD

    transforms: List[Any] = []

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

        # RandomAffine
        scale = config.get("scale", [0.5, 1.5])
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            scale_range = tuple(scale)
        else:
            scale_range = (0.5, 1.5)

        degrees = config.get("degrees", 0.0)
        translate_val = config.get("translate", 0.0)
        shear_val = config.get("shear", 0.0)

        transforms.append(v2.RandomAffine(
            degrees=degrees,
            translate=(translate_val, translate_val) if translate_val > 0 else None,
            scale=scale_range,
            shear=(-shear_val, shear_val, -shear_val, shear_val) if shear_val > 0 else None,
            fill=114,
        ))

        # Clamp + sanitize after spatial transforms
        transforms.append(v2.ClampBoundingBoxes())
        transforms.append(v2.SanitizeBoundingBoxes(min_area=1.0))

        # ColorJitter (replaces HSVAugment)
        hsv_h = config.get("hsv_h", 0.0)
        hsv_s = config.get("hsv_s", 0.0)
        hsv_v = config.get("hsv_v", 0.0)
        if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
            transforms.append(v2.ColorJitter(
                hue=hsv_h,
                saturation=hsv_s,
                brightness=hsv_v,
            ))

        # IR simulation
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

    # Always applied (both train and eval)
    transforms.append(v2.Resize(size=list(input_size), antialias=True))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Normalize(mean=mean, std=std))

    return DetectionTransform(transforms, canvas_size=input_size)
