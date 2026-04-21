"""Model evaluation pipeline for trained object detection models.

Runs inference on a dataset split, collects predictions, and computes
full evaluation metrics (mAP, per-class AP, confusion matrix, failure cases).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root
from core.p05_data.base_dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_EXTENSIONS
from core.p05_data.classification_dataset import (
    ClassificationDataset,
    build_classification_transforms,
    classification_collate_fn,
)
from core.p06_training.postprocess import postprocess as _dispatch_postprocess
from core.p08_evaluation.sv_metrics import (
    _compute_precision_recall_from_iou,
    compute_map,
)
from utils.config import resolve_path
from utils.device import get_device
from utils.metrics import compute_iou
from utils.progress import ProgressBar


class ModelEvaluator:
    """Evaluate a trained detection model on a dataset split.

    Runs batched inference, collects predictions and ground truths,
    and computes detection metrics (mAP, confusion matrix, failure cases).

    Args:
        model: Trained PyTorch model (nn.Module). Must accept image tensors
            of shape (B, 3, H, W) and return (B, N, 5+C).
        data_config: Data config dict (from load_config). Must contain
            "path", "names", "num_classes", "input_size".
        device: Compute device. Auto-detected if None.
        conf_threshold: Confidence threshold for filtering predictions.
        iou_threshold: IoU threshold for NMS and metric matching.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        output_format: Postprocessor name (e.g. "yolox", "dfine"). If None,
            auto-detected from model.output_format or defaults to "yolox".
    """

    def __init__(
        self,
        model: nn.Module,
        data_config: dict,
        device: torch.device | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        batch_size: int = 16,
        num_workers: int = 4,
        output_format: str | None = None,
    ) -> None:
        self.model = model
        self.data_config = data_config
        self.device = device or get_device()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect output_format from model if not specified
        if output_format is not None:
            self.output_format = output_format
        elif hasattr(model, 'output_format'):
            self.output_format = model.output_format
        else:
            self.output_format = "yolox"

        self.num_classes = data_config["num_classes"]
        self.class_names: dict[int, str] = {
            int(k): v for k, v in data_config["names"].items()
        }
        self.input_size = tuple(data_config["input_size"])

        self.model.to(self.device)
        self.model.eval()

    def _build_dataloader(self, split: str) -> torch.utils.data.DataLoader:
        """Build a DataLoader for the requested split.

        Tries to import the project's YOLOXDataset and build_dataloader.
        Falls back to a minimal implementation if not available.

        Args:
            split: Dataset split name ("train", "val", or "test").

        Returns:
            PyTorch DataLoader yielding (images, targets) batches.

        Raises:
            FileNotFoundError: If the split directory does not exist.
        """
        config_path = self.data_config.get("_config_dir", Path.cwd())
        base_path = resolve_path(self.data_config["path"], config_path)
        split_subdir = self.data_config.get(split)
        if split_subdir is None:
            raise ValueError(f"Split '{split}' not found in data config")

        images_dir = base_path / split_subdir
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {images_dir}. "
                f"Ensure dataset is prepared at {base_path}"
            )

        # Use the project's YOLOXDataset with the same val-split transform
        # pipeline used during training (Resize + ToDtype + ImageNet Normalize).
        # Target tensors are converted to the dict format expected by the
        # mAP matching code in _get_predictions_detection.
        from core.p05_data.detection_dataset import YOLOXDataset
        from core.p05_data.detection_dataset import collate_fn as det_collate
        from core.p05_data.transforms import build_transforms

        transforms = build_transforms(
            config={}, is_train=False,
            input_size=tuple(self.input_size),
            mean=self.data_config.get("mean", IMAGENET_MEAN),
            std=self.data_config.get("std", IMAGENET_STD),
        )
        # YOLOXDataset expects a dict config with at least 'path' and split key
        dataset = YOLOXDataset(
            data_config=self.data_config,
            split=split,
            transforms=transforms,
            base_dir=str(config_path),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=det_collate,
            pin_memory=self.device.type == "cuda",
        )

    def get_predictions(
        self, split: str = "test"
    ) -> tuple[list[dict], list[dict]]:
        """Run inference and return raw predictions and ground truths.

        For detection models, returns box-based predictions and ground truths.
        For classification models, returns logit-based predictions and labels.

        Args:
            split: Dataset split to evaluate ("train", "val", or "test").

        Returns:
            Tuple of (predictions_list, ground_truths_list).
            Detection: {"boxes": (N,4), "scores": (N,), "labels": (N,)}
            Classification: {"logits": (C,), "label": int}
        """
        if self.output_format == "classification":
            return self._get_predictions_classification(split)
        if self.output_format == "segmentation":
            return self._get_predictions_segmentation(split)
        return self._get_predictions_detection(split)

    def _get_predictions_classification(
        self, split: str
    ) -> tuple[list[dict], list[dict]]:
        """Run classification inference and collect logits + labels."""
        input_size = self.input_size
        mean = self.data_config.get("mean", IMAGENET_MEAN)
        std = self.data_config.get("std", IMAGENET_STD)
        config_dir = self.data_config.get("_config_dir", Path.cwd())

        transforms = build_classification_transforms(
            is_train=False, input_size=input_size, mean=mean, std=std,
        )
        dataset = ClassificationDataset(
            self.data_config, split=split, transforms=transforms,
            base_dir=str(config_dir),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=classification_collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        all_predictions: list[dict] = []
        all_ground_truths: list[dict] = []

        with ProgressBar(total=len(dataloader), desc=f"Evaluating [{split}]") as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    images = batch["images"].to(self.device)
                    targets = batch["targets"]

                    logits = self.model(images)  # (B, C)
                    probs = torch.softmax(logits, dim=-1)

                    pred_classes = probs.argmax(dim=1)  # (B,)
                    confidences = probs.gather(
                        1, pred_classes.unsqueeze(1)
                    ).squeeze(1)  # (B,)
                    logits_np = logits.cpu().numpy()
                    pred_classes_np = pred_classes.cpu().numpy()
                    confidences_np = confidences.cpu().numpy()
                    for i in range(len(logits_np)):
                        all_predictions.append({
                            "logits": logits_np[i],
                            "class_id": int(pred_classes_np[i]),
                            "confidence": float(confidences_np[i]),
                        })
                        all_ground_truths.append({
                            "label": targets[i].item(),
                        })

                    pbar.update()

        return all_predictions, all_ground_truths

    def _get_predictions_detection(
        self, split: str
    ) -> tuple[list[dict], list[dict]]:
        """Run detection inference and collect box predictions + ground truths."""
        dataloader = self._build_dataloader(split)
        all_predictions: list[dict] = []
        all_ground_truths: list[dict] = []
        input_h, input_w = self.input_size

        def _gt_to_dict(t: torch.Tensor) -> dict:
            """YOLOXDataset emits per-image (N, 5) tensor [cls, cx, cy, w, h]
            normalized [0, 1]. Convert to xyxy pixel-space dict for mAP."""
            arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
            if arr.shape[0] == 0:
                return {"boxes": np.zeros((0, 4), dtype=np.float64),
                        "labels": np.zeros(0, dtype=np.int64)}
            cx = arr[:, 1] * input_w
            cy = arr[:, 2] * input_h
            w = arr[:, 3] * input_w
            h = arr[:, 4] * input_h
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            return {
                "boxes": np.stack([x1, y1, x2, y2], axis=1).astype(np.float64),
                "labels": arr[:, 0].astype(np.int64),
            }

        with ProgressBar(total=len(dataloader), desc=f"Evaluating [{split}]") as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    images = batch["images"].to(self.device)
                    targets = batch["targets"]  # list of (N, 5) tensors

                    # Forward pass
                    outputs = self.model(images)

                    # Postprocess: use model's built-in postprocessor for
                    # non-YOLOX models, fall back to registry for YOLOX
                    if self.output_format != "yolox" and hasattr(self.model, "postprocess"):
                        target_sizes = torch.tensor(
                            [list(self.input_size)] * outputs.shape[0],
                            device=outputs.device,
                        )
                        batch_preds = self.model.postprocess(
                            outputs, self.conf_threshold, target_sizes,
                        )
                        for pred_dict, gt in zip(batch_preds, targets, strict=True):
                            all_predictions.append(pred_dict)
                            all_ground_truths.append(_gt_to_dict(gt))
                        pbar.update()
                        continue

                    detections = _dispatch_postprocess(
                        self.output_format,
                        self.model,
                        predictions=outputs,
                        conf_threshold=self.conf_threshold,
                        nms_threshold=self.iou_threshold,
                    )

                    # Collect results (postprocess returns dicts with boxes/scores/labels)
                    for pred_dict, gt in zip(detections, targets, strict=True):
                        all_predictions.append(pred_dict)
                        all_ground_truths.append(_gt_to_dict(gt))

                    pbar.update()

        return all_predictions, all_ground_truths

    def evaluate(self, split: str = "test") -> dict:
        """Run full evaluation and return metrics.

        Dispatches by output_format:
        - Detection: mAP, per-class AP, precision, recall
        - Classification: accuracy, top-5 accuracy, per-class precision/recall/F1

        Args:
            split: Dataset split to evaluate.

        Returns:
            Dictionary of metrics appropriate for the task.
        """
        predictions, ground_truths = self.get_predictions(split)

        if self.output_format == "classification":
            return self._evaluate_classification(predictions, ground_truths)
        if self.output_format == "segmentation":
            return self._evaluate_segmentation(predictions, ground_truths)
        return self._evaluate_detection(predictions, ground_truths)

    def _evaluate_classification(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> dict:
        """Compute classification metrics: accuracy, top-5, per-class F1."""
        pred_classes = np.array([p["class_id"] for p in predictions])
        gt_labels = np.array([g["label"] for g in ground_truths])

        # Overall accuracy
        correct = pred_classes == gt_labels
        accuracy = float(correct.mean()) if len(correct) > 0 else 0.0

        # Top-5 accuracy (from logits)
        top5_accuracy = 0.0
        if predictions and "logits" in predictions[0]:
            logits = np.stack([p["logits"] for p in predictions])
            if logits.shape[1] >= 5:
                top5_preds = np.argsort(logits, axis=-1)[:, -5:]
                top5_correct = np.any(top5_preds == gt_labels[:, None], axis=1)
                top5_accuracy = float(top5_correct.mean())

        # Per-class precision, recall, F1
        per_class = {}
        for cls_id in range(self.num_classes):
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            tp = int(((pred_classes == cls_id) & (gt_labels == cls_id)).sum())
            fp = int(((pred_classes == cls_id) & (gt_labels != cls_id)).sum())
            fn = int(((pred_classes != cls_id) & (gt_labels == cls_id)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[cls_name] = {
                "precision": precision, "recall": recall, "f1": f1,
                "support": int((gt_labels == cls_id).sum()),
            }

        return {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "per_class": per_class,
            "num_images": len(predictions),
        }

    def _evaluate_detection(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> dict:
        """Compute detection metrics: mAP, per-class AP, precision, recall."""
        metrics = compute_map(
            predictions, ground_truths,
            iou_threshold=self.iou_threshold,
            num_classes=self.num_classes,
        )
        metrics["num_images"] = len(predictions)
        metrics["conf_threshold"] = self.conf_threshold
        metrics["iou_threshold"] = self.iou_threshold
        return metrics

    def _get_predictions_segmentation(
        self, split: str
    ) -> tuple[list[dict], list[dict]]:
        """Run segmentation inference and collect predicted + GT masks."""
        import torch.nn.functional as F_seg

        from core.p05_data.segmentation_dataset import (
            SegmentationDataset,
            build_segmentation_transforms,
            segmentation_collate_fn,
        )

        input_size = self.input_size
        mean = self.data_config.get("mean", IMAGENET_MEAN)
        std = self.data_config.get("std", IMAGENET_STD)
        config_dir = self.data_config.get("_config_dir", Path.cwd())

        transforms = build_segmentation_transforms(
            is_train=False, input_size=input_size, mean=mean, std=std,
        )
        dataset = SegmentationDataset(
            self.data_config, split=split, transforms=transforms,
            base_dir=str(config_dir),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=segmentation_collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        all_predictions: list[dict] = []
        all_ground_truths: list[dict] = []

        with ProgressBar(total=len(dataloader), desc=f"Evaluating [{split}]") as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    images = batch["images"].to(self.device)
                    gt_masks = batch["targets"]  # list of (H, W) tensors

                    logits = self.model(images)  # (B, C, H', W')
                    class_maps = logits.argmax(dim=1)  # (B, H', W')

                    input_h, input_w = input_size
                    if class_maps.shape[-2:] != (input_h, input_w):
                        class_maps = F_seg.interpolate(
                            class_maps.unsqueeze(1).float(),
                            size=(input_h, input_w),
                            mode="nearest",
                        ).squeeze(1).long()

                    for i in range(class_maps.shape[0]):
                        all_predictions.append({
                            "class_map": class_maps[i].cpu().numpy(),
                        })
                        all_ground_truths.append({
                            "mask": gt_masks[i].numpy()
                            if isinstance(gt_masks[i], torch.Tensor)
                            else gt_masks[i],
                        })

                    pbar.update()

        return all_predictions, all_ground_truths

    def _evaluate_segmentation(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> dict:
        """Compute segmentation metrics: mIoU and per-class IoU."""
        intersection = np.zeros(self.num_classes)
        union = np.zeros(self.num_classes)

        if predictions:
            pred_maps = np.stack(
                [p["class_map"] for p in predictions], axis=0
            )
            gt_masks = np.stack(
                [g["mask"] for g in ground_truths], axis=0
            )
            for c in range(self.num_classes):
                p = pred_maps == c
                g = gt_masks == c
                intersection[c] = np.logical_and(p, g).sum()
                union[c] = np.logical_or(p, g).sum()

        iou = np.where(union > 0, intersection / (union + 1e-10), 0.0)
        per_class: dict = {}
        for c in range(self.num_classes):
            name = self.class_names.get(c, f"class_{c}")
            per_class[name] = {"iou": float(iou[c])}

        return {
            "mIoU": float(np.mean(iou)),
            "per_class": per_class,
            "num_images": len(predictions),
        }

    def evaluate_per_class(self, split: str = "test") -> dict:
        """Run evaluation with detailed per-class breakdown.

        For classification, returns per-class precision/recall/F1 from evaluate().
        For detection, computes per-class AP and PR curves.

        Args:
            split: Dataset split to evaluate.

        Returns:
            Dictionary mapping class name to:
                - ap: float (detection) or f1: float (classification)
                - precision: float (at best F1)
                - recall: float (at best F1)
                - n_gt: int (number of GT instances)
                - pr_curve: (precision_array, recall_array, thresholds)
        """
        predictions, ground_truths = self.get_predictions(split)

        # Classification: per-class metrics are already in evaluate()
        if self.output_format == "classification":
            metrics = self._evaluate_classification(predictions, ground_truths)
            return metrics.get("per_class", {})

        if self.output_format == "segmentation":
            metrics = self._evaluate_segmentation(predictions, ground_truths)
            return metrics.get("per_class", {})

        return self._evaluate_per_class_detection(predictions, ground_truths)

    def _evaluate_per_class_detection(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> dict:
        """Compute per-class AP and PR curves from pre-computed predictions."""
        result: dict = {}

        # Pre-compute full per-image IoU matrices once; the per-class PR helper
        # slices class rows/cols instead of recomputing IoU per class.
        iou_matrices: list[np.ndarray] = []
        for pred, gt in zip(predictions, ground_truths, strict=True):
            pred_boxes = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 4)
            gt_boxes = np.asarray(gt["boxes"], dtype=np.float64).reshape(-1, 4)
            if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
                iou_matrices.append(
                    np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)
                )
            else:
                iou_matrices.append(compute_iou(pred_boxes, gt_boxes))

        for cls_id in range(self.num_classes):
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            prec, rec, thresholds = _compute_precision_recall_from_iou(
                predictions, ground_truths, iou_matrices,
                cls_id, self.iou_threshold,
            )

            if prec.size == 0:
                result[cls_name] = {
                    "ap": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "n_gt": 0,
                    "pr_curve": (prec, rec, thresholds),
                }
                continue

            # AP via all-point interpolation (COCO-style)
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([1.0], prec, [0.0]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))

            f1 = np.where(
                (prec + rec) > 0,
                2 * prec * rec / (prec + rec + 1e-16),
                0.0,
            )
            best_idx = int(np.argmax(f1))

            # Count GT instances for this class
            n_gt = sum(
                int((np.asarray(gt["labels"]) == cls_id).sum())
                for gt in ground_truths
            )

            result[cls_name] = {
                "ap": ap,
                "precision": float(prec[best_idx]),
                "recall": float(rec[best_idx]),
                "n_gt": n_gt,
                "pr_curve": (prec, rec, thresholds),
            }

        return result

    def get_failure_cases(
        self, split: str = "test", max_cases: int = 50
    ) -> list[dict]:
        """Collect false positive and false negative examples.

        Args:
            split: Dataset split to analyze.
            max_cases: Maximum number of failure cases to return.

        Returns:
            List of dicts, each with:
                - image_idx: int
                - type: "false_positive" or "false_negative"
                - box: (4,) array [x1,y1,x2,y2]
                - class_id: int
                - score: float (for FP) or None (for FN)

        For classification models, returns misclassified samples instead.
        """
        if self.output_format == "classification":
            return self._get_misclassifications(split, max_cases)

        predictions, ground_truths = self.get_predictions(split)
        failures: list[dict] = []

        for img_idx in range(len(predictions)):
            if len(failures) >= max_cases:
                break

            pred = predictions[img_idx]
            gt = ground_truths[img_idx]

            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]
            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            n_pred = pred_boxes.shape[0]
            n_gt = gt_boxes.shape[0]

            if n_pred == 0 and n_gt == 0:
                continue

            # Match predictions to GT
            gt_matched = np.zeros(n_gt, dtype=bool)

            if n_pred > 0 and n_gt > 0:
                iou_matrix = compute_iou(pred_boxes, gt_boxes)
                sorted_idx = np.argsort(-pred_scores)
                pred_matched = np.zeros(n_pred, dtype=bool)

                for pi in sorted_idx:
                    best_gt = -1
                    best_iou = self.iou_threshold
                    for gi in range(n_gt):
                        if gt_matched[gi]:
                            continue
                        if (pred_labels[pi] == gt_labels[gi] and
                                iou_matrix[pi, gi] >= best_iou):
                            best_iou = iou_matrix[pi, gi]
                            best_gt = gi
                    if best_gt >= 0:
                        gt_matched[best_gt] = True
                        pred_matched[pi] = True
                    else:
                        # False positive
                        if len(failures) < max_cases:
                            failures.append({
                                "image_idx": img_idx,
                                "type": "false_positive",
                                "box": pred_boxes[pi],
                                "class_id": int(pred_labels[pi]),
                                "score": float(pred_scores[pi]),
                            })
            elif n_pred > 0:
                # All preds are FP
                for pi in range(min(n_pred, max_cases - len(failures))):
                    failures.append({
                        "image_idx": img_idx,
                        "type": "false_positive",
                        "box": pred_boxes[pi],
                        "class_id": int(pred_labels[pi]),
                        "score": float(pred_scores[pi]),
                    })

            # False negatives: unmatched GTs
            for gi in range(n_gt):
                if not gt_matched[gi] and len(failures) < max_cases:
                    failures.append({
                        "image_idx": img_idx,
                        "type": "false_negative",
                        "box": gt_boxes[gi],
                        "class_id": int(gt_labels[gi]),
                        "score": None,
                    })

        return failures[:max_cases]

    def _get_misclassifications(
        self, split: str, max_cases: int
    ) -> list[dict]:
        """Collect misclassified samples for classification models."""
        predictions, ground_truths = self.get_predictions(split)
        failures = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths, strict=True)):
            if pred["class_id"] != gt["label"]:
                failures.append({
                    "image_idx": i,
                    "type": "misclassification",
                    "predicted": pred["class_id"],
                    "actual": gt["label"],
                    "confidence": pred["confidence"],
                })
                if len(failures) >= max_cases:
                    break
        return failures


# ---------------------------------------------------------------------------
# Minimal fallback dataset (used when pipeline dataset module unavailable)
# ---------------------------------------------------------------------------


class _MinimalDetectionDataset(torch.utils.data.Dataset):
    """Minimal YOLO-format detection dataset for evaluation.

    Reads images and labels from the standard YOLO directory layout:
        images_dir/  *.jpg
        labels_dir/  *.txt  (sibling to images/)
    """

    IMAGE_EXTENSIONS = IMG_EXTENSIONS

    def __init__(self, images_dir: Path, input_size: tuple[int, int]) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = self.images_dir.parent / "labels"
        self.input_size = input_size  # (H, W)
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            # Return blank on read failure
            h, w = self.input_size
            img = np.zeros((h, w, 3), dtype=np.uint8)

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = self.input_size

        # Resize with letterbox padding
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # HWC -> CHW, normalize to [0, 1]
        img_tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Parse labels
        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []
        if label_path.exists():
            text = label_path.read_text().strip()
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                # YOLO normalized -> pixel coords on original image
                abs_cx = cx * orig_w
                abs_cy = cy * orig_h
                abs_w = w * orig_w
                abs_h = h * orig_h
                # Scale and shift to padded image coords
                x1 = (abs_cx - abs_w / 2) * scale + pad_left
                y1 = (abs_cy - abs_h / 2) * scale + pad_top
                x2 = (abs_cx + abs_w / 2) * scale + pad_left
                y2 = (abs_cy + abs_h / 2) * scale + pad_top
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

        target = {
            "boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4),
            "labels": np.array(labels, dtype=np.int64),
        }

        return {
            "image": torch.from_numpy(img_tensor),
            "target": target,
            "image_path": str(img_path),
            "scale": scale,
            "pad": (pad_left, pad_top),
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Collate function for _MinimalDetectionDataset.

    Args:
        batch: List of sample dicts from the dataset.

    Returns:
        Batched dict with "images" tensor and "targets" list.
    """
    images = torch.stack([sample["image"] for sample in batch])
    targets = [sample["target"] for sample in batch]
    paths = [sample["image_path"] for sample in batch]
    return {"images": images, "targets": targets, "image_paths": paths}
