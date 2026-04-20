#!/usr/bin/env python
"""RT-DETRv2 fine-tuning on CPPE-5 — direct port of qubvel's reference notebook.

Source: notebooks/detr_finetune_reference/reference/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb
Upstream: https://github.com/qubvel/transformers-notebooks

Run in the isolated notebook env:
    .venv-notebook/bin/python notebooks/detr_finetune_reference/rtdetr_v2_finetune_cppe5.py

Deps pinned in notebooks/detr_finetune_reference/pyproject.toml (installed by
scripts/setup-notebook-venv.sh).

Self-contained: all checkpoints, tensorboard logs, and eval artefacts are
written under `notebooks/detr_finetune_reference/runs/rtdetr_v2_r50_cppe5/`
regardless of the invoking cwd (resolved via `__file__`).
"""
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # notebooks/detr_finetune_reference/
_RUN_DIR = _HERE / "runs" / "rtdetr_v2_r50_cppe5"

# For training
# To get started, we'll define global constants, namely the model checkpoint and image size. Feel free to select other pretrained checkpoint available on the [hub](https://huggingface.co/PekingU).

checkpoint = "PekingU/rtdetr_v2_r50vd"
image_size = 480

# ## Load dataset
# 
# Next we'll load the dataset on which we'd like to fine-tune RT-DETRv2.
# 
# In case of a custom object detection dataset, I'd recommend the guide [here](https://huggingface.co/docs/datasets/image_dataset#object-detection). Here we load an existing dataset from the hub, namely [CPPE-5](https://huggingface.co/datasets/cppe-5) which contains images with annotations identifying medical personal protective equipment (PPE) in the context of the COVID-19 pandemic.
# 
# Start by loading the dataset and creating a `validation` split from `train`:

from datasets import load_dataset

dataset = load_dataset("cppe-5")

if "validation" not in dataset:
    split = dataset["train"].train_test_split(0.15, seed=1337)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

print("dataset:", dataset)
# You'll see that this dataset has 1000 images for train and validation sets and a test set with 29 images.
# 
# To get familiar with the data, explore what the examples look like.

# dataset["train"][0]  # (Jupyter display no-op)
# The examples in the dataset have the following fields:
# - `image_id`: the example image id
# - `image`: a `PIL.Image.Image` object containing the image
# - `width`: width of the image
# - `height`: height of the image
# - `objects`: a dictionary containing bounding box metadata for the objects in the image:
#   - `id`: the annotation id
#   - `area`: the area of the bounding box
#   - `bbox`: the object's bounding box (in the [COCO format](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#coco) )
#   - `category`: the object's category, with possible values including `Coverall (0)`, `Face_Shield (1)`, `Gloves (2)`, `Goggles (3)` and `Mask (4)`
# 
# You may notice that the `bbox` field follows the COCO format, which is the format that the RT-DETRv2 model expects.
# However, the grouping of the fields inside `objects` differs from the annotation format RT-DETRv2 requires. You will
# need to apply some preprocessing transformations before using this data for training.
# 
# To get an even better understanding of the data, visualize an example in the dataset.

import numpy as np
from PIL import Image, ImageDraw

# Get mapping from category id to category name
categories = dataset["train"].features["objects"]["category"].feature.names
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

# Load image and annotations
image = dataset["train"][2]["image"]
annotations = dataset["train"][2]["objects"]

# Draw bounding boxes and labels
draw = ImageDraw.Draw(image)
for i in range(len(annotations["id"])):
    box = annotations["bbox"][i]
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

# image  # (Jupyter display no-op)
# To visualize the bounding boxes with associated labels, you can get the labels from the dataset's metadata, specifically
# the `category` field.
# You'll also want to create dictionaries that map a label id to a label class (`id2label`) and the other way around (`label2id`).
# You can use them later when setting up the model. Including these maps will make your model reusable by others if you share
# it on the Hugging Face Hub. Please note that, the part of above code that draws the bounding boxes assume that it is in `COCO` format `(x_min, y_min, width, height)`. It has to be adjusted to work for other formats like `(x_min, y_min, x_max, y_max)`.
# 
# As a final step of getting familiar with the data, explore it for potential issues. One common problem with datasets for
# object detection is bounding boxes that "stretch" beyond the edge of the image. Such "runaway" bounding boxes can raise
# errors during training and should be addressed. There are a few examples with this issue in this dataset.
# To keep things simple in this guide, we will set `clip=True` for `BboxParams` in transformations below.

# ## Preprocess the data

# To finetune a model, you must preprocess the data you plan to use to match precisely the approach used for the pre-trained model.
# [AutoImageProcessor](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoImageProcessor) takes care of processing image data to create `pixel_values`, `pixel_mask`, and
# `labels` that a DETR model can train with. The image processor has some attributes that you won't have to worry about:
# 
# - `image_mean = [0.485, 0.456, 0.406 ]`
# - `image_std = [0.229, 0.224, 0.225]`
# 
# These are the mean and standard deviation used to normalize images during the model pre-training. These values are crucial
# to replicate when doing inference or finetuning a pre-trained image model.
# 
# Instantiate the image processor from the same checkpoint as the model you want to finetune.

from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    checkpoint,
    do_resize=True,
    size={"width": image_size, "height": image_size},
    use_fast=True,
)

# Before passing the images to the `image_processor`, apply two preprocessing transformations to the dataset:
# - Augmenting images
# - Reformatting annotations to meet RT-DETRv2 expectations
# 
# First, to make sure the model does not overfit on the training data, you can apply image augmentation with any data augmentation library. Here we use [Albumentations](https://albumentations.ai/docs/).
# This library ensures that transformations affect the image and update the bounding boxes accordingly.
# The 🤗 Datasets library documentation has a detailed [guide on how to augment images for object detection](https://huggingface.co/docs/datasets/object_detection),
# and it uses the exact same dataset as an example. Apply the same approach here, resize each image,
# flip it horizontally, and brighten it. For additional augmentation options, explore the [Albumentations Demo Space](https://huggingface.co/spaces/qubvel-hf/albumentations-demo).

import albumentations as A

train_augmentation_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1),
)

# to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
)

# Visualize some augmented images to make sure images look natural and annotations are correct:

for i in [15, 16, 17]:
    image = dataset["train"][i]["image"]
    annotations = dataset["train"][i]["objects"]

    # Apply the augmentation
    output = train_augmentation_and_transform(image=np.array(image), bboxes=annotations["bbox"], category=annotations["category"])

    # Unpack the output
    image = Image.fromarray(output["image"])
    categories, boxes = output["category"], output["bboxes"]

    # Draw the augmented image
    draw = ImageDraw.Draw(image)
    for category, box in zip(categories, boxes):
        x, y, w, h = box
        draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
        draw.text((x, y), id2label[category], fill="white")

    # display(image.resize([256, 256]))
# The `image_processor` expects the annotations to be in the following format: `{'image_id': int, 'annotations': List[Dict]}`,
#  where each dictionary is a COCO object annotation. Let's add a function to reformat annotations for a single example:

from torch.utils.data import Dataset

class CPPE5Dataset(Dataset):
    def __init__(self, dataset, image_processor, transform=None):
        self.dataset = dataset
        self.image_processor = image_processor
        self.transform = transform

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        """Format one set of image annotations to the COCO format

        Args:
            image_id (str): image id. e.g. "0001"
            categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
            boxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
                ([center_x, center_y, width, height] in absolute coordinates)

        Returns:
            dict: {
                "image_id": image id,
                "annotations": list of formatted annotations
            }
        """
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": list(bbox),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image_id = sample["image_id"]
        image = sample["image"]
        boxes = sample["objects"]["bbox"]
        categories = sample["objects"]["category"]

        # Convert image to RGB numpy array
        image = np.array(image.convert("RGB"))

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        # Format annotations in COCO format for image_processor
        formatted_annotations = self.format_image_annotations_as_coco(image_id, categories, boxes)

        # Apply the image processor transformations: resizing, rescaling, normalization
        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result

# Now you can combine the image and annotation transformations to use on a batch of examples:

train_dataset = CPPE5Dataset(dataset["train"], image_processor, transform=train_augmentation_and_transform)
validation_dataset = CPPE5Dataset(dataset["validation"], image_processor, transform=validation_transform)
test_dataset = CPPE5Dataset(dataset["test"], image_processor, transform=validation_transform)

# train_dataset[15]  # (Jupyter display no-op)
# Apply this preprocessing function to the entire dataset using 🤗 Datasets [with_transform](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.with_transform) method. This method applies
# transformations on the fly when you load an element of the dataset.
# 
# At this point, you can check what an example from the dataset looks like after the transformations. You should see a tensor
# with `pixel_values`, a tensor with `pixel_mask`, and `labels`.

# Check images once again after applying the all the transformations, verify that boxes and labels are correct!

for i in [15, 16, 17]:
    sample = train_dataset[i]

    # De-normalize image
    image = sample["pixel_values"]
    print("Image tensor shape:", image.shape)
    image = image.numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min()) * 255.
    image = Image.fromarray(image.astype(np.uint8))

    # Convert boxes from [center_x, center_y, width, height] to [x, y, width, height] for visualization
    boxes = sample["labels"]["boxes"].numpy()
    print("Boxes shape:", boxes.shape)
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    w, h = image.size
    boxes = boxes * np.array([w, h, w, h])[None]

    categories = sample["labels"]["class_labels"].numpy()
    print("Categories shape:", categories.shape)

    # Draw boxes and labels on image
    draw = ImageDraw.Draw(image)
    for box, category in zip(boxes, categories):
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
        draw.text((x, y), id2label[category], fill="white")

    # display(image)
# You have successfully augmented the images and prepared their annotations. In the final step, create a custom `collate_fn` to batch images together.

import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data

# ## Preparing function to compute mAP

# Object detection models are commonly evaluated with a set of <a href="https://cocodataset.org/#detection-eval">COCO-style metrics</a>. We are going to use `torchmetrics` to compute `mAP` (mean average precision) and `mAR` (mean average recall) metrics and will wrap it to `compute_metrics` function in order to use in [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) for evaluation.
# 
# Intermediate format of boxes used for training is `YOLO` (normalized) but we will compute metrics for boxes in `Pascal VOC` (absolute) format in order to correctly handle box areas. Let's define a function that converts bounding boxes to `Pascal VOC` format:

# Then, in `compute_metrics` function we collect `predicted` and `target` bounding boxes, scores and labels from evaluation loop results and pass it to the scoring function.

import numpy as np
from dataclasses import dataclass
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):

                # here we have "yolo" format (x_center, y_center, width, height) in relative coordinates 0..1
                # and we need to convert it to "pascal" format (x_min, y_min, x_max, y_max) in absolute coordinates
                height, width = size
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([[width, height, width, height]])

                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics

eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label=id2label)

# ## Training the detection model

# You have done most of the heavy lifting in the previous sections, so now you are ready to train your model!
# The images in this dataset are still quite large, even after resizing. This means that finetuning this model will
# require at least one GPU.
# 
# Training involves the following steps:
# 1. Load the model with [AutoModelForObjectDetection](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForObjectDetection) using the same checkpoint as in the preprocessing.
# 2. Define your training hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments).
# 3. Pass the training arguments to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, image processor, and data collator.
# 4. Call [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.
# 
# When loading the model from the same checkpoint that you used for the preprocessing, remember to pass the `label2id`
# and `id2label` maps that you created earlier from the dataset's metadata. Additionally, we specify `ignore_mismatched_sizes=True` to replace the existing classification head with a new one.

from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# In the [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) use `output_dir` to specify where to save your model, then configure hyperparameters as you see fit. For `num_train_epochs=10` training will take about 15 minutes in Google Colab T4 GPU, increase the number of epoch to get better results.
# 
# Important notes:
#  - Do not remove unused columns because this will drop the image column. Without the image column, you
# can't create `pixel_values`. For this reason, set `remove_unused_columns` to `False`.
#  - Set `eval_do_concat_batches=False` to get proper evaluation results. Images have different number of target boxes, if batches are concatenated we will not be able to determine which boxes belongs to particular image.
# 
# If you wish to share your model by pushing to the Hub, set `push_to_hub` to `True` (you must be signed in to Hugging
# Face to upload your model).

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=str(_RUN_DIR),
    num_train_epochs=40,
    max_grad_norm=0.1,
    learning_rate=5e-5,
    warmup_steps=300,
    per_device_train_batch_size=8,
    dataloader_num_workers=2,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="tensorboard",  # or "wandb"
)

# Finally, bring everything together, and call [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train):

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()

# ## Evaluate

from pprint import pprint

metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="eval")
pprint(metrics)

# If you have set `push_to_hub` to `True` in the `training_args`, and you're authenticated with your Hugging Face token, the training checkpoints are pushed to the
# Hugging Face Hub. Upon training completion, push the final model to the Hub as well by calling the [push_to_hub()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) method.

# trainer.push_to_hub()  # skipped — HF Hub push not needed for local reproduction

# These results can be further improved by adjusting the hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). Give it a go!

# ## Inference

# Now that you have finetuned a model, evaluated it, and uploaded it to the Hugging Face Hub, you can use it for inference.

import torch
import requests
from PIL import Image, ImageDraw

device = "cuda"

url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
image = Image.open(requests.get(url, stream=True).raw)

# Load model and image processor from the Hugging Face Hub (skip to use already trained in this session):

from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_repo = "<your-name-on-hf>/rtdetr-v2-r50-cppe5-finetune"

image_processor = AutoImageProcessor.from_pretrained(model_repo)
model = AutoModelForObjectDetection.from_pretrained(model_repo)
model = model.to(device)

# And detect bounding boxes:

inputs = image_processor(images=[image], return_tensors="pt")
inputs = inputs.to(device)
with torch.no_grad():
    outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])

result = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

# Let's plot the result:

image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    text_label = model.config.id2label[label.item()]
    draw.text((x, y), f"{text_label} [ {score.item():.2f} ]", fill="blue")

# image_with_boxes  # (Jupyter display no-op)