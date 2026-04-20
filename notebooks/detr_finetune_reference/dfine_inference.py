"""D-FINE inference — direct port of qubvel's reference notebook.

Source: notebooks/detr_finetune_reference/reference/DFine_inference.ipynb
"""
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Pretrained on Object365 dataset first, then trained on COCO
checkpoint = "ustc-community/dfine-medium-obj365"

# Alternative checkpoints:
#   - https://huggingface.co/models?other=d_fine&author=ustc-community
#
# For example:
# checkpoint = "ustc-community/dfine-xlarge-obj365"
# checkpoint = "ustc-community/dfine-large-obj365"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)

from PIL import Image
import requests

url = "https://live.staticflickr.com/65535/33021460783_1646d43c54_b.jpg"
image = Image.open(requests.get(url, stream=True).raw)
# image  # (Jupyter display no-op)
inputs = image_processor(image, return_tensors="pt")
inputs = inputs.to(device)

print(inputs.keys())

import torch

with torch.no_grad():
  outputs = model(**inputs)

# postprocess model outputs
postprocessed_outputs = image_processor.post_process_object_detection(
    outputs,
    target_sizes=[(image.height, image.width)],
    threshold=0.3,
)
image_detections = postprocessed_outputs[0]  # take only first image results

import matplotlib.pyplot as plt

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
] * 100

def plot_results(pil_image, scores, labels, boxes, visible_classes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_image)
    ax = plt.gca()
    for score, label, box, color in zip(scores, labels, boxes, COLORS):
        # skip not specified classes
        class_name = model.config.id2label[label]
        if visible_classes is not None and not class_name.lower() in visible_classes:
            continue

        xmin, ymin, xmax, ymax = box
        ax.add_patch(
          plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False,
            color=color,
            linewidth=1,
          )
        )
        text = f"{class_name}: {score:0.2f}"
        ax.text(
            xmin, ymin, text, fontsize=8,
            bbox=dict(facecolor='yellow', alpha=0.25),
        )
    plt.axis('off')
    plt.show()

plot_results(
    pil_image=image,
    scores=image_detections['scores'].tolist(),
    labels=image_detections['labels'].tolist(),
    boxes=image_detections['boxes'].tolist(),
)

plot_results(
    pil_image=image,
    scores=image_detections['scores'].tolist(),
    labels=image_detections['labels'].tolist(),
    boxes=image_detections['boxes'].tolist(),
    visible_classes=["car"]
)

from transformers import pipeline

pipe = pipeline("object-detection", model=checkpoint, image_processor=checkpoint)

results = pipe(url, threshold=0.3)

results

from PIL import ImageDraw

# visualize
# let's use Pillow's ImageDraw feature

annotated_image = image.copy()
draw = ImageDraw.Draw(annotated_image)

for i, result in enumerate(results):
  box = result["box"]
  color = tuple([int(x * 255) for x in COLORS[i]])
  xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
  draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=color, width=1)
  draw.text((xmin, ymin, xmax, ymax), text=result["label"])

annotated_image

checkpoints = [
  "ustc-community/dfine-nano-coco",
  "ustc-community/dfine-small-obj365",
  "ustc-community/dfine-xlarge-obj365",
  "ustc-community/dfine-medium-obj2coco",
]

for checkpoint in checkpoints:
  print(checkpoint)
  pipe = pipeline("object-detection", model=checkpoint, image_processor=checkpoint, device=device)
  results = pipe(image, threshold=0.3)

  annotated_image = image.copy()
  draw = ImageDraw.Draw(annotated_image)

  for i, result in enumerate(results):
    box = result["box"]
    color = tuple([int(x * 255) for x in COLORS[i]])
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=color, width=1)
    draw.text((xmin, ymin, xmax, ymax), text=result["label"])

  # display(annotated_image.resize([640, 480]))