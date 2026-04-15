---
library_name: transformers
license: apache-2.0
language:
  - en
pipeline_tag: object-detection
tags:
  - object-detection
  - vision
datasets:
  - coco
---
## D-FINE

### **Overview**

The D-FINE model was proposed in [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842) by
Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, Feng Wu

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber) with the help of [@qubvel-hf](https://huggingface.co/qubvel-hf)

This is the HF transformers implementation for D-FINE

_coco -> model trained on COCO

_obj365 -> model trained on Object365

_obj2coco -> model trained on Object365 and then finetuned on COCO

### **Performance**

D-FINE, a powerful real-time object detector that achieves outstanding localization precision by redefining the bounding box regression task in DETR models. D-FINE comprises two key components: Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). 

![COCO.png](https://huggingface.co/datasets/vladislavbro/images/resolve/main/COCO.PNG)

### **How to use**

```python
import torch
import requests

from PIL import Image
from transformers import DFineForObjectDetection, AutoImageProcessor

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("ustc-community/dfine-small-coco")
model = DFineForObjectDetection.from_pretrained("ustc-community/dfine-small-coco")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```

### **Training**

D-FINE is trained on COCO (Lin et al. [2014]) train2017 and validated on COCO val2017 dataset. We report the standard AP metrics (averaged over uniformly sampled IoU thresholds ranging from 0.50 − 0.95 with a step size of 0.05), and APval5000 commonly used in real scenarios.

### **Applications**
D-FINE is ideal for real-time object detection in diverse applications such as **autonomous driving**, **surveillance systems**, **robotics**, and **retail analytics**. Its enhanced flexibility and deployment-friendly design make it suitable for both edge devices and large-scale systems + ensures high accuracy and speed in dynamic, real-world environments.