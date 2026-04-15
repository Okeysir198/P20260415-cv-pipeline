---
license: apache-2.0
base_model: hustvl/yolos-tiny
tags:
- generated_from_trainer
- Workplace Safety
- Safety
datasets:
- hard-hat-detection
model-index:
- name: yolos-tiny-Hard_Hat_Detection
  results: []
language:
- en
pipeline_tag: object-detection
---

# yolos-tiny-Hard_Hat_Detection

This model is a fine-tuned version of [hustvl/yolos-tiny](https://huggingface.co/hustvl/yolos-tiny) on the hard-hat-detection dataset.

## Model description

For more information on how it was created, check out the following link: https://github.com/DunnBC22/Vision_Audio_and_Multimodal_Projects/blob/main/Computer%20Vision/Object%20Detection/Hard%20Hat%20Detection/Hard_Hat_Object_Detection_YOLOS.ipynb

## Intended uses & limitations

This model is intended to demonstrate my ability to solve a complex problem using technology.

## Training and evaluation data

Dataset Source: https://huggingface.co/datasets/keremberke/hard-hat-detection

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 8

### Training results

| Metric Name | IoU | Area| maxDets | Metric Value |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Average Precision (AP)| IoU=0.50:0.95 | all | maxDets=100 | 0.346 |
| Average Precision (AP)| IoU=0.50 | all | maxDets=100 | 0.747 |
| Average Precision (AP)| IoU=0.75 | all | maxDets=100 | 0.275 |
| Average Precision (AP)| IoU=0.50:0.95 | small | maxDets=100 | 0.128 |
| Average Precision (AP)| IoU=0.50:0.95 | medium | maxDets=100 | 0.343 |
| Average Precision (AP)| IoU=0.50:0.95 | large | maxDets=100 | 0.521 |
| Average Recall (AR)| IoU=0.50:0.95 | all | maxDets=1 | 0.188 |
| Average Recall (AR)| IoU=0.50:0.95 | all | maxDets=10 |  0.484 |
| Average Recall (AR)| IoU=0.50:0.95 | all | maxDets=100 | 0.558 |
| Average Recall (AR)| IoU=0.50:0.95 | small | maxDets=100 | 0.320 |
| Average Recall (AR)| IoU=0.50:0.95 | medium | maxDets=100 | 0.538 |
| Average Recall (AR)| IoU=0.50:0.95 | large | maxDets=100 | 0.743 |

### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu118
- Datasets 2.14.3
- Tokenizers 0.13.3


## License Notice
This model is a fine-tuned derivative of a pretrained model.
Users must comply with the original model license.


## Dataset Notice
This model was fine-tuned on third-party datasets which may have separate licenses or usage restrictions.