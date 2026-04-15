---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: vehicle_classification
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.8466780238500852
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vehicle_classification

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5738
- Accuracy: 0.8467

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 15

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 147  | 1.4917          | 0.7785   |
| No log        | 2.0   | 294  | 1.0285          | 0.8160   |
| No log        | 3.0   | 441  | 0.8369          | 0.8177   |
| 1.294         | 4.0   | 588  | 0.7112          | 0.8399   |
| 1.294         | 5.0   | 735  | 0.6621          | 0.8313   |
| 1.294         | 6.0   | 882  | 0.5977          | 0.8450   |
| 0.4624        | 7.0   | 1029 | 0.5856          | 0.8518   |
| 0.4624        | 8.0   | 1176 | 0.6511          | 0.8160   |
| 0.4624        | 9.0   | 1323 | 0.6450          | 0.8365   |
| 0.4624        | 10.0  | 1470 | 0.6241          | 0.8296   |
| 0.2619        | 11.0  | 1617 | 0.6217          | 0.8382   |
| 0.2619        | 12.0  | 1764 | 0.6504          | 0.8177   |
| 0.2619        | 13.0  | 1911 | 0.5994          | 0.8433   |
| 0.1776        | 14.0  | 2058 | 0.5969          | 0.8433   |
| 0.1776        | 15.0  | 2205 | 0.5693          | 0.8569   |


### Framework versions

- Transformers 4.37.2
- Pytorch 2.1.0+cu121
- Datasets 2.17.1
- Tokenizers 0.15.2
