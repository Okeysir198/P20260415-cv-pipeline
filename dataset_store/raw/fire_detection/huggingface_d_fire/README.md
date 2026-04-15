---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: string
  - name: filename
    dtype: string
  splits:
  - name: train
    num_bytes: 4230807805
    num_examples: 17221
  - name: test
    num_bytes: 1110189309
    num_examples: 4306
  download_size: 3118077392
  dataset_size: 5340997114
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
