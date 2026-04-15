---
language: en
license: cc-by-nc-4.0
pipeline_tag: keypoint-detection
tags:
- sapiens
---

# Pose-Sapiens-0.6B-Torchscript

### Model Details
Sapiens is a family of vision transformers pretrained on 300 million human images at 1024 x 1024 image resolution. The pretrained models, when finetuned for human-centric vision tasks, generalize to in-the-wild conditions.
Sapiens-0.6B natively support 1K high-resolution inference. The resulting models exhibit remarkable generalization to in-the-wild data, even when labeled data is scarce or entirely synthetic.

- **Developed by:** Meta
- **Model type:** Vision Transformer
- **License:** Creative Commons Attribution-NonCommercial 4.0
- **Task:** pose
- **Format:** torchscript
- **File:** sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2

### Model Card
- **Image Size:** 1024 x 768 (H x W)
- **Num Parameters:** 0.664 B
- **FLOPs:** 2.583 TFLOPs
- **Patch Size:** 16 x 16
- **Embedding Dimensions:** 1280
- **Num Layers:** 32
- **Num Heads:** 16
- **Feedforward Channels:** 5120

### More Resources
- **Repository:** [https://github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens)
- **Paper:** [https://arxiv.org/abs/2408.12569](https://arxiv.org/abs/2408.12569)
- **Demo:** [https://huggingface.co/spaces/facebook/sapiens-pose](https://huggingface.co/spaces/facebook/sapiens-pose)
- **Project Page:** [https://about.meta.com/realitylabs/codecavatars/sapiens](https://about.meta.com/realitylabs/codecavatars/sapiens/)
- **Additional Results:** [https://rawalkhirodkar.github.io/sapiens](https://rawalkhirodkar.github.io/sapiens/)
- **HuggingFace Collection:** [https://huggingface.co/collections/facebook/sapiens-66d22047daa6402d565cb2fc](https://huggingface.co/collections/facebook/sapiens-66d22047daa6402d565cb2fc)

## Uses
Pose 0.6B model can be used for estimate 308 keypoints (body + face + hands + feet) on a single image.