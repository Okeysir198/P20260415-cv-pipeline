
footwear - v1 2022-10-31 12:41pm
==============================

This dataset was exported via roboflow.com on March 17, 2026 at 12:05 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 7983 images.
Footwear are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1920x1080 (Stretch)

The following augmentation was applied to create 3 versions of each source image:

The following transformations were applied to the bounding boxes of each image:
* Randomly crop between 0 and 20 percent of the bounding box
* Random rotation of between -17 and +17 degrees
* Random brigthness adjustment of between -25 and +25 percent


