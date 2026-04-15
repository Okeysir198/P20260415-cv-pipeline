# Open Datasets Research: AI Traffic Light & Smart Parking (Bintulu, Malaysia)

Research date: 2026-03-19

---

## TABLE OF CONTENTS

1. [UC1: AI Traffic Light Datasets](#uc1-ai-traffic-light)
   - [1A. Vehicle Detection & Classification](#1a-vehicle-detection--classification)
   - [1B. Traffic Light / Signal Detection](#1b-traffic-light--signal-detection)
   - [1C. Traffic Flow / Queue / Counting](#1c-traffic-flow--queue--counting)
   - [1D. Emergency Vehicle Detection](#1d-emergency-vehicle-detection)
   - [1E. Adverse Weather / Tropical Conditions](#1e-adverse-weather--tropical-conditions)
   - [1F. Southeast Asian / Malaysian Traffic](#1f-southeast-asian--malaysian-traffic)
2. [UC2: Smart Parking Datasets](#uc2-smart-parking)
   - [2A. Parking Occupancy Detection](#2a-parking-occupancy-detection)
   - [2B. License Plate Recognition (ANPR/LPR)](#2b-license-plate-recognition-anprlpr)
   - [2C. Vehicle Type Classification](#2c-vehicle-type-classification)
   - [2D. Parking Violation / Anomaly Detection](#2d-parking-violation--anomaly-detection)
3. [Synthetic Data Generation Tools](#synthetic-data-generation-tools)
4. [Data Augmentation Strategies](#data-augmentation-strategies)
5. [Transfer Learning Recommendations](#transfer-learning-recommendations)
6. [Recommended Dataset Combinations](#recommended-dataset-combinations)

---

## UC1: AI TRAFFIC LIGHT

### 1A. Vehicle Detection & Classification

#### 1. COCO (Common Objects in Context)

| Field | Detail |
|---|---|
| **Size** | 330K images (200K annotated); train 118K, val 5K, test 20K |
| **Vehicle classes** | bicycle, car, motorcycle, bus, train, truck (+ person, 74 other classes) |
| **Annotation format** | COCO JSON (bbox + segmentation + keypoints) |
| **License** | CC BY 4.0 — **commercial OK** |
| **Download** | https://cocodataset.org/#download (~20 GB) |
| **Resolution** | Variable, typically 640x480 |
| **Camera angle** | Mixed (web-crawled, mostly eye-level) |
| **Relevance** | **5/5** — Gold standard pretrain source. All our models (YOLOX, D-FINE, RT-DETRv2) use COCO pretrained weights as initialization. |
| **Notes** | Not traffic-specific but indispensable for transfer learning. Vehicle classes well-represented. |

#### 2. BDD100K (Berkeley DeepDrive)

| Field | Detail |
|---|---|
| **Size** | 100K images (from 100K video clips); train 70K, val 10K, test 20K |
| **Vehicle classes** | car, truck, bus, bike, motor, rider, person, traffic sign, traffic light, train (10 detection classes) |
| **Annotation format** | Custom JSON (convertible to COCO); also lane markings + drivable area |
| **License** | BSD 3-Clause — **commercial OK** |
| **Download** | https://bdd-data.berkeley.edu/ (registration required, ~624 MB for detection subset) |
| **Resolution** | 1280x720 |
| **Camera angle** | Dashcam (front-facing) |
| **Relevance** | **4/5** — Excellent for traffic scene understanding. Dashcam angle differs from intersection overhead but diversity (time of day, weather) is very strong. |
| **Notes** | Includes drivable area + lane annotations. Weather/time labels available (day/night/dawn, clear/rain/fog/snow). Good for multi-task learning. |

#### 3. UA-DETRAC

| Field | Detail |
|---|---|
| **Size** | 140,000+ frames from 100 video sequences (10 hours at 25 fps) |
| **Vehicle classes** | car, bus, van, others |
| **Annotation format** | XML (custom); bounding boxes + attributes (occlusion, truncation, illumination, vehicle type) |
| **License** | **Academic use only** |
| **Download** | https://detrac-db.rit.albany.edu/ |
| **Resolution** | 960x540 |
| **Camera angle** | **Overhead / elevated** (road overpasses in Beijing/Tianjin) |
| **Relevance** | **5/5** — Best match for intersection overhead camera angle. Vehicle types match our needs. |
| **Notes** | Camera perspective closely matches traffic intersection surveillance. Chinese traffic but vehicle types are universal. Excellent for tracking algorithms. |

#### 4. VisDrone

| Field | Detail |
|---|---|
| **Size** | 288 video clips (261,908 frames) + 10,209 static images; 2.6M+ bounding boxes |
| **Vehicle classes** | pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, tricycle (10 classes) |
| **Annotation format** | Custom TXT (easily convertible to YOLO) |
| **License** | **Research use only** (request access) |
| **Download** | https://github.com/VisDrone/VisDrone-Dataset |
| **Resolution** | Variable (drone footage) |
| **Camera angle** | **Aerial / overhead** (drone-mounted cameras, 14 cities in China) |
| **Relevance** | **4/5** — Excellent overhead angle matching intersection cameras. Includes motorcycle/tricycle common in SE Asia. |
| **Notes** | Covers wide range of density, altitude, scene. Good for small object detection training. Tricycle/awning-tricycle classes relevant to Malaysian traffic. |

#### 5. MIO-TCD (MIOvision Traffic Camera Dataset)

| Field | Detail |
|---|---|
| **Size** | 786,702 total images; Localization: 137,743 frames with bboxes; Classification: 648,959 cropped objects |
| **Vehicle classes** | articulated truck, bicycle, bus, car, motorcycle, motorized vehicle, non-motorized vehicle, pedestrian, pickup truck, single unit truck, work van (11 classes) |
| **Annotation format** | Custom CSV/TXT |
| **License** | Research use (Miovision challenge terms) |
| **Download** | https://tcd.miovision.com/ |
| **Resolution** | Variable (hundreds of real traffic cameras) |
| **Camera angle** | **Traffic surveillance cameras** (overhead, various angles) |
| **Relevance** | **5/5** — Largest traffic camera dataset. Camera angles match intersection deployment. Fine-grained vehicle classes (pickup truck, work van, articulated truck). |
| **Notes** | Captured from surveillance cameras across US/Canada. Top methods achieve 96%+ accuracy on classification and 77% mAP on localization. Very realistic for deployment. |

#### 6. KITTI

| Field | Detail |
|---|---|
| **Size** | 14,999 images (train 7,481 + test 7,518) |
| **Vehicle classes** | car, pedestrian, cyclist, truck, tram, miscellaneous (9 annotation classes) |
| **Annotation format** | Custom TXT (per-image label files); also 3D bbox |
| **License** | CC BY-NC-SA 3.0 — **non-commercial only** |
| **Download** | https://www.cvlibs.net/dataset_store/kitti/ |
| **Resolution** | 1242x375 |
| **Camera angle** | Dashcam / vehicle-mounted |
| **Relevance** | **2/5** — Primarily autonomous driving benchmark, not traffic surveillance. Dashcam angle not ideal for intersection cameras. |
| **Notes** | Good for general vehicle detection R&D but less relevant for overhead fixed-camera deployment. |

#### 7. Cityscapes

| Field | Detail |
|---|---|
| **Size** | 5,000 fine annotations + 20,000 coarse annotations |
| **Vehicle classes** | car, truck, bus, motorcycle, bicycle, train, caravan, trailer (+ 22 other urban scene classes) |
| **Annotation format** | Pixel-level segmentation (custom JSON polygons) |
| **License** | **Non-commercial only** (custom terms) |
| **Download** | https://www.cityscapes-dataset.com/ (registration required) |
| **Resolution** | 2048x1024 |
| **Camera angle** | Dashcam (street-level) |
| **Relevance** | **2/5** — Segmentation focus, dashcam angle. Useful for scene understanding research but not directly for intersection detection. |
| **Notes** | 50 cities, excellent quality. Best for semantic segmentation pretraining. |

#### 8. nuScenes

| Field | Detail |
|---|---|
| **Size** | 1.44M camera images; 1,000 driving scenes, 40K annotated keyframes |
| **Vehicle classes** | car, truck, bus, motorcycle, bicycle, pedestrian, + construction vehicle, trailer, barrier, traffic cone |
| **Annotation format** | Custom JSON (3D bbox + tracking) |
| **License** | **Non-commercial only** (CC BY-NC-SA 4.0) |
| **Download** | https://www.nuscenes.org/ (registration required) |
| **Resolution** | 1600x900 (6 cameras, 360-degree) |
| **Camera angle** | Vehicle-mounted (Boston + Singapore) |
| **Relevance** | **3/5** — Singapore subset is SE Asian traffic. Vehicle-mounted angle less ideal but Singapore data valuable for domain adaptation to tropical environment. |
| **Notes** | Singapore subset contains SE Asian road conditions, motorcycle-heavy traffic. Multi-modal (camera + LiDAR + radar). |

#### 9. Waymo Open Dataset

| Field | Detail |
|---|---|
| **Size** | 1,150 scenes, 12M+ 3D labels, 12M+ 2D labels |
| **Vehicle classes** | vehicle, pedestrian, cyclist, sign |
| **Annotation format** | TFRecord (protobuf); modular v2.0 format |
| **License** | **Non-commercial only** (Waymo terms) |
| **Download** | https://waymo.com/open/download |
| **Resolution** | 1920x1280 (front), 1920x886 (side) |
| **Camera angle** | Vehicle-mounted (5 cameras) |
| **Relevance** | **2/5** — Premium quality but non-commercial license and vehicle-mounted perspective limit usefulness. |
| **Notes** | Excellent for R&D and benchmarking. Cannot be used in production/commercial systems. |

---

### 1B. Traffic Light / Signal Detection

#### 10. Bosch Small Traffic Light Dataset (BSTLD)

| Field | Detail |
|---|---|
| **Size** | 13,427 images; ~24,000 annotated traffic lights |
| **Classes** | green, red, yellow, off (4 classes) |
| **Annotation format** | YAML (bbox + state) |
| **License** | **Non-commercial only** |
| **Download** | https://zenodo.org/doi/10.5281/zenodo.12706045 (~18 GB total) |
| **Resolution** | 1280x720 |
| **Camera angle** | Dashcam |
| **Relevance** | **4/5** — Best available traffic light dataset. Critical for AI Traffic Light UC but dashcam angle; may need domain adaptation for intersection-mounted cameras. |
| **Notes** | HDR 12-bit raw + 8-bit RGB versions. Small traffic lights well-annotated. |

#### 11. LISA Traffic Light Dataset

| Field | Detail |
|---|---|
| **Size** | 43,007 frames; 113,888 annotated traffic lights |
| **Classes** | go (green), stop (red), warning (yellow), go-left, stop-left, warning-left |
| **Annotation format** | CSV (bbox + state) |
| **License** | Research use (UCSD) |
| **Download** | https://www.kaggle.com/dataset_store/mbornoe/lisa-traffic-light-dataset |
| **Resolution** | Variable |
| **Camera angle** | Dashcam (stereo) |
| **Relevance** | **4/5** — Large scale, includes directional arrow states. |
| **Notes** | Captured in San Diego. Varying light/weather conditions. Includes video sequences. |

#### 12. S2TLD (SJTU Small Traffic Light Dataset)

| Field | Detail |
|---|---|
| **Size** | 5,786 images; 14,130 traffic light instances |
| **Classes** | red, yellow, green, off, wait_on (5 classes) |
| **Annotation format** | PASCAL VOC XML |
| **License** | Research use |
| **Download** | https://github.com/Thinklab-SJTU/S2TLD |
| **Resolution** | 1080x1920 and 720x1280 |
| **Camera angle** | Dashcam (Chinese roads) |
| **Relevance** | **3/5** — Good for small traffic light detection. Chinese traffic lights but detection task is transferable. |
| **Notes** | Focus on small/distant traffic lights. Good complement to BSTLD. |

#### 13. DriveU Traffic Light Dataset (DTLD)

| Field | Detail |
|---|---|
| **Size** | 230,000+ images; 340,000+ traffic light annotations |
| **Classes** | Red, yellow, green, red-yellow (+ arrow states, pedestrian signals) |
| **Annotation format** | PASCAL VOC XML |
| **License** | Research use |
| **Download** | https://www.uni-ulm.de/en/in/driveu/projects/driveu-traffic-light-dataset/ |
| **Resolution** | 2048x1024 |
| **Camera angle** | Dashcam (German roads) |
| **Relevance** | **3/5** — Largest traffic light dataset. European signals differ from Malaysian but detection model transfers well. |
| **Notes** | Very detailed state annotations. Good for pretraining traffic light detectors. |

---

### 1C. Traffic Flow / Queue / Counting

#### 14. TRANCOS

| Field | Detail |
|---|---|
| **Size** | 1,244 images; 46,796 vehicle annotations |
| **Classes** | Vehicle (counting only, no class distinction) |
| **Annotation format** | Dot annotations (center points) + density maps |
| **License** | Research use |
| **Download** | https://gram.web.uah.es/data/dataset_store/trancos/index.html |
| **Resolution** | Variable (traffic surveillance cameras) |
| **Camera angle** | **Overhead / surveillance** (Spanish highway cameras) |
| **Relevance** | **4/5** — Perfect camera angle for traffic counting. Congestion scenarios ideal for queue length estimation. |
| **Notes** | From Spanish DGT surveillance cameras. Focus on extremely overlapping vehicles. Good for density estimation models. |

#### 15. WebCamT (CityCam)

| Field | Detail |
|---|---|
| **Size** | 60,000 annotated frames (from 60M total frames over 4 weeks, 1.4 TB raw) |
| **Classes** | taxi, black sedan, other car, truck, van, bus (6 vehicle types + orientation) |
| **Annotation format** | Custom (bbox + vehicle type + orientation + density count) |
| **License** | Research use |
| **Download** | https://github.com/Lotuslisa/WebCamT |
| **Resolution** | Variable (webcam feeds) |
| **Camera angle** | **Fixed surveillance webcams** (NYC traffic cameras) |
| **Relevance** | **4/5** — Real-world traffic cameras, density counting, vehicle type classification. Perfect for traffic flow analysis use case. |
| **Notes** | Includes vehicle counting ground truth. 4 orientation classes. Long-term temporal data. |

#### 16. AI City Challenge Datasets

| Field | Detail |
|---|---|
| **Size** | ~9 hours of video from 20 vantage points (counting); 190K+ images (re-id) |
| **Classes** | Varies by track: multi-class vehicle counting, re-identification, anomaly detection |
| **Annotation format** | Custom per track |
| **License** | Research use (requires application form) |
| **Download** | https://www.aicitychallenge.org/ai-city-challenge-dataset-access/ |
| **Resolution** | 960p or better, 10 fps |
| **Camera angle** | **Intersection, highway, city street** (mixed surveillance angles) |
| **Relevance** | **5/5** — Directly targets traffic management at intersections. Multi-movement vehicle counting track is exactly our UC. |
| **Notes** | Annual challenge datasets. 2021 Track 1 (multi-class multi-movement counting) and Track 2 (vehicle re-id) are most relevant. Request access via form. |

---

### 1D. Emergency Vehicle Detection

#### 17. GAN-Augmented Emergency Vehicle Dataset

| Field | Detail |
|---|---|
| **Size** | 20,000 images |
| **Classes** | ambulance, police car, fire truck |
| **Annotation format** | Standard image classification labels |
| **License** | Open (GitHub) |
| **Download** | https://github.com/Shatnawi-Moath/EMERGENCY-VEHICLES-ON-ROAD-NETWORKS-A-NOVEL-GENERATED-DATASET-USING-GANs |
| **Resolution** | Variable |
| **Camera angle** | Mixed (real-world + GAN-generated) |
| **Relevance** | **3/5** — Good starting point for emergency vehicle classification. May need additional bbox annotation. |
| **Notes** | Augmented with GANs for diversity. Classification only, not detection. Would need COCO vehicle pretrained detector + emergency vehicle classifier. |

#### 18. Kaggle Emergency Vehicles Identification

| Field | Detail |
|---|---|
| **Size** | ~2,000 images |
| **Classes** | emergency vehicle, non-emergency vehicle |
| **Annotation format** | Classification labels (no bboxes natively; YOLO bboxes generated via pretrained models) |
| **License** | Kaggle terms |
| **Download** | https://www.kaggle.com/dataset_store/abhisheksinghblr/emergency-vehicles-identification |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **2/5** — Small, binary classification only. Useful as supplementary data. |
| **Notes** | Needs bbox annotation. Best combined with other datasets. |

#### 19. Emergency Vehicle Siren Audio Dataset

| Field | Detail |
|---|---|
| **Size** | 1,800 audio files (900 siren + 900 road noise); 3-15 seconds each |
| **Classes** | siren sound, road noise (binary); Extended version: firetruck, ambulance, police, traffic noise |
| **Annotation format** | WAV files + extracted MFCC features in CSV |
| **License** | Open (GitHub + Figshare) |
| **Download** | https://figshare.com/articles/dataset/Large-Scale_Dataset_for_Emergency_Vehicle_Siren_and_Road_Noises/17560865 |
| **Resolution** | N/A (audio) |
| **Camera angle** | N/A |
| **Relevance** | **3/5** — Multi-modal detection: combine visual + audio for higher accuracy emergency vehicle detection. |
| **Notes** | MFCC features pre-extracted. Good for multi-modal approach: camera visual detection + microphone audio confirmation. |

---

### 1E. Adverse Weather / Tropical Conditions

#### 20. DAWN (Detection in Adverse Weather Nature)

| Field | Detail |
|---|---|
| **Size** | 1,000 images |
| **Classes** | Vehicle bounding boxes (multi-class) |
| **Annotation format** | PASCAL VOC XML |
| **License** | Open (Mendeley Data) |
| **Download** | https://data.mendeley.com/dataset_store/766ygrbt8y/3 |
| **Resolution** | Variable |
| **Camera angle** | Mixed (urban, highway, freeway) |
| **Relevance** | **3/5** — Small but specifically targets adverse weather. Rain subset very relevant for tropical Bintulu climate. |
| **Notes** | Four weather conditions: fog, snow, rain, sandstorms. Rain subset most relevant for Malaysia. |

#### 21. ACDC (Adverse Conditions Dataset with Correspondences)

| Field | Detail |
|---|---|
| **Size** | 8,012 images (4,006 adverse + 4,006 normal correspondences) |
| **Classes** | 19 Cityscapes classes (car, truck, bus, motorcycle, bicycle, person, traffic light, traffic sign, etc.) |
| **Annotation format** | Pixel-level panoptic annotations (Cityscapes format) |
| **License** | Research use |
| **Download** | https://acdc.vision.ee.ethz.ch/ |
| **Resolution** | 1920x1080 |
| **Camera angle** | Dashcam |
| **Relevance** | **4/5** — Fog, night, rain, snow conditions with paired normal images. Rain + night subsets critical for Bintulu's tropical climate (heavy afternoon rain, high humidity). |
| **Notes** | Each adverse image has a matching normal-condition image of the same scene. Excellent for domain adaptation training. |

---

### 1F. Southeast Asian / Malaysian Traffic

#### 22. IDD (Indian Driving Dataset)

| Field | Detail |
|---|---|
| **Size** | 10,000 images with 34 classes; from 182 drive sequences |
| **Classes** | car, truck, bus, motorcycle, bicycle, auto-rickshaw, pedestrian, animal, + 26 more |
| **Annotation format** | Cityscapes-compatible (polygon segmentation + bbox) |
| **License** | Research use |
| **Download** | https://idd.insaan.iiit.ac.in/ |
| **Resolution** | 1080p |
| **Camera angle** | Dashcam |
| **Relevance** | **4/5** — Closest available proxy for SE Asian traffic. Dense motorcycle traffic, auto-rickshaws (similar to Malaysian three-wheelers), unstructured traffic. |
| **Notes** | Indian traffic patterns share many characteristics with Malaysian traffic: motorcycle density, mixed vehicle types, tropical weather. Good domain adaptation source. |

#### 23. DriveIndia

| Field | Detail |
|---|---|
| **Size** | 66,986 images (train 53,586, val 6,700, test 6,700) |
| **Classes** | bicycle, car, motorcycle, bus, commercial vehicle, truck, autorickshaw, + others |
| **Annotation format** | COCO JSON |
| **License** | Research use |
| **Download** | https://arxiv.org/abs/2507.19912 (check paper for download link) |
| **Resolution** | High-resolution RGB |
| **Camera angle** | Dashcam (120+ hours of driving) |
| **Relevance** | **4/5** — Large-scale, diverse vehicle types common in developing Asian countries. |
| **Notes** | Released 2025. Autorickshaw class relevant to Malaysian traffic. Dense mixed traffic scenarios. |

#### 24. HELMET Dataset (Myanmar Motorcycle Traffic)

| Field | Detail |
|---|---|
| **Size** | 910 video clips (91,000 frames); 10-second clips at 10fps |
| **Classes** | Motorcycle bounding boxes + helmet usage annotations |
| **Annotation format** | Custom (bbox per motorcycle) |
| **License** | Research use |
| **Download** | https://hyper.ai/en/dataset_store/32927 |
| **Resolution** | 1920x1080 |
| **Camera angle** | Mixed surveillance |
| **Relevance** | **4/5** — SE Asian motorcycle-heavy traffic. Myanmar traffic very similar to Malaysian roads. |
| **Notes** | Directly applicable to Bintulu where motorcycle density is high. Also useful for our helmet detection model. |

#### 25. TFP-BD (Bangladesh Traffic Flow & Pedestrian Dataset)

| Field | Detail |
|---|---|
| **Size** | Research dataset (recently published 2025) |
| **Classes** | Vehicles + pedestrians in dense South Asian traffic |
| **Annotation format** | Custom |
| **License** | Research use |
| **Download** | https://www.sciencedirect.com/science/article/pii/S2352340925001301 (check paper for access) |
| **Resolution** | Variable |
| **Camera angle** | Urban road |
| **Relevance** | **3/5** — South Asian traffic patterns similar to SE Asian. Useful for domain knowledge. |
| **Notes** | Bangladesh urban traffic shares characteristics with Malaysian traffic: motorcycle density, pedestrian mixing. |

---

## UC2: SMART PARKING

### 2A. Parking Occupancy Detection

#### 26. PKLot

| Field | Detail |
|---|---|
| **Size** | 12,416 images; ~695,900 segmented parking space instances |
| **Classes** | occupied, empty (per parking space) |
| **Annotation format** | XML (parking space coordinates + occupancy label) |
| **License** | **CC BY 4.0 — commercial OK** |
| **Download** | https://huggingface.co/dataset_store/Voxel51/PKLot or https://public.roboflow.com/object-detection/pklot (~4.6 GB) |
| **Resolution** | Variable (3 parking lots) |
| **Camera angle** | **Overhead / elevated** (CCTV cameras looking down at parking lots) |
| **Relevance** | **5/5** — Gold standard parking occupancy dataset. Three different lots, three weather conditions (sunny, cloudy, rainy). |
| **Notes** | From PUCPR, UFPR04, UFPR05 parking lots in Brazil. Commercial license makes it deployment-ready. |

#### 27. CNRPark+EXT

| Field | Detail |
|---|---|
| **Size** | ~150,000 labeled patches (occupied/vacant); 164 parking spaces |
| **Classes** | occupied, vacant |
| **Annotation format** | Image patches with folder-based labels |
| **License** | Research use |
| **Download** | http://cnrpark.it/ (~449.5 MB for patches) |
| **Resolution** | Variable (patches from 9 cameras) |
| **Camera angle** | **Overhead** (rooftop cameras) |
| **Relevance** | **5/5** — Large-scale, multiple cameras/viewpoints, weather variation. Ideal training data for occupancy classification. |
| **Notes** | Collected July 2015 from Pisa, Italy parking lot. 9 cameras at different angles. Good for testing viewpoint robustness. |

#### 28. ACPDS (Aerial Car Parking Detection/Segmentation)

| Field | Detail |
|---|---|
| **Size** | 293 images; each from unique viewpoint |
| **Classes** | occupied, vacant parking spaces |
| **Annotation format** | Segmentation masks |
| **License** | **MIT — commercial OK** |
| **Download** | https://github.com/martin-marek/parking-space-occupancy |
| **Resolution** | GoPro Hero 6 (high-res) |
| **Camera angle** | **Overhead** (~10m height) |
| **Relevance** | **4/5** — Unique viewpoints per image, MIT license for commercial use. Small but high quality. |
| **Notes** | Each parking lot in train/val/test is unique (no overlap). Baseline achieves 98% accuracy. |

#### 29. CARPK

| Field | Detail |
|---|---|
| **Size** | ~90,000 car annotations; drone photography from 4 parking lots |
| **Classes** | Car (bounding boxes for counting) |
| **Annotation format** | Bounding boxes (custom) |
| **License** | Research use (EULA required) |
| **Download** | https://lafi.github.io/LPN/ (~2 GB, password after EULA) |
| **Resolution** | Drone imagery (~40m altitude) |
| **Camera angle** | **Aerial/overhead** (drone) |
| **Relevance** | **4/5** — Large-scale car counting in parking lots. Aerial view matches some smart parking camera angles. |
| **Notes** | First and largest drone-view vehicle counting dataset. Good for parking capacity monitoring. |

#### 30. Parking Space Detection Dataset (UniqueData/HuggingFace)

| Field | Detail |
|---|---|
| **Size** | Variable (check HuggingFace for latest) |
| **Classes** | free, not_free, partially_free |
| **Annotation format** | Standard detection format |
| **License** | Check HuggingFace terms |
| **Download** | https://huggingface.co/dataset_store/UniqueData/parking-space-detection-dataset |
| **Resolution** | Variable |
| **Camera angle** | Overhead |
| **Relevance** | **3/5** — Three-class occupancy (includes "partially free" which is useful for real-world deployment). |
| **Notes** | Newer dataset, good complement to PKLot/CNRPark. |

#### 31. Kaggle Parking Space Detection & Classification

| Field | Detail |
|---|---|
| **Size** | Available on Kaggle |
| **Classes** | Parking space detection + occupancy classification |
| **Annotation format** | Standard |
| **License** | Kaggle terms |
| **Download** | https://www.kaggle.com/dataset_store/trainingdatapro/parking-space-detection-dataset |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **3/5** — Supplementary dataset for parking occupancy. |

---

### 2B. License Plate Recognition (ANPR/LPR)

#### 32. CCPD (Chinese City Parking Dataset)

| Field | Detail |
|---|---|
| **Size** | 290K+ images of unique license plates |
| **Classes** | License plate detection + character recognition (Chinese provinces, alphabets A-Z excluding O, digits 0-9) |
| **Annotation format** | Filename-encoded annotations (LP number, bbox vertices, brightness, blur, tilt) |
| **License** | **MIT — commercial OK** |
| **Download** | https://github.com/detectRecog/CCPD |
| **Resolution** | Variable |
| **Camera angle** | Parking lot cameras |
| **Relevance** | **4/5** — Largest LP dataset. Chinese plates differ from Malaysian but detection model architecture transfers well. MIT license. |
| **Notes** | Two versions: CCPD 2019 (blue plates) + CCPD 2020/Green (new energy plates). Good for pretraining plate detector; fine-tune on Malaysian plates. |

#### 33. Global License Plate Dataset (GLPD)

| Field | Detail |
|---|---|
| **Size** | 5M+ images from 74 countries |
| **Classes** | LP detection + character recognition + vehicle make/color/model |
| **Annotation format** | LP characters, segmentation masks, corner vertices, vehicle attributes |
| **License** | Research use (data from Platesmania.com) |
| **Download** | https://arxiv.org/abs/2405.10949 (check paper for access) |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **5/5** — **CRITICAL**: 74 countries likely includes Malaysia or similar SE Asian plates. Largest multi-country LP dataset. |
| **Notes** | Published 2024. Includes diverse vehicle types (motorcycles, trucks, buses, cars). Check if Malaysia is among the 74 countries. |

#### 34. UniDataPro License Plate Detection

| Field | Detail |
|---|---|
| **Size** | 1.2M+ images with OCR from 32+ countries |
| **Classes** | Plate number, country, bbox, mask |
| **Annotation format** | Custom (bbox + OCR text) |
| **License** | Commercial dataset (check pricing) |
| **Download** | https://huggingface.co/dataset_store/UniDataPro/license-plate-detection |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **4/5** — Multi-country coverage. Check if Malaysian plates included. |
| **Notes** | May require commercial license for full access. |

#### 35. Malaysian Number Plate Dataset (Roboflow)

| Field | Detail |
|---|---|
| **Size** | Small (community-contributed) |
| **Classes** | Malaysian license plate detection |
| **Annotation format** | YOLO TXT, COCO JSON, Darknet (multiple export formats) |
| **License** | Roboflow community terms |
| **Download** | https://universe.roboflow.com/malaysian-number-plate/malaysian-number-plate/dataset/1 |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **5/5** — **CRITICAL**: Only known Malaysian-specific plate dataset. Small but directly relevant for fine-tuning. |
| **Notes** | May be small. Use as fine-tuning data on top of CCPD/GLPD pretrained detector. Malaysian plate formats: ABC 1234, W 1234 A, Sarawak Q/QS/QA/QK prefixes. |

#### 36. Gocar Malaysia Car Plate Number (Roboflow)

| Field | Detail |
|---|---|
| **Size** | Community-contributed |
| **Classes** | Malaysian car plate number |
| **Annotation format** | YOLO format available |
| **License** | Roboflow terms |
| **Download** | https://universe.roboflow.com/gocar/malaysia-car-plate-number |
| **Resolution** | Variable |
| **Camera angle** | Mixed |
| **Relevance** | **5/5** — Another Malaysian-specific plate dataset. |
| **Notes** | From Gocar (Malaysian car sharing company). Real Malaysian plates. |

#### 37. UFPR-ALPR

| Field | Detail |
|---|---|
| **Size** | 4,500 images (150 vehicles, 30K+ LP characters) |
| **Classes** | Vehicle type, LP detection, LP characters, character positions |
| **Annotation format** | Custom TXT (per-image: vehicle bbox, type, LP bbox, LP text, character positions) |
| **License** | **Academic use only** (requires university email approval) |
| **Download** | https://github.com/raysonlaroca/ufpr-alpr-dataset |
| **Resolution** | 1920x1080 |
| **Camera angle** | Vehicle-mounted (moving camera) |
| **Relevance** | **3/5** — Well-annotated but Brazilian plates. Good for ALPR pipeline architecture development. |
| **Notes** | Split: 40% train, 40% test, 20% val. Three different cameras. Detailed annotations include all character positions. |

#### 38. RodoSol-ALPR

| Field | Detail |
|---|---|
| **Size** | 20,000 images |
| **Classes** | Vehicle type (car/motorcycle), LP layout (Brazilian/Mercosur), LP text, corner vertices |
| **Annotation format** | Custom TXT (4 corner vertices instead of bbox — enables rectification) |
| **License** | **Academic use only** (requires university email approval) |
| **Download** | https://github.com/raysonlaroca/rodosol-alpr-dataset |
| **Resolution** | 1280x720 |
| **Camera angle** | **Fixed toll cameras** (static, similar to parking entry cameras) |
| **Relevance** | **3/5** — Fixed camera angle matches parking entry/exit cameras. Corner-based annotation useful for plate rectification. |
| **Notes** | Static camera setup very similar to parking gate ANPR deployment. Good architecture reference. |

#### 39. AOLP (Application-Oriented License Plate)

| Field | Detail |
|---|---|
| **Size** | 2,049 images (Taiwan plates) |
| **Classes** | Three subsets: Access Control (681), Law Enforcement (757), Road Patrol (611) |
| **Annotation format** | Custom |
| **License** | Research use (requires written approval) |
| **Download** | https://github.com/AvLab-CV/AOLP |
| **Resolution** | Variable |
| **Camera angle** | Mixed (toll station, roadside, patrol vehicle) |
| **Relevance** | **3/5** — Taiwan plates are Asian format. Access Control subset matches parking gate scenario. |
| **Notes** | Small but three distinct scenarios. Access Control subset directly relevant to parking entry. |

#### Malaysian Plate Format Reference

For custom data collection / annotation in Bintulu:
- Standard format: `ABC 1234` or `W 1234 A`
- Sarawak plates: `Q` prefix + division code
  - `QK` = Kuching (old), `QA` = Kuching (current), `QAA-QAY` = extended
  - `QS` = Sibu/Mukah
  - Letters I, O, Z not used; no leading zeroes
- Special plates: taxi, diplomatic, military, Putrajaya (F/Fxx)

---

### 2C. Vehicle Type Classification

#### 40. Stanford Cars

| Field | Detail |
|---|---|
| **Size** | 16,185 images (train 8,144 + test 8,041) |
| **Classes** | 196 fine-grained car classes (make + model + year) |
| **Annotation format** | Bounding boxes + class labels |
| **License** | Research use |
| **Download** | https://www.kaggle.com/dataset_store/eduardo4jesus/stanford-cars-dataset |
| **Resolution** | Variable |
| **Camera angle** | Mixed (web images) |
| **Relevance** | **2/5** — Fine-grained classification useful for vehicle re-id but 196 Western car classes less relevant for Malaysian vehicles. |
| **Notes** | Good for transfer learning backbone pretraining for vehicle classification. |

#### 41. CompCars

| Field | Detail |
|---|---|
| **Size** | 136,727 web images (1,716 models from 163 makes) + 44,481 surveillance images |
| **Classes** | Car make, model, year, viewpoint, attributes (max speed, displacement, doors, seats, type) |
| **Annotation format** | Bounding boxes + attributes |
| **License** | Research use |
| **Download** | https://mmlab.ie.cuhk.edu.hk/dataset_store/comp_cars/ |
| **Resolution** | Variable |
| **Camera angle** | Web images + **surveillance camera images** |
| **Relevance** | **3/5** — Surveillance subset relevant. Large variety of vehicle types. |
| **Notes** | Two parts: web-nature (diverse views) and surveillance-nature (frontal view from cameras). Surveillance part useful for parking camera deployment. |

#### 42. BoxCars116k

| Field | Detail |
|---|---|
| **Size** | 116,000 images |
| **Classes** | Fine-grained vehicle make/model from surveillance cameras |
| **Annotation format** | 3D bbox + fine-grained labels |
| **License** | Research use |
| **Download** | https://medusa.fit.vutbr.cz/traffic/data/BoxCars116k.zip |
| **Resolution** | Variable (surveillance camera crops) |
| **Camera angle** | **Surveillance cameras** (elevated/overhead) |
| **Relevance** | **4/5** — Surveillance camera angle matches parking/traffic deployment. Fine-grained vehicle classification from overhead. |
| **Notes** | Includes 3D bounding boxes from surveillance views. Good for vehicle type classification at parking gates. |

#### 43. VeRi-776

| Field | Detail |
|---|---|
| **Size** | 49,357 images of 776 vehicles |
| **Classes** | Vehicle identity (re-identification) + type, color |
| **Annotation format** | Image + metadata |
| **License** | Research use |
| **Download** | Request from authors |
| **Resolution** | Variable |
| **Camera angle** | Surveillance (20 cameras) |
| **Relevance** | **3/5** — Vehicle re-identification useful for parking lot tracking (entry/exit matching). |
| **Notes** | 20 camera viewpoints per vehicle. Good for multi-camera parking re-identification. |

---

### 2D. Parking Violation / Anomaly Detection

#### 44. UCF-Crime

| Field | Detail |
|---|---|
| **Size** | 1,900 untrimmed videos (128 hours total) |
| **Classes** | 13 anomaly types including abuse, arrest, arson, assault, burglary, explosion, fighting, road accident, robbery, shooting, shoplifting, stealing, vandalism |
| **Annotation format** | Video-level labels (weakly supervised) |
| **License** | Research use |
| **Download** | https://www.crcv.ucf.edu/projects/real-world/ |
| **Resolution** | Variable |
| **Camera angle** | Surveillance cameras |
| **Relevance** | **2/5** — General anomaly detection, not parking-specific. But methodology transfers to parking anomaly detection. |
| **Notes** | Weakly supervised (video-level labels only). Good for learning anomaly detection framework. |

#### 45. ShanghaiTech Campus

| Field | Detail |
|---|---|
| **Size** | 130 abnormal events; 270,000+ training frames; 13 scenes |
| **Classes** | Normal / abnormal (pixel-level GT for abnormal events) |
| **Annotation format** | Pixel-level anomaly masks |
| **License** | Research use |
| **Download** | https://svip-lab.github.io/dataset/campus_dataset.html |
| **Resolution** | Variable |
| **Camera angle** | **Fixed surveillance cameras** |
| **Relevance** | **3/5** — Fixed camera surveillance, pixel-level anomaly detection. Good methodology for parking violation detection. |
| **Notes** | Complex light conditions and camera angles. 13 different scenes. Anomaly types include loitering and unusual movement patterns applicable to parking context. |

#### 46. CUHK Avenue

| Field | Detail |
|---|---|
| **Size** | 37 videos (16 train + 21 test); 47 abnormal events |
| **Classes** | Normal / abnormal (throwing objects, loitering, running) |
| **Annotation format** | Frame-level + pixel-level annotations |
| **License** | Research use |
| **Download** | http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html |
| **Resolution** | 640x360 |
| **Camera angle** | Fixed surveillance |
| **Relevance** | **2/5** — General anomaly detection. Loitering detection applicable to parking security. |

#### 47. TU-DAT (Temple University Data on Anomalous Traffic)

| Field | Detail |
|---|---|
| **Size** | ~210 videos; 17,255 accident keyframes + 505,245 standard frames |
| **Classes** | Road accidents / anomalies |
| **Annotation format** | Custom (keyframe annotations) |
| **License** | Research use |
| **Download** | Check https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit |
| **Resolution** | 24-30 FPS video |
| **Camera angle** | Roadside cameras |
| **Relevance** | **3/5** — Traffic anomaly detection from fixed cameras. Methodology applicable to parking anomalies. |
| **Notes** | Published 2025. Roadside camera perspective. Focus on accident detection but framework transfers to parking violations. |

---

## SYNTHETIC DATA GENERATION TOOLS

### CARLA Simulator

| Field | Detail |
|---|---|
| **Purpose** | Full autonomous driving simulation with sensor suite |
| **Vehicle types** | Cars, trucks, buses, motorcycles, bicycles, pedestrians |
| **Sensors** | RGB cameras, LiDAR, radar, depth, semantic segmentation |
| **Traffic** | Configurable traffic density, patterns, signal timing |
| **Weather** | Rain, fog, sun position, wet roads, puddles |
| **License** | MIT — **commercial OK** |
| **URL** | https://carla.org/ |
| **Relevance for Bintulu** | Can simulate tropical rain, high-density motorcycle traffic, intersection scenarios. Configure overhead camera to match deployment angle. |
| **Key capability** | Generate unlimited perfectly-annotated training data for domain adaptation. Combine CARLA synthetic + real data for robust models. |

### SUMO (Simulation of Urban MObility)

| Field | Detail |
|---|---|
| **Purpose** | Microscopic traffic simulation |
| **Key features** | Vehicle counting, queue length estimation, traffic signal optimization, turning movement analysis |
| **Data sources** | Import from OpenStreetMap, real traffic counts |
| **License** | EPL-2.0 — **commercial OK** |
| **URL** | https://eclipse.dev/sumo/ |
| **Relevance for Bintulu** | Model Bintulu intersections for traffic signal optimization. Import real turning movement counts to calibrate. Generate ground truth for queue length / density estimation. |
| **Key capability** | Can model exact Bintulu intersection geometry + traffic patterns. Output vehicle counts per approach/direction. |

### Unity Perception

| Field | Detail |
|---|---|
| **Purpose** | Synthetic data generation for computer vision (object detection, segmentation, pose estimation) |
| **Features** | Randomization framework, multiple labeler types, COCO/PASCAL VOC export |
| **License** | Unity license (check terms for commercial use) |
| **URL** | https://github.com/Unity-Technologies/com.unity.perception |
| **Relevance for Bintulu** | Generate synthetic parking lot scenes, customize vehicle types, lighting conditions. |
| **Key capability** | Place 3D vehicle models in parking lot scene, randomize occupancy, generate perfect annotations. |

---

## DATA AUGMENTATION STRATEGIES

### For Tropical / Bintulu Conditions

1. **Rain simulation**: Add synthetic rain streaks, wet road reflections, water droplets on lens. Use DAWN rain subset as reference.

2. **Glare / high sun**: Simulate tropical midday glare (near-equatorial sun angle). Brightness and contrast augmentation. Lens flare overlay.

3. **High humidity haze**: Light fog/haze overlay to simulate humid tropical air. Use ACDC fog subset as domain adaptation source.

4. **Nighttime**: BDD100K night subset for pretraining. Adjust brightness/gamma for night-time traffic light detection.

5. **Motorcycle-specific**: Horizontal flip (reversed handedness), scale variation (motorcycles appear small at distance), occlusion (motorcycles partially hidden by larger vehicles).

6. **Standard augmentations** (already in our pipeline):
   - Mosaic (4-image composition)
   - MixUp (alpha blending)
   - HSV jitter
   - Random affine (rotation, scale, shear)
   - Random crop and resize

7. **Domain-specific augmentations**:
   - Copy-paste augmentation: paste Malaysian vehicle crops onto intersection background
   - Style transfer: transfer CARLA synthetic scenes to look like Bintulu camera feeds
   - Cutout / GridMask: simulate partial occlusion common at intersections

---

## TRANSFER LEARNING RECOMMENDATIONS

### Strategy for AI Traffic Light (UC1)

```
Phase 1: Pretrain on COCO (vehicle classes)
   |
Phase 2: Fine-tune on BDD100K + MIO-TCD (traffic-specific)
   |
Phase 3: Fine-tune on UA-DETRAC + VisDrone (overhead angle)
   |
Phase 4: Domain adapt with Bintulu-collected data (50-200 images)
   |
Phase 5: Augment with CARLA synthetic (tropical weather, intersection layout)
```

**Recommended base models (Apache 2.0):**
- Vehicle detection: D-FINE-S or YOLOX-M pretrained on COCO, fine-tuned on traffic data
- Traffic light detection: D-FINE-N (small, fast) fine-tuned on BSTLD + LISA
- Emergency vehicle: YOLOX-M vehicle detector + emergency vehicle classifier head
- Counting/queue: D-FINE-S detector + ByteTrack tracker + line-crossing counter

### Strategy for Smart Parking (UC2)

```
Phase 1: Occupancy detection
   PKLot + CNRPark+EXT pretrain -> fine-tune on Bintulu parking lot images
   Model: MobileNetV3-Small (lightweight, per-bay classifier)

Phase 2: License plate detection
   CCPD + GLPD pretrain -> fine-tune on Malaysian plate datasets (Roboflow)
   Model: D-FINE-N (plate detection) + CRNN/TrOCR (character recognition)

Phase 3: Vehicle classification
   COCO pretrain -> CompCars/BoxCars fine-tune -> Bintulu vehicle data
   Model: YOLOX-Tiny or D-FINE-N

Phase 4: Anomaly detection (double parking, loitering)
   Train on normal parking behavior, detect deviations
   Model: Autoencoder / temporal consistency check on detection outputs
```

**Malaysian plate OCR pipeline:**
1. Plate detector: D-FINE-N fine-tuned on CCPD + Malaysian Roboflow data
2. Plate rectifier: 4-corner detection (from RodoSol approach) + perspective transform
3. Character recognizer: TrOCR or CRNN trained on:
   - Synthetic Malaysian plates (generate using known formats: QA/QS/QK + alphanumeric)
   - Real Malaysian plate crops from Roboflow datasets
   - CCPD data for general character recognition transfer

---

## RECOMMENDED DATASET COMBINATIONS

### UC1: AI Traffic Light — Priority Datasets

| Priority | Dataset | Purpose | License |
|---|---|---|---|
| **P0** | COCO | Pretrain base detector | CC BY 4.0 |
| **P0** | BDD100K | Traffic scene fine-tuning | BSD 3-Clause |
| **P0** | UA-DETRAC | Overhead angle vehicle detection | Academic |
| **P1** | MIO-TCD | Traffic camera vehicle classification (11 classes) | Research |
| **P1** | BSTLD + LISA | Traffic light detection | Non-commercial / Research |
| **P1** | AI City Challenge | Intersection counting + flow | Research |
| **P2** | VisDrone | Aerial/overhead angle + small objects | Research |
| **P2** | ACDC | Adverse weather domain adaptation | Research |
| **P2** | IDD / DriveIndia | SE Asian traffic patterns | Research |
| **P2** | HELMET (Myanmar) | Motorcycle-heavy traffic | Research |
| **P3** | Emergency vehicle datasets | Multi-class emergency detection | Various |
| **P3** | CARLA synthetic | Tropical weather + intersection layout | MIT |

### UC2: Smart Parking — Priority Datasets

| Priority | Dataset | Purpose | License |
|---|---|---|---|
| **P0** | PKLot | Parking occupancy detection | CC BY 4.0 |
| **P0** | CNRPark+EXT | Parking occupancy (large scale) | Research |
| **P0** | CCPD | License plate detection pretrain | MIT |
| **P0** | Malaysian plate (Roboflow) | Malaysian plate fine-tuning | Roboflow terms |
| **P1** | GLPD (74 countries) | Multi-country plate detection | Research |
| **P1** | CARPK | Parking lot vehicle counting (aerial) | Research (EULA) |
| **P1** | BoxCars116k | Surveillance vehicle classification | Research |
| **P2** | ACPDS | Parking space detection (MIT license) | MIT |
| **P2** | CompCars | Vehicle type from surveillance | Research |
| **P2** | RodoSol-ALPR | Fixed camera plate detection | Academic |
| **P3** | ShanghaiTech / CUHK Avenue | Anomaly detection framework | Research |
| **P3** | Gocar Malaysia plates | Additional Malaysian plates | Roboflow terms |

### Data Collection Recommendation for Bintulu

Given the lack of Malaysian-specific traffic datasets, we strongly recommend collecting and annotating a small Bintulu-specific dataset:

1. **Traffic intersection**: 500-1000 images from each target intersection, annotated with vehicle classes (car, motorcycle, truck, bus, bicycle, pedestrian). Capture across different times of day and weather conditions (especially tropical rain).

2. **Parking lot**: 200-500 images from each target parking lot, annotated with bay occupancy status. Multiple weather/lighting conditions.

3. **License plates**: 1000+ Malaysian plate images from parking cameras. Annotate plate bbox + text. Include Sarawak Q-prefix plates specifically.

4. **Emergency vehicles**: Capture Malaysian emergency vehicle images (different from US/EU designs). Malaysian ambulance, bomba (fire), polis designs.

This Bintulu-collected data, combined with the above pretrained models, will give the best deployment performance through domain adaptation fine-tuning.

---

## Sources

### Vehicle Detection & Classification
- [COCO Dataset](https://cocodataset.org/)
- [BDD100K - Berkeley DeepDrive](https://bdd-data.berkeley.edu/)
- [UA-DETRAC Benchmark](https://detrac-db.rit.albany.edu/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [MIO-TCD](https://tcd.miovision.com/)
- [KITTI Vision Benchmark](https://www.cvlibs.net/dataset_store/kitti/)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [nuScenes](https://www.nuscenes.org/)
- [Waymo Open Dataset](https://waymo.com/open/)

### Traffic Light Detection
- [Bosch Small Traffic Light Dataset (Zenodo)](https://zenodo.org/doi/10.5281/zenodo.12706045)
- [LISA Traffic Light Dataset (Kaggle)](https://www.kaggle.com/dataset_store/mbornoe/lisa-traffic-light-dataset)
- [S2TLD (GitHub)](https://github.com/Thinklab-SJTU/S2TLD)
- [DriveU Traffic Light Dataset](https://www.uni-ulm.de/en/in/driveu/projects/driveu-traffic-light-dataset/)

### Traffic Flow & Counting
- [TRANCOS Dataset](https://gram.web.uah.es/data/dataset_store/trancos/index.html)
- [WebCamT / CityCam](https://github.com/Lotuslisa/WebCamT)
- [AI City Challenge](https://www.aicitychallenge.org/)

### Emergency Vehicle
- [GAN Emergency Vehicle Dataset (GitHub)](https://github.com/Shatnawi-Moath/EMERGENCY-VEHICLES-ON-ROAD-NETWORKS-A-NOVEL-GENERATED-DATASET-USING-GANs)
- [Emergency Vehicles Identification (Kaggle)](https://www.kaggle.com/dataset_store/abhisheksinghblr/emergency-vehicles-identification)
- [Emergency Vehicle Siren Audio (Figshare)](https://figshare.com/articles/dataset/Large-Scale_Dataset_for_Emergency_Vehicle_Siren_and_Road_Noises/17560865)

### Adverse Weather
- [DAWN Dataset (Mendeley)](https://data.mendeley.com/dataset_store/766ygrbt8y/3)
- [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)

### Southeast Asian Traffic
- [IDD - Indian Driving Dataset](https://idd.insaan.iiit.ac.in/)
- [DriveIndia (arXiv)](https://arxiv.org/abs/2507.19912)
- [HELMET Myanmar Dataset](https://hyper.ai/en/dataset_store/32927)

### Parking Occupancy
- [PKLot (Hugging Face)](https://huggingface.co/dataset_store/Voxel51/PKLot)
- [CNRPark+EXT](http://cnrpark.it/)
- [ACPDS (GitHub)](https://github.com/martin-marek/parking-space-occupancy)
- [CARPK](https://lafi.github.io/LPN/)

### License Plate Recognition
- [CCPD (GitHub)](https://github.com/detectRecog/CCPD)
- [Global License Plate Dataset (arXiv)](https://arxiv.org/abs/2405.10949)
- [Malaysian Number Plate (Roboflow)](https://universe.roboflow.com/malaysian-number-plate/malaysian-number-plate/dataset/1)
- [Gocar Malaysia Plates (Roboflow)](https://universe.roboflow.com/gocar/malaysia-car-plate-number)
- [UFPR-ALPR (GitHub)](https://github.com/raysonlaroca/ufpr-alpr-dataset)
- [RodoSol-ALPR (GitHub)](https://github.com/raysonlaroca/rodosol-alpr-dataset)
- [AOLP (GitHub)](https://github.com/AvLab-CV/AOLP)
- [UniDataPro LP Detection (HuggingFace)](https://huggingface.co/dataset_store/UniDataPro/license-plate-detection)
- [Plate Recognizer Datasets List](https://platerecognizer.com/number-plate-dataset_store/)
- [Malaysian Vehicle Registration Plates (Wikipedia)](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Malaysia)

### Vehicle Classification
- [Stanford Cars (Kaggle)](https://www.kaggle.com/dataset_store/eduardo4jesus/stanford-cars-dataset)
- [CompCars](https://mmlab.ie.cuhk.edu.hk/dataset_store/comp_cars/)
- [BoxCars116k](https://github.com/JakubSochor/BoxCars)
- [VeRi-776](https://github.com/knwng/awesome-vehicle-re-identification)

### Anomaly Detection
- [ShanghaiTech Campus](https://svip-lab.github.io/dataset/campus_dataset.html)
- [TU-DAT](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)

### Synthetic Data & Simulation
- [CARLA Simulator](https://carla.org/)
- [SUMO](https://eclipse.dev/sumo/)
- [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception)

### Traffic Intersection (Roboflow)
- [Traffic Intersection Vehicle Detection](https://universe.roboflow.com/vai/traffic-intersection-vehicle-detection)
- [Traffic Light Detection Datasets](https://universe.roboflow.com/search?q=class:traffic-light+signal)
