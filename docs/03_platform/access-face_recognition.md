# Access Control: Face Recognition
> ID: — | Owner: TBD | Phase: 1 | Status: training

## Customer Requirements

**Source:** Nitto Denko factory safety camera project requirements

**Explicit Requirements:**
- Detect faces within violation events captured by other detection models (PPE, phone, fall, zone intrusion)
- Identify enrolled individuals by matching detected faces against a pre-built face gallery
- Attribute safety violations to specific workers for follow-up action and accountability
- Support access control by distinguishing authorized personnel from unauthorized entrants
- Maintain an audit trail linking violation events to identified individuals

## Business Problem Statement

- Factory managers need to know *who* committed a safety violation so they can provide targeted training or disciplinary follow-up, rather than only knowing *that* a violation occurred
- Without identity attribution, violation reports lack accountability -- workers cannot be individually coached, and repeat offenders cannot be identified
- Access control requires distinguishing authorized workers from visitors or unauthorized entrants in restricted zones, but manual monitoring is impractical across multiple camera feeds
- Regulatory and internal audit requirements demand a record of which individuals were present during safety incidents
- Privacy regulations (local labor law, data protection) restrict how worker facial data can be collected, stored, and used, requiring careful handling of biometric information

## Technical Problem Statement

- **Violation attribution → Varying capture conditions:** Faces captured during violation events have inconsistent lighting (indoor/outdoor, backlight, low-light), angles (side profile, looking down), and partial occlusion (PPE, hard hats), making reliable matching difficult
- **Audit trail → Low-resolution and small faces:** Safety cameras capture wide scenes; faces in violation crops may be under 30px, degrading detection and recognition accuracy below usable thresholds
- **Accountability → Enrollment quality and gallery management:** Recognition accuracy depends heavily on the quality and diversity of enrollment photos; a single well-lit front-facing photo may not match the variety of conditions seen in the field, and managing a growing gallery of workers (additions, removals, updates) introduces operational complexity
- **Access control → Matching accuracy and threshold tuning:** False positives (wrong person flagged) damage trust and cause unjustified disciplinary action, while false negatives (missed violator) undermine the system's purpose; choosing the right similarity threshold involves a tradeoff that varies by deployment site
- **Privacy compliance → Secure storage and data handling:** Face embeddings are biometric data subject to legal protections; the system must store embeddings securely, support deletion requests, and avoid retaining raw face images beyond what is necessary

## Technical Solution Options

### Option 1: SCRFD-500M + MobileFaceNet-ArcFace (Recommended)

- **Approach:** Two-stage ONNX pipeline. SCRFD-500M (0.57M params) detects faces and outputs 5-point landmarks. MobileFaceNet-ArcFace (4M params) produces 512-d L2-normalized embeddings. Matching via cosine similarity against an enrolled gallery with configurable threshold. No training required -- uses pretrained weights.
- **Addresses:** Varying capture conditions (SCRFD handles multi-scale), enrollment quality (5-point landmark alignment normalizes pose), gallery management (simple file-based gallery), matching accuracy (ArcFace loss provides discriminative embeddings), privacy (embeddings only, no raw image retention)
- **Pros:** Extremely lightweight (4.57M total params), MIT license, ONNX Runtime (CPU-sufficient, no GPU needed), sub-10ms latency on edge chips, proven on AX650N/CV186AH, enrollment-based (no training pipeline needed)
- **Cons:** Requires affine alignment step (landmark-dependent), gallery management is file-based (no database), recognition accuracy degrades with heavy occlusion (masks, helmets covering face)

### Option 2: RetinaFace + ArcFace

- **Approach:** Alternative open-source two-stage pipeline. RetinaFace for face detection with landmark prediction, ArcFace (InsightFace implementation) for embedding extraction and matching.
- **Addresses:** Same core challenges as Option 1 -- face detection, landmark alignment, embedding extraction, gallery matching
- **Pros:** Well-documented, large community, multiple pretrained backbones available (ResNet, MobileNet), InsightFace ecosystem provides tooling
- **Cons:** Heavier than SCRFD (RetinaFace ResNet-50 is ~25M params), Apache 2.0 license (acceptable), larger inference footprint, more complex deployment on edge chips

### Option 3: DeepFace (wrapper)

- **Approach:** Use the DeepFace Python library as a unified wrapper around multiple face detection and recognition backends (VGG-Face, FaceNet, ArcFace, SFace, etc.). Handles detection, alignment, embedding, and matching in a single API call.
- **Addresses:** Reduces integration complexity by providing a single interface; allows backend swapping without code changes
- **Pros:** Easy prototyping, supports multiple backends, good for benchmarking, active maintenance
- **Cons:** Heavy dependencies (tensorflow/pillow/opencv/gdown), not optimized for edge deployment, abstracts away control needed for production tuning (landmark quality, alignment precision, threshold calibration), AGPL-3.0 license for some backends

**Decision:** Option 1 (SCRFD-500M + MobileFaceNet-ArcFace) -- minimal footprint (4.57M params), MIT license, proven edge performance, no training required, and already integrated in the camera_edge pipeline.

## Pipeline Overview

Identity recognition pipeline for matching violators (PPE, phone, fall) to enrolled worker identities. Uses SCRFD-500M face detector + MobileFaceNet-ArcFace embedder + cosine similarity gallery matching. No training required -- uses pretrained ONNX models.

## Pipeline

1. **Violation detected** → crop violation bounding box (expand ratio configurable)
2. **Face detection** → SCRFD-500M detects face + 5 landmarks within violation crop
3. **Alignment** → affine warp to 112×112 using ArcFace reference landmarks
4. **Embedding** → MobileFaceNet produces 512-d L2-normalized vector
5. **Gallery match** → cosine similarity against enrolled identities, threshold-based

## Models

| Component | Model | Params | License | Input | Output |
|-----------|-------|--------|---------|-------|--------|
| Face Detector | SCRFD-500M | 0.57M | MIT | any size | boxes + 5 landmarks |
| Face Embedder | MobileFaceNet-ArcFace | 4M | MIT | 112×112 | 512-d vector |

## Configuration

- **Key config:** `features/access-face_recognition/configs/face.yaml`
- **Gallery path:** configured in face.yaml
- **Similarity threshold:** 0.4 (default, configurable)
- **Violation class IDs:** mapped per use case

## Edge Deployment

- **Target chip:** AX650N / CV186AH
- **Runtime:** ONNX Runtime (CPU sufficient for face pipeline)
- **Latency:** < 10ms per face (detector + embedder)

### Model Card (Deliverables)

Each release produces a model card at `docs/model_cards/face_recognition.md`.

**Model artifacts:**

| Artifact | Format | Path |
|---|---|---|
| Face detector (SCRFD-500M) | `.onnx` | `pretrained/scrfd_500m.onnx` |
| Face embedder (MobileFaceNet) | `.onnx` | `pretrained/mobilefacenet_arcface.onnx` |
| Config | `.yaml` | `features/access-face_recognition/configs/face.yaml` |
| Face gallery | `.json` | `data/face_gallery/` |

## Limitations & Known Issues

- Requires enrolled gallery — no gallery = no identity matching
- Small/occluded faces (< 30px) may not be detected by SCRFD-500M
- Mask/PPE covering face degrades recognition accuracy

## Changelog

| Date | Version | Change |
|------|---------|--------|
