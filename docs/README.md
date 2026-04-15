# camera_edge Documentation

**[Product Roadmap](ROADMAP.md)** — Phase 1 & Phase 2 features, timeline, targets

## Quick Links

| I need to... | Go to |
|---|---|
| Read project requirements | [01_requirements/](01_requirements/) |
| Set up tools & environment | [02_project/tool-stack.md](02_project/tool-stack.md) |
| Read a specific feature spec | [03_platform/](03_platform/) |
| Label data for a model | [04_guides/01_labeling-guide.md](04_guides/01_labeling-guide.md) |
| Export model to ONNX | [04_guides/03_export-and-deploy.md](04_guides/03_export-and-deploy.md) |
| Add a new feature | Copy [03_platform/_TEMPLATE.md](03_platform/_TEMPLATE.md) |
| Find customer deliverables | [05_applications/](05_applications/) |

## Folder Map

| Folder | Purpose | Audience |
|--------|---------|----------|
| `01_requirements/` | Customer specs, platform requirements, performance targets | PM + leads |
| `02_project/` | Governance, tools, phases, development plans, technical approach | All team |
| `03_platform/` | One file per CV feature (cluster-prefixed) | Model owners |
| `04_guides/` | Model-agnostic lifecycle how-tos | All engineers |
| `05_applications/` | Customer deployments combining platform features | BD + deployment team |

## Platform Features

| ID | Feature | Cluster | Owner | Phase | Status |
|----|---------|---------|-------|-------|--------|
| a | [Fire Detection](03_platform/safety-fire_detection.md) | Safety | Tri | 1 | Training |
| b | [Helmet Detection](03_platform/ppe-helmet_detection.md) | PPE | Nguyen | 1 | Training |
| f | [Safety Shoes](03_platform/ppe-shoes_detection.md) | PPE | Nguyen | 1 | Training |
| g | [Fall Classification](03_platform/safety-fall_classification.md) | Safety | Tri | 1 | Training |
| g | [Fall Pose Estimation](03_platform/safety-fall_pose_estimation.md) | Safety | Tri | 1 | Training |
| h | [Poketenashi Violations](03_platform/safety-poketenashi.md) | Safety | Bang | 1 | Training |
| i | [Zone Intrusion](03_platform/access-zone_intrusion.md) | Access | Nguyen | 1 | Training |
| — | [Face Recognition](03_platform/access-face_recognition.md) | Access | TBD | 1 | Training |
| — | [Smart Parking](03_platform/traffic-smart_parking.md) | Traffic | TBD | — | Planned |
| — | [Traffic Signal](03_platform/traffic-signal_control.md) | Traffic | TBD | — | Planned |

## Applications

| Customer | Location | Platform Features | Docs |
|----------|----------|-------------------|------|
| Nitto Denko | Factory | PPE + Safety + Access | [nitto-denko/](05_applications/nitto-denko/) |
| Bintulu | Smart City | Traffic + Access | [bintulu/](05_applications/bintulu/) |

## Adding a New Feature

1. Copy `03_platform/_TEMPLATE.md` to `03_platform/{cluster}-{feature_name}.md`
2. Fill in all TODO sections
3. Add a row to the Platform Features table above
4. Create config at `configs/{feature_name}/05_data.yaml` + `06_training.yaml`
5. Create experiment workspace at `experiments/{feature_name}/`
