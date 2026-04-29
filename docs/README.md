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

| ID | Feature | Cluster | Owner | Phase | Status | Platform doc |
|----|---------|---------|-------|-------|--------|--------------|
| a | Fire Detection | Safety | Tri | 1 | Training | [safety-fire_detection.md](03_platform/safety-fire_detection.md) |
| b | Helmet Detection | PPE | Nguyen | 1 | Training | [ppe-helmet_detection.md](03_platform/ppe-helmet_detection.md) |
| f | Safety Shoes | PPE | Nguyen | 1 | Training | [ppe-shoes_detection.md](03_platform/ppe-shoes_detection.md) |
| — | Gloves Detection | PPE | TBD | 1 | Training | _doc TODO_ |
| g | Fall Detection | Safety | Tri | 1 | Training | [safety-fall-detection.md](03_platform/safety-fall-detection.md) |
| g | Fall Pose Estimation | Safety | Tri | 1 | Training | [safety-fall_pose_estimation.md](03_platform/safety-fall_pose_estimation.md) |
| h | Poketenashi Violations (umbrella) | Safety | Bang | 1 | Training | [safety-poketenashi.md](03_platform/safety-poketenashi.md) |
| h | Poketenashi: Phone Usage | Safety | Bang | 1 | Training | _see safety-poketenashi.md_ |
| h | Poketenashi: Hands in Pockets | Safety | Bang | 1 | Training | _see safety-poketenashi.md_ |
| h | Poketenashi: No Handrail | Safety | Bang | 1 | Training | _see safety-poketenashi.md_ |
| h | Poketenashi: Stair Diagonal | Safety | Bang | 1 | Training | _see safety-poketenashi.md_ |
| h | Poketenashi: Point-and-Call | Safety | Bang | 1 | Training | _see safety-poketenashi.md_ |
| i | Zone Intrusion | Access | Nguyen | 1 | Training | [access-zone_intrusion.md](03_platform/access-zone_intrusion.md) |
| — | Face Recognition | Access | TBD | 1 | Training | [access-face_recognition.md](03_platform/access-face_recognition.md) |
| — | Traffic Signal | Traffic | TBD | — | Planned | [traffic-signal_control.md](03_platform/traffic-signal_control.md) |
| — | Smart Parking | Traffic | TBD | — | Planned | _doc TODO_ |

Code lives under `../features/<category>-<name>/` — see
[`features/README.md`](../features/README.md) for the 11 production
feature folders.

## Applications

| Customer | Location | Platform Features | Docs |
|----------|----------|-------------------|------|
| Nitto Denko | Factory | PPE + Safety + Access | [nitto-denko/](05_applications/nitto-denko/) |
| Bintulu | Smart City | Traffic + Access | [bintulu/](05_applications/bintulu/) |

## Adding a New Feature

1. Copy `03_platform/_TEMPLATE.md` to `03_platform/<cluster>-<name>.md`
   and fill in all TODO sections.
2. Add a row to the Platform Features table above.
3. Scaffold the code workspace: `bash scripts/new_feature.sh <cluster>-<name>`
   — this creates `features/<cluster>-<name>/configs/{05_data,06_training,10_inference}.yaml`.
4. See the root [`README.md`](../README.md) for the end-to-end CLI
   (train / evaluate / export / infer) and the naming contract.
