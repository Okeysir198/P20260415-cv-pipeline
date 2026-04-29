# Application: Nitto Denko Factory Safety

## Overview

Smart camera system for factory safety monitoring at Nitto Denko facilities. Combines multiple platform features for comprehensive workplace safety compliance.

## Platform Features Used

| Feature | Platform Doc | Purpose |
|---------|-------------|---------|
| Fire Detection | [safety-fire_detection](../../../03_platform/safety-fire_detection.md) | Fire/smoke early warning |
| Helmet Detection | [ppe-helmet_detection](../../../03_platform/ppe-helmet_detection.md) | Helmet + Nitto hat compliance |
| Safety Shoes | [ppe-shoes_detection](../../../03_platform/ppe-shoes_detection.md) | Safety footwear compliance |
| Fall Detection | [safety-fall-detection](../../../03_platform/safety-fall-detection.md) | Worker fall alerts |
| Poketenashi Violations (`safety-poketenashi_*` rule family) | [safety-poketenashi](../../../03_platform/safety-poketenashi.md) | Phone, hands-in-pockets, handrail, stair safety, point-and-call |
| Zone Intrusion | [access-zone_intrusion](../../../03_platform/access-zone_intrusion.md) | Restricted area monitoring |
| Face Recognition | [access-face_recognition](../../../03_platform/access-face_recognition.md) | Violator identity matching |

## Edge Hardware

- **Primary:** AX650N (18 INT8 TOPS)
- **Alternative:** CV186AH (7.2 INT8 TOPS)

## Deliverables

- [Technical Proposal](proposal.pptx)
