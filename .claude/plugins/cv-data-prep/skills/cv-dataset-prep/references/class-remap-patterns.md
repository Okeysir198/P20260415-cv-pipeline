# Class Remap Patterns

Common canonical class taxonomies used in this repo and how raw-source class names map onto them. Use as a starting point; adjust for the specific feature.

## PPE — Helmet (Model B)

**Canonical**: `person`, `head_with_helmet`, `head_without_helmet`, `head_with_nitto_hat`

| Source class | → Canonical |
|---|---|
| `Hardhat`, `helmet`, `Helmet`, `safety-helmet` | `head_with_helmet` |
| `NO-Hardhat`, `no_helmet`, `head` | `head_without_helmet` |
| `Person`, `person`, `worker` | `person` |
| `Bump-cap` (from rf_bump_cap) | `head_with_nitto_hat` |
| `No BumpCap` (from rf_bump_cap) | `head_without_helmet` |

## PPE — Safety Shoes (Model F)

**Canonical**: `person`, `foot_with_safety_shoes`, `foot_without_safety_shoes`

| Source class | → Canonical |
|---|---|
| `Boots`, `shoes`, `safety_shoes`, `safety-shoe`, `Shoes` | `foot_with_safety_shoes` |
| `no_shoes`, `NO-Shoes`, `barefoot` | `foot_without_safety_shoes` |
| `Sandal`, `sandals`, `flip_flops`, `FlipFlops`, `Sneakers`, `sneaker`, `Dress_Shoes`, `Heels`, `Clogs`, `Flats`, `Mules`, `slippers` | `foot_without_safety_shoes` |
| `Shoe` (ambiguous, from rf_footwear_sandal_shoe) | **DROP** (needs visual review) |

## Safety — Fire/Smoke (Model A)

**Canonical**: `fire`, `smoke`

| Source class | → Canonical |
|---|---|
| `fire`, `Fire`, `flame`, `0` (in roboflow indoor-fire-smoke) | `fire` |
| `smoke`, `Smoke`, `1` (in roboflow indoor-fire-smoke) | `smoke` |
| `chemical hazard`, `no helmet`, `water leak` (industrial_hazards extras) | **DROP** |

## Safety — Fall (Model G)

**Canonical**: `person`, `fallen_person`

| Source class | → Canonical |
|---|---|
| `person-nofall`, `standing` | `person` |
| `person-fall`, `Fall-Detected`, `laying` | `fallen_person` |

## Poketenashi — Phone Usage (Model H sub-model)

**Canonical**: `person`, `phone_usage`

| Source class | → Canonical |
|---|---|
| `Phone`, `phone`, `Using Phone`, `phone usage`, `using phone`, `mobile`, `phone_calling`, `phone_playing`, `c0-Texting`, `c1-Calling` | `phone_usage` |
| `Person`, `face` (from fpi_det — face bbox is where the person is) | `person` |
| `c2-Drinking`, `cell phone` (object-only) | **DROP** |

## Harness (Phase 2 Model K)

**Canonical**: `person`, `harness_worn`, `harness_not_worn`, `harness_anchored`, `harness_not_anchored`

| Source class | → Canonical |
|---|---|
| `safety harness`, `safety_harness` | `harness_worn` |
| `no_Safety_harness`, `harness_not_worn` | `harness_not_worn` |
| `anchored` (from techling/safety_harness_v2) | `harness_anchored` |
| `non_anchored` (from techling/safety_harness_v2) | `harness_not_anchored` |
| `worker` | `person` |

## Zone Intrusion (Model I — mostly uses pretrained person detector)

Usually **no custom training** needed — uses a pretrained COCO person detector plus zone polygon logic. Only build a dedicated training set if the pretrained model underperforms in deployment.

If training: canonical is `person`, `intrusion` (latter only in datasets that pre-label zone violations).

## Rules when designing a new class-map

1. **Keep canonical class names stable across versions** — they become YOLO class IDs. Reordering later breaks every existing checkpoint.
2. **Map many source classes to one canonical** is fine. The reverse (splitting one source class into multiple canonicals) is not possible without per-instance rules — drop the source class.
3. **If a canonical class has zero contributing source classes** after all mapping, drop the class entirely or mark it for `site_collected/` population.
4. **Case-sensitive** — `"Helmet"` ≠ `"helmet"`. Check exact strings with `inspect_source.sh`.
5. **Spaces/hyphens matter** — `"NO-Hardhat"` vs `"NO Hardhat"` vs `"no_hardhat"` are all different.
