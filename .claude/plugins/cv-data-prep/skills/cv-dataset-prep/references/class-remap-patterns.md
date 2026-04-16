# Class Remap Patterns

How to design canonical class taxonomies and map raw source class names onto them.
Applies to any detection feature — not specific to any use case or project phase.

---

## Principles for class taxonomy design

### 1. Always include negatives

Detection models need to learn what the object is NOT, not just what it is. Without negative
classes, the model has no way to penalise false positives.

| Task type | Negative class pattern |
|---|---|
| PPE compliance | `no_helmet`, `no_glove`, `barefoot` alongside the positive class |
| Object present/absent | `<object>_present` / `<object>_absent` |
| Action detection | Label the negative action (standing, walking) when the alert action is specific |
| Presence-only alert | Add `person` so the model sees the whole scene, not just the alert object |

If no public source provides negatives, the model will overfit to positive patterns and
generate false positives in production. Flag this at source-selection time (step 3).

### 2. Keep the taxonomy small and stable

- **2–5 classes** is typical for a deployment model. Larger taxonomies dilute data per class.
- **Class order is frozen** once training starts — it defines YOLO class IDs. Reordering
  breaks all existing checkpoints. Settle the list before step 7 (full merge).
- **Don't split a class you can't tell apart** from the image alone. If distinguishing
  `helmet_red` from `helmet_yellow` requires context a camera can't provide, use one class.

### 3. Many-to-one mapping is fine; one-to-many is not

Multiple source classes can map to one canonical class:
```yaml
class_map:
  "Hardhat": "head_with_helmet"
  "safety-helmet": "head_with_helmet"
  "helmet": "head_with_helmet"
```

One source class cannot split into multiple canonical classes without per-instance rules
(which p00 does not support). If a source class is genuinely ambiguous (e.g. `"Shoe"` could
be safety or casual), **drop it** and add a clearer source.

### 4. Dropping source classes

Classes not in `class_map` are silently dropped by p00. Always make drops explicit:
- List them in `dropped_classes:` on the source entry in the YAML.
- This makes the generated `DATASET_REPORT.md` auditable.
- Common drop reasons:
  - Off-topic class in a multi-class dataset (e.g. `vest`, `goggles` in a helmet-only model)
  - Ambiguous class name (e.g. `"Shoe"` without safety/casual distinction)
  - Opaque auto-generated name (e.g. `"mobile_dataset - v9 2025-04-05"`)

### 5. Naming conventions

Use `snake_case`. Prefix with body part or context when it prevents future ambiguity:
- `head_with_helmet` not `helmet` (leaves room for `head_without_helmet`)
- `foot_with_safety_shoes` not `shoes`
- `fallen_person` not `fall` (a verb can confuse)

---

## Common patterns by task category

### Compliance (wear / no-wear)

```
person
<item>_compliant      # wearing the required PPE
<item>_non_compliant  # not wearing it
```

Examples: `head_with_helmet` / `head_without_helmet`, `foot_with_safety_shoes` / `foot_without_safety_shoes`

Typical raw-source class name variants for the compliant side:
`Hardhat`, `helmet`, `Helmet`, `safety-helmet`, `Boots`, `safety_shoes`, `glove`, `Gloves`

Typical raw-source variants for the non-compliant side:
`NO-Hardhat`, `no_helmet`, `head`, `no_shoes`, `barefoot`, `no_glove`

### Hazard / safety event detection

```
<event_positive>   # the thing you want to alert on
```
Often paired with `person` if tracking is required.

Examples: `fire`, `smoke`, `fallen_person`, `spill`

Typical multi-class sources have extra classes to drop:
`chemical`, `water_leak`, `no_helmet` (from a fire dataset — drop them unless also building a helmet model)

### Action / behaviour detection

```
person
<action>        # the specific action triggering an alert
```

Typical raw → canonical:
`Using Phone` / `phone` / `mobile` / `using phone` / `c0-Texting` / `c1-Calling` → `phone_usage`

Drop off-topic actions: `c2-Drinking` when you only care about phone usage.

### Object presence / zone intrusion

Usually handled by a pretrained COCO person detector plus zone polygon logic — no custom
training needed unless the pretrained model underperforms in the deployment environment.

If custom training: `person` / `vehicle` are usually sufficient canonicals.

---

## This repo's existing feature class maps (reference, not prescription)

Kept here for convenience. Copy and adapt for new features; do not treat as fixed.

### PPE — Helmet detection

**Canonical**: `person`, `head_with_helmet`, `head_without_helmet`, `head_with_nitto_hat`

| Source class | → Canonical |
|---|---|
| `Hardhat`, `helmet`, `Helmet`, `safety-helmet` | `head_with_helmet` |
| `NO-Hardhat`, `no_helmet`, `head` | `head_without_helmet` |
| `Person`, `person`, `worker` | `person` |
| `Bump-cap` | `head_with_nitto_hat` |
| `No BumpCap` | `head_without_helmet` |

### PPE — Safety shoes

**Canonical**: `person`, `foot_with_safety_shoes`, `foot_without_safety_shoes`

| Source class | → Canonical |
|---|---|
| `Boots`, `shoes`, `safety_shoes`, `Shoes` | `foot_with_safety_shoes` |
| `no_shoes`, `NO-Shoes`, `barefoot` | `foot_without_safety_shoes` |
| `Sandal`, `FlipFlops`, `Sneakers`, `sneaker`, `Dress_Shoes`, `Heels`, `Clogs`, `Flats`, `Mules`, `slippers` | `foot_without_safety_shoes` |
| `Shoe` (ambiguous, from rf_footwear_sandal_shoe) | **DROP** |

### Safety — Fire/smoke

**Canonical**: `fire`, `smoke`

| Source class | → Canonical |
|---|---|
| `fire`, `Fire`, `flame`, `0` (numeric ID) | `fire` |
| `smoke`, `Smoke`, `1` (numeric ID) | `smoke` |
| `chemical hazard`, `no helmet`, `water leak` | **DROP** |

### Safety — Fall detection

**Canonical**: `person`, `fallen_person`

| Source class | → Canonical |
|---|---|
| `person-nofall`, `standing` | `person` |
| `person-fall`, `Fall-Detected`, `laying` | `fallen_person` |

### Action — Phone usage

**Canonical**: `person`, `phone_usage`

| Source class | → Canonical |
|---|---|
| `Phone`, `phone`, `Using Phone`, `phone usage`, `using phone`, `mobile`, `c0-Texting`, `c1-Calling` | `phone_usage` |
| `Person`, `face` | `person` |
| `c2-Drinking`, `cell phone` (object-only) | **DROP** |

### PPE — Safety harness

**Canonical**: `person`, `harness_worn`, `harness_not_worn`, `harness_anchored`, `harness_not_anchored`

| Source class | → Canonical |
|---|---|
| `safety harness`, `safety_harness` | `harness_worn` |
| `no_Safety_harness` | `harness_not_worn` |
| `anchored` | `harness_anchored` |
| `non_anchored` | `harness_not_anchored` |
| `worker` | `person` |

---

## Rules for any new class-map

1. **Class order is frozen after the first training run** — additions must append, never reorder.
2. **Case-sensitive** — `"Helmet"` ≠ `"helmet"`. Verify exact strings with `inspect_source.sh`.
3. **Spaces and hyphens matter** — `"NO-Hardhat"` vs `"NO Hardhat"` are different strings.
4. **Zero-coverage class** — if a canonical class has no contributing sources after mapping,
   drop it or mark it for `site_collected/` population; don't train on an empty class.
5. **Numeric-ID YOLO** — when a source's `data.yaml` lists `names: ['0', '1']`, add
   `source_classes:` in the YAML with the real names in the correct order.
