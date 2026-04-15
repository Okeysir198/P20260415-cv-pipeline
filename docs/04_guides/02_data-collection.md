# CCTV Data Collection Strategy

## A. Raw Retention Plan

> **Nitto Denko context:** This plan analyzes risk based on FPC (Flexible Printed Circuits) and adhesive tape manufacturing processes.
> *   **Special areas:** Clean Room (Class 1k/10k), Yellow Room (Lithography), Solvent Storage.
> *   **Specific risks:** Static discharge, chemical fumes, high-speed winding machines.

### 1. Classification & Retention Duration
*   **Group H (High-risk):** Staircases, entrance gates, fences, chemical/flammable storage. **Retention: 7 days.**
*   **Group N (Normal):** Production lines, internal corridors. **Retention: 3 days.**

### 2. Resource Formula & Example
$\text{Total Camera-Days} = (H \times 7) + (N \times 3)$

**Nitto:** H=15, N=35 → $(15 \times 7) + (35 \times 3) = 210$ Camera-Days (~40% savings vs full 350).

### 3. Hardware Scope
Camera SKUs: Bullet 5MP (`DWC-MB75Wi4TW`, `DWC-MV75Wi28TW`), Dome 5MP (`DWC-MV72Wi28TW`), Dome 2.1MP (`DWC-MV82DiVT`, `DWC-PVF5Di1TW`), Fisheye 360 (`DWC-MB62DiVTW` -- raw stream, no de-warp), Bullet 2.1MP (`DWC-MB45WiATW`). Each model needs at least **50GB** raw data for de-noise/SR training.

## B. Minimal Sampling Strategy

### 1. Sampling Quantities
**60 min/shift** x 3 shifts = **3 hours/camera/day**. Total: $\text{Camera-Days} \times 3$ hours.

### 2. Why Sufficient
1 hour/shift captures environmental variations (lighting, shadows, crowd density). 87.5% cost savings vs 24h. 3 time windows (Morning/Afternoon/Night) cover edge cases.

### 3. Supplementary / Open Source Data
Sources: Kaggle, Roboflow, COCO. Must not exceed **20-30%** of total volume to avoid domain shift from Nitto factory environment.

## C. Staged Data Requirements

**Units:** Min Qty = MVP/PoC minimum. Prod Qty = production target (>95% accuracy). 1 unit = 1 sequence (10-20s).

**Diversity:** Viewpoint (top-down priority + frontal/side/back), pose (standing/walking/running/bending/squatting), lighting variations, occlusion.

**3-Layer Structure:** (1) Clear cases -- unambiguous, well-lit. (2) Operational noise -- distant views, backlighting, 50% occlusion, IR blur. (3) Confusion cases -- hard negatives easily confused with targets.

### 1. Fire / Smoke Detection
*   **Objective:** Early detection of chemical smoke, solvent fires, electrical shorts, trash fires.
*   **Nitto VN specifics:** High humidity/condensation, insect fogging, Toluene/MEK solvents, drying ovens.

| Type | Description | Min Qty | Prod Qty | Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Positive (Clear)** | **Black smoke (oil/plastic):** Burning tape rolls, wiring, plastic — thick, dark brown, rising high. | 25 | **200** | Aug: H265+Noise+MotionBlur; VLG: [P-A-SMOKE-BLACK](#catalog-A) |
| **Layer 1: Positive (Clear)** | **White smoke (chemical):** Acid/solvent reaction at plating/cleaning tanks — opaque white, low-spreading. | 25 | **200** | Aug: H265+Noise; VLG: [P-A-SMOKE-WHITE](#catalog-A) |
| **Layer 1: Positive (Clear)** | **Electrical sparking:** Short circuit at control cabinet, continuous sparks — fast flicker, orange/blue glow. | 25 | **150** | Aug: Flicker+H265; VLG: [P-A-SPARK](#catalog-A) |
| **Layer 1: Positive (Clear)** | **Trash bin fire:** Smoke rising from trash bin due to cigarette butts — light gray, spreading, intermittent. | 25 | **150** | Aug: H265+Noise; VLG: [P-A-SMOKE-TRASH](#catalog-A) |
| **Layer 2: Positive (Noise)** | **Occluded smoke:** Smoke rising behind storage racks or large coating machines (only partially visible). | 25 | **200** | Aug: Cutout/Mosaic+H265 |
| **Layer 2: Positive (Noise)** | **Small flame:** Faint flickering flame on floor from solvent leak (hard to see via ceiling camera). | 25 | **150** | Aug: Lowlight+Noise; VLG: [P-A-FLAME-SMALL](#catalog-A) |
| **Layer 3: Confusion** | **Insect fogging:** Dense smoke from periodic pest control spraying (very similar to fire smoke). | 40 | **300** | VLG: [P-A-FOGGING](#catalog-A) |
| **Layer 3: Confusion** | **Drying steam:** High-pressure steam escaping from adhesive drying ovens/boilers — pure white, high velocity. | 30 | **250** | Aug: MotionBlur+ExposureShift; VLG: [P-A-STEAM-PLUME](#catalog-A) |
| **Layer 3: Confusion** | **Fog / condensation:** Lens fogging due to high humidity in early morning (Northern Vietnam). | 30 | **250** | Aug: Haze+Blur+LowContrast |
| **Layer 3: Confusion (Zone 14/16)** | **Kitchen steam:** White vapor rising from industrial rice cookers/kitchen area. | 20 | **150** | VLG: [P-A-STEAM-KITCHEN](#catalog-A) |
| **Layer 3: Confusion (Zone 4)** | **Forklift rotating beacon:** Yellow/orange rotating light reflecting off glossy epoxy floor (resembles fire). | 30 | **200** | Aug: Glare+BrightnessShift |
| **Layer 3: Confusion (Zone 8)** | **Industrial dust:** Dust collector discharge creating momentary dust cloud. | 20 | **150** | VLG: [P-A-DUST-CLOUD](#catalog-A) |
| **Layer 3: Confusion** | **Machining dust:** Dust kicked up from grinding/slitting machines (slitter dust). | 30 | **200** | VLG: [P-A-DUST-LIGHT](#catalog-A) |
| **Layer 3: Confusion** | **Welding:** Maintenance metal cutting, gray smoke + sparks (with safety barriers). | 40 | **300** | Aug: H265+Noise+Flicker |
| **Layer 3: Confusion** | **Warning lights:** Orange/red rotating beacons on forklifts or machinery (resembles fire glow). | 30 | **200** | Aug: Flicker+Glare |
| **Layer 3: Confusion** | **Reflective vests:** Group of workers in orange/yellow vests moving quickly (easily mistaken for fire). | 30 | **150** | Aug: MotionBlur+ExposureShift |
| **Layer 3: Confusion** | **Compressed air discharge:** Sudden valve release creating a strong white air plume. | 20 | **150** | VLG: [P-A-AIR-VENT](#catalog-A) |

### 2. Fall Detection
*   **Objective:** Detect people falling in walkways, staircases, chemical areas.
*   **Nitto VN specifics:** Glossy epoxy floors, 5S culture, night shift operations.

| Type | Description | Min Qty | Prod Qty | Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Positive (Clear)** | **Slip fall:** Slipping on wet floor (cleaning water/solvent) — prioritize top-down view, ~30% occluded by pallets. | 25 | **200** | Aug: MotionBlur+H265 |
| **Layer 1: Positive (Clear)** | **Trip fall (cable/threshold):** Tripping on cables/door thresholds — side/back view, ~60% occluded by bins. | 25 | **200** | Aug: Occlusion60+MotionBlur |
| **Layer 1: Positive (Clear)** | **Trip fall (pallet/rack):** Tripping while carrying items, kicking low pallets/racks — top-down, moving obstruction. | 25 | **180** | Aug: Occlusion30+H265 |
| **Layer 1: Positive (Clear)** | **Fainting:** Sudden collapse — top-down, ~30% crowd occlusion during peak hours. | 25 | **200** | Aug: CrowdOcclusion30+Noise |
| **Layer 1: Positive (Clear)** | **Ladder fall:** Falling while climbing aluminum ladder to retrieve items (height < 2m) — frontal/side. | 25 | **150** | Aug: Lowlight+Noise |
| **Layer 2: Positive (Noise)** | **Occluded fall:** Falling into gap between two machines, only legs visible flailing. | 25 | **200** | Aug: Cutout+H265 |
| **Layer 3: Confusion** | **5S cleaning:** Bending/crawling on floor to pick up debris, wiping small stains. | 40 | **300** | Aug: PoseVar+Occlusion |
| **Layer 3: Confusion** | **Under-machine repair:** Technician lying on back/crawling under machine for maintenance. | 40 | **250** | Aug: Lowlight+H265 |
| **Layer 3: Confusion** | **Tying shoelaces:** Kneeling on one leg or sitting down to tie shoes. | 40 | **250** | Aug: MotionBlur+BrightnessShift |
| **Layer 3: Confusion** | **Squatting:** Squatting while waiting or chatting. | 40 | **300** | Aug: PoseVar+H265 |
| **Layer 3: Confusion** | **Lunch nap:** Lying on cardboard/mat to nap in warehouse during break. | 30 | **200** | Aug: ExposureShift+H265 |
| **Layer 3: Confusion** | **Retrieving items from low shelf:** Lying prone/bending down to reach items under low racks. | 20 | **150** | Aug: Occlusion60+H265 |
| **Layer 3: Confusion** | **Mid-shift exercises:** Bending/stretching exercises during break. | 30 | **150** | Aug: MotionBlur+Noise |

### 3. Intrusion / Loitering / Vehicle
*   **Objective:** In-plant traffic safety, restricted area security.
*   **Nitto VN specifics:** Forklifts, hand pallet trucks, AGV robots.

| Type | Description | Min Qty | Prod Qty | Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Positive (Clear)** | **Vehicle in pedestrian lane:** Forklift/electric cart entering green walkway. | 25 | **200** | Aug: MotionBlur+Glare; VLG: [P-I-FORKLIFT-IN-WALKWAY](#catalog-I) |
| **Layer 1: Positive (Clear)** | **Pedestrian in vehicle lane:** Person walking in forklift lane (no markings). | 25 | **200** | Aug: MotionBlur+H265; VLG: [P-I-PERSON-IN-LANE](#catalog-I) |
| **Layer 1: Positive (Clear)** | **Wrong way:** Walking/driving against directional arrows (both people and vehicles). | 25 | **200** | Aug: H265+Noise; VLG: [P-I-WRONG-WAY](#catalog-I) |
| **Layer 1: Positive (Clear)** | **Tailgating:** Following closely behind someone through access card/turnstile door. | 30 | **200** | VLG: [P-I-TAILGATE](#catalog-I) |
| **Layer 1: Positive (Clear)** | **Small vehicle intrusion:** Bicycle/small e-bike entering restricted area or wrong lane. | 25 | **200** | Aug: MotionBlur+H265; VLG: [P-I-BIKE-IN-ZONE](#catalog-I) |
| **Layer 1: Positive (Clear)** | **AGV wrong zone:** AGV/robot entering unauthorized zone or crossing forbidden line. | 25 | **180** | VLG: [P-I-AGV-IN-ZONE](#catalog-I) |
| **Layer 1: Positive (Clear)** | **Unauthorized loitering:** Person standing/pacing near chemical storage or restricted area >30s (especially night shift, IR mode). | 30 | **220** | Aug: IR+Noise; VLG: [P-I-LOITER-DOOR](#catalog-I) |
| **Layer 3: Confusion** | **Intersection crossing:** Vehicles/people crossing at marked intersections (legitimate). | 40 | **300** | Aug: MotionBlur+H265 |
| **Layer 3: Confusion** | **Hand truck:** Manual hand truck entering pedestrian walkway (permitted). | 40 | **250** | VLG: [P-I-HANDTRUCK](#catalog-I) |
| **Layer 3: Confusion** | **Cleaning cart:** Janitor pushing cleaning cart into any area. | 30 | **200** | VLG: [P-I-CLEANER-CART](#catalog-I) |
| **Layer 3: Confusion** | **Contractor:** Wearing differently colored vest, moving around for repairs (permitted). | 40 | **300** | VLG: [P-I-CONTRACTOR-VEST](#catalog-I) |
| **Layer 3: Confusion (Zone 2)** | **PVC strip curtains:** Warehouse/workshop door curtains fluttering from wind or air curtain fans. | 40 | **300** | Aug: MotionBlur+Haze |
| **Layer 3: Confusion** | **Shadows / reflections:** Forklift shadows moving across glossy floor. | 30 | **200** | Aug: Glare+ExposureShift |
| **Layer 3: Confusion** | **Security patrol:** Security guard performing scheduled rounds (with badge/uniform). | 50 | **400** | VLG: [P-I-SECURITY-PATROL](#catalog-I) |
| **Layer 3: Confusion (Night)** | **Insects / spider webs (IR Mode):** Spiders or insects near IR lens causing white bloom artifacts. | 100 | **500** | Aug: IR-Noise+Bloom |
| **Layer 3: Confusion (Outdoor)** | **Wildlife:** Stray dogs/cats crossing fence/waste areas. | 30 | **150** | VLG: [P-I-ANIMAL-OUTDOOR](#catalog-I) |

### 4. PPE Detection (Helmet / Shoes / Attire)
*   **Objective:** Uniform and personal protective equipment compliance.
*   **Nitto VN specifics:** Cleanroom attire, bunny hoods, face masks.

| Type | Description | Min Qty | Prod Qty | Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Positive (Clear)** | **Worker helmet (yellow) - correct:** Properly worn with chin strap fastened — top-down/frontal/side/back views. | 25 | **200** | Aug: H265+Noise+MotionBlur; VLG: [P-B-HELMET-YELLOW](#catalog-B) |
| **Layer 1: Positive (Clear)** | **Engineer helmet (white) - correct:** Properly worn with chin strap fastened — top-down/side view. | 25 | **200** | Aug: H265+Noise; VLG: [P-B-HELMET-WHITE](#catalog-B) |
| **Layer 1: Positive (Clear)** | **Safety helmet (blue/red) - correct:** Properly worn — include lowlight/IR to avoid color bias. | 25 | **180** | Aug: IR/Lowlight+Noise; VLG: [P-B-HELMET-COLOR](#catalog-B) |
| **Layer 1: Positive (Clear)** | **No helmet:** Bare head in mandatory helmet zone — black hair/bald, top-down view. | 30 | **250** | VLG: [P-B-NO-HELMET](#catalog-B) |
| **Layer 1: Positive (Clear)** | **Helmet worn incorrectly:** Strap unfastened/dangling, worn backwards, tilted — top-down + 30/60% occlusion. | 30 | **250** | Aug: Occlusion30/60+H265; VLG: [P-B-HELMET-WEAR-WRONG](#catalog-B) |
| **Layer 1: Positive (Clear)** | **Safety boots - correct:** Steel-toe high-top boots — top-down view showing toe/sole. | 25 | **200** | Aug: H265+Noise; VLG: [P-F-BOOT-OK](#catalog-F) |
| **Layer 1: Positive (Clear)** | **Safety shoes (lace) - correct:** Low-top lace-up shoes — frontal/side view. | 25 | **200** | Aug: MotionBlur+H265; VLG: [P-F-SHOE-LACE-OK](#catalog-F) |
| **Layer 1: Positive (Clear)** | **Safety shoes (velcro) - correct:** Velcro-strap shoes — back/side view. | 25 | **180** | Aug: H265+Noise; VLG: [P-F-SHOE-VELCRO-OK](#catalog-F) |
| **Layer 1: Positive (Clear)** | **No safety shoes:** Wearing sandals/slippers or barefoot in mandatory shoe zone. | 30 | **250** | VLG: [P-F-NO-SAFETY-SHOE](#catalog-F) |
| **Layer 1: Positive (Clear)** | **Heel stepping:** Wearing safety shoes as slip-ons with heel stepped down (open heel). | 30 | **250** | Aug: MotionBlur+H265; VLG: [P-F-HEEL-STEP](#catalog-F) |
| **Layer 1: Positive (Clear)** | **Hair exposed (Cleanroom):** Hair strands visible outside bunny hood edges. | 30 | **250** | VLG: [P-B-CLEANROOM-HAIR-OUT](#catalog-B) |
| **Layer 2: Positive (Noise)** | **Full-cover hood:** Face fully covered, only eyes visible (face detection difficult). | 50 | **300** | Aug: Lowlight+H265+Noise |
| **Layer 3: Confusion** | **Helmet misplaced:** Helmet left on table/cabinet top (AI mistakes it for a person). | 40 | **250** | VLG: [P-B-HELMET-ON-TABLE](#catalog-B) |
| **Layer 3: Confusion** | **Pallet bag wrap:** White plastic bag covering pallet load (resembles cleanroom-suited person). | 30 | **200** | VLG: [P-B-PALLET-BAG](#catalog-B) |
| **Layer 3: Confusion** | **Sweat towel:** Towel wrapped around neck/head (resembles violation / unusual headwear). | 30 | **150** | VLG: [P-B-TOWEL-HEAD](#catalog-B) |
| **Layer 3: Confusion** | **Shoe covers:** Visitors wearing normal shoes with disposable plastic covers (legitimate). | 30 | **200** | VLG: [P-F-SHOE-COVER](#catalog-F) |
| **Layer 3: Confusion** | **Hairnet:** Female workers wearing thin hairnet (legitimate for packaging). | 40 | **200** | VLG: [P-B-HAIRNET](#catalog-B) |
| **Layer 3: Confusion** | **Black hair / bald head:** Person without helmet but with black hair/bald head (easily mistaken for helmet). | 30 | **200** | VLG: [P-B-BAREHEAD-HAIRVAR](#catalog-B) |
| **Layer 3: Confusion** | **Casual headwear:** Baseball cap, cloth hat, hoodie covering head. | 30 | **200** | VLG: [P-B-CAP-HOODIE](#catalog-B) |
| **Layer 3: Confusion** | **Carrying items over head:** Carrying tall boxes that occlude the head area. | 30 | **200** | Aug: Occlusion60+MotionBlur |

### 5. Safety Rules (Behavioral Safety)
*   **Objective:** Detect unsafe behaviors (phone use, hands in pockets, handrail non-use, jaywalking, lack of observation).
*   **Nitto VN specifics:** Work Zalo messaging habits, walking behavior patterns.

| Type | Description | Min Qty | Prod Qty | Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Positive (Clear)** | **Phone use:** Texting, browsing, holding phone to ear while walking. | 30 | **250** | Aug: MotionBlur+H265; VLG: [P-H-PHONE-IN-HAND](#catalog-H) |
| **Layer 1: Positive (Clear)** | **Hands in pockets:** Both hands in trouser pockets while walking (balance risk). | 30 | **250** | Aug: MotionBlur+H265; VLG: [P-H-HANDS-IN-POCKETS](#catalog-H) |
| **Layer 1: Positive (Clear)** | **Jaywalking:** Crossing lanes diagonally, not at right angles. | 30 | **250** | Aug: MotionBlur+H265; VLG: [P-H-JAYWALK](#catalog-H) |
| **Layer 1: Positive (Clear)** | **Not holding handrail:** Going up/down stairs with arms hanging or carrying items. | 30 | **250** | Aug: Lowlight+Noise; VLG: [P-H-NO-HANDRAIL](#catalog-H) |
| **Layer 1: Positive (Clear)** | **Lack of observation:** Crossing/turning without looking left/right. | 30 | **250** | Aug: MotionBlur+H265; VLG: [P-H-NO-LOOK](#catalog-H) |
| **Layer 3: Confusion** | **Handheld work devices:** Holding PDA, scanner, radio, tablet (legitimate). | 40 | **300** | VLG: [P-H-PDA-SCANNER](#catalog-H) |
| **Layer 3: Confusion** | **Holding small objects:** Wallet, notebook, pen, water bottle (easily mistaken for phone). | 30 | **200** | VLG: [P-H-SMALL-OBJECT](#catalog-H) |
| **Layer 3: Confusion** | **Cold / itch:** Briefly putting hand in pocket to retrieve item, scratching leg (short duration). | 40 | **300** | VLG: [P-H-POCKET-BRIEF](#catalog-H) |
| **Layer 3: Confusion (Zone 1)** | **Sticky mat cleaning:** Bending/stomping feet to peel sticky dust mat layer at cleanroom entrance. | 30 | **250** | VLG: [P-H-STICKY-MAT](#catalog-H) |
| **Layer 3: Confusion** | **Heavy lifting:** Carrying large box with both hands (not holding handrail — legitimate due to carrying). | 30 | **200** | Aug: Occlusion60+MotionBlur |
| **Layer 3: Confusion** | **Adjusting ear protection:** Adjusting earmuffs or safety glasses. | 30 | **150** | VLG: [P-H-ADJUST-EARMUFF](#catalog-H) |
| **Layer 3: Confusion** | **Talking to colleague:** Turning head to the side while speaking (resembles observation check). | 30 | **150** | VLG: [P-H-TALK-TURN](#catalog-H) |

## D. Shift Coverage Rules

*   **S1 (Day - 06:00-14:00):** **30%**. (Good lighting, high foot traffic).
*   **S2 (Lowlight - 14:00-22:00):** **30%**. (Low light, day-night transition).
*   **S3 (Night/IR - 22:00-06:00):** **40%**. (IR black-and-white, grain noise).

**Mandatory:** Intrusion: 70% from S3+S2. Fall: at least 20% from S2.

## E. Stopping Criteria

1.  **Raw volume:** Min **300 hours** of clean/classified camera footage.
2.  **Confusion volume:** Each use case has at least **150 confusion samples** (per tables in Section C).
3.  **Night ratio:** Total S3 duration $\ge$ **35%**.
4.  **Stair traversals:** At least **500 recorded traversals** (for handrail non-use false positive filtering).
5.  **Night intrusion sequences:** At least **100 sequences** (including flashlight and crawling scenarios).

## F. Camera-Zone Matrix

| Zone | Camera Type | Focus Use Cases | Staging Notes |
| :--- | :--- | :--- | :--- |
| **Clean Room (1F)** | Dome 5MP | **F (Shoes), I (Intrusion)** | Simulate wrong-way air shower entry, no cleanroom shoes. |
| **Material Handling** | Dome 2.1MP | **G (Fall), H (Rules)** | Simulate falling while carrying items, tripping on pallets. |
| **Warehouse (1F/3F)** | Dome/Bullet | **A (Fire), G (Fall), I (Intrusion)** | Simulated smoke behind storage racks. |
| **Hazardous Mat. W/H** | Bullet Ex-proof | **A (Fire), B (PPE), I (Intrusion)** | **High Priority:** Simulate white chemical smoke following safety protocols. |
| **Machine/Dust Room** | Dome 2.1MP | **A (Fire), G (Fall)** | Simulate control cabinet short circuit, maintenance fall. |
| **Corridor/Stairs** | Dome Fixed | **G (Fall), H (Handrail/Phone)** | Walking while texting, not holding handrail, stair falls. |
| **Fence/Outdoor** | Bullet 5MP | **I (Intrusion), A (Fire)** | Fence climbing, trash burning near perimeter wall. |
| **Smoking Area** | Bullet Vari | **A (Fire)** | Simulate cigarette butt causing trash bin smoke. |
| **Generator/Pump** | Bullet 5MP | **A (Fire), I (Intrusion)** | Unauthorized intruder sabotage, fuel leak simulation (using water). |

## G. Legal & Data Handover

### 1. Privacy Compliance
*   **Notification:** Post signage "AI testing video recording area" at Zone 14 (Kitchen), Zone 10 (Smoking), and Zone 1 (Locker).
*   **Data Pledge:** Data is used exclusively for AI safety training; faces will be blurred/deleted if non-safety-related incidents are inadvertently captured (e.g., personal relationships).
*   **Non-Safety Events:** Behaviors such as "lunch napping", "scratching", "adjusting clothing" classified as Layer 3 Confusion must be labeled as "Negative" and must not be used for employee discipline.

### 2. Handover Specifications
*   **HDD format:** HDD 4TB/8TB, formatted as **NTFS** or **exFAT** (Windows/Linux compatible).
*   **Folder structure:**
    ```
    /HDD_01
      ├── /Zone_01_CleanRoom
      │     ├── /Cam_01_Dome5MP
      │     │     ├── 20231025_Day_Shift.mp4
      │     │     ├── 20231025_Night_Shift.mp4
      │     │     └── metadata.json (Model, Focal Length, Mounting Height)
      │     └── ...
      ├── /Zone_04_Warehouse
      └── ...
    ```
*   **Video Codec:** H.264 or H.265 (original from NVR, do not transcode to avoid detail loss).

### 3. Daily Quality Check
*   [ ] **Lens Cleanliness:** Not fogged by grease/dust (especially Zone 8 & Zone 14).
*   [ ] **IR Function:** Working properly at night (check at 22:00).
*   [ ] **Time Sync:** Camera time synchronized with NTP Server.
*   [ ] **FPS Stability:** $\ge$ 15 (walking behavior), $\ge$ 25 (forklift/fall events).
## H. Output Summary

**Total Camera-Days:** $7H + 3N$ | **Sampling hours:** $(7H + 3N) \times 3$ | **Staged clips:** ~1,600 (10-20s each).

## I. Data Quality & Augmentation

### <a id="labeling-rules"></a>1. Labeling Rules
*   **Bounding Box:** Draw tight bounding boxes flush with object edges.
*   **Class Definition:**
    *   **Helmet:** `HELMET` (worn correctly), `NO_HELMET` (not worn), `INCORRECT_HELMET` (worn backwards / strap unfastened — if needed).
    *   **Phone Usage:** Only label `PHONE` when a smartphone is visible. PDA, scanner, and crane remote devices must be labeled as `BACKGROUND` (or a separate `TOOL` class if needed for negative training).
    *   **Fire/Smoke:**
        *   Combustion smoke (brown/black/orange) = `SMOKE`.
        *   Steam/chemical vapor (white, quickly dissipating) = `BACKGROUND` (do not label as smoke).

### 2. Augmentation & Balance
*   **Augmentations:** Blur/noise (ISO grain, transmission), brightness/contrast (flickering lights, sunlight), mosaic/cutout (occlusion by machinery/pillars).
*   **Class balance:** Positive/Negative ratio **1:1** or **1.2:1**. Focus on Confusion Cases (hard negatives) to reduce FP.
*   **Camera maintenance:** Clean lenses **weekly** in chemical/dust areas. Verify no acid residue fogging before collection.

## J. Augmentation & VLG Pipeline

### 1. Definitions
*   **Aug:** Image transforms on real frames (no scene content changes). **VLG-edit:** VLM inpainting/outpainting to create hard negatives. **Aug + VLG-edit:** Both layers (Aug first, VLG second).

### 2. Usage Rules
*   VLG data for training/fine-tuning only -- **never for final evaluation**. VLG-edit $\le$ **15%** of total training data per use case.
*   Prefer VLG-edit on real frames from correct zone/camera (matching angle, height, distortion); avoid pure text-to-image.

### 3. Augmentation Pipeline (for CCTV video)
1.  **Frame extraction:** Preserve metadata (zone, cam, time, IR mode).
2.  **Augmentations:** Motion blur, Gaussian/ISO noise, H.264/H.265 artifacts, brightness/contrast, glare, haze, cutout/mosaic.
3.  **Preserve labels:** Apply transforms to bbox/mask simultaneously; discard if >70% bbox area lost.
4.  **QA:** Review 100 samples/day per zone for unrealistic artifacts.

### 4. VLG-edit Pipeline (for frame-level object detection)
**Best for:** Helmet/Shoes/Phone hard negatives, moderate smoke/steam/dust overlays.
1.  **Background frame:** Real frame from Nitto CCTV (matching zone/camera/IR mode).
2.  **Edit region:** Mask target area only (head, shoes, hand, etc.) to preserve full scene.
3.  **Prompts:** Constrain to "CCTV top-down, indoor factory, same angle/lighting, no text/logos/watermark." Specify target modification. Preserve floor markings, machines, background.
4.  **Post-checks:** No generated text/symbols on PPE. No identifiable faces. No "extra limb" artifacts.
5.  **Labeling:** Auto-label + human review; verify class consistency per [Section I.1](#labeling-rules).

### 5. Per Use Case Recommendations
*   **B/F (Helmet/Shoes):** VLG-edit effective for type/color/state changes, then Aug for camera noise.
*   **A (Fire/Smoke):** VLG supplementary only; prioritize staged/real data for high-risk cases.
*   **G (Fall):** Prioritize real sequences; Aug for lowlight/blur; VLG sparingly for hard negatives.
*   **I/H (Intrusion/Rules):** VLG for clothing/badge/handheld diversity; real sequences for behavior/flow.

### 6. VLG Prompt Catalog

> **Reference catalog** -- consult when preparing VLG prompts. IDs referenced in Augmentation column of Section C.

#### 6.1 Base prompt (shared)
**Input:** 1 real frame + 1 mask. **Base prompt:**
```
You are editing a real CCTV frame from an indoor factory. Keep the same camera angle, height, lens distortion, framing, and lighting style. Preserve the original background layout outside the mask.
Only modify the content inside the mask. Do not change floor markings, machine shapes, labels, signs, or any background objects outside the mask.
No text, no logos, no watermark. Do not create a clear human face. Keep a realistic CCTV look (slight noise and compression).
```

#### 6.2 Per use case prompts
#### <a id="catalog-A"></a> A (Fire/Smoke)
*   **P-A-SMOKE-BLACK:** "Add a thick dark smoke plume rising from the specified source area, natural turbulent edges, no flames, realistic opacity."
*   **P-A-SMOKE-WHITE:** "Add a low-lying white chemical smoke cloud, spreading near the floor, semi-transparent, realistic diffusion."
*   **P-A-SPARK:** "Add brief electrical sparking near the control cabinet area, small bright flickers, no large flames."
*   **P-A-SMOKE-TRASH:** "Add light gray smoke coming from a trash bin, intermittent and weak, realistic flow."
*   **P-A-FLAME-SMALL:** "Add a very small low-intensity flame at floor level (solvent spill), subtle, not cinematic."
*   **P-A-FOGGING:** "Add dense insect-fogging smoke filling the corridor area, uniform haze-like, not originating from burning."
*   **P-A-STEAM-PLUME:** "Add a fast white steam jet plume from an oven/pipe direction, high velocity streaky texture."
*   **P-A-STEAM-KITCHEN:** "Add kitchen steam rising from a pot/steamer area, soft white vapor, localized."
*   **P-A-DUST-CLOUD:** "Add a short dust cloud burst near dust collector outlet, brown/gray, granular, quickly dispersing."
*   **P-A-DUST-LIGHT:** "Add a small light dust haze in air, subtle, not like smoke."
*   **P-A-AIR-VENT:** "Add a strong white compressed-air vent plume, directional, short duration look."

#### <a id="catalog-B"></a> B (Helmet / Cleanroom hair)
*   **P-B-HELMET-YELLOW:** "Replace the helmet to a yellow safety helmet with chin strap fastened; keep head pose unchanged."
*   **P-B-HELMET-WHITE:** "Replace the helmet to a white engineer helmet with chin strap fastened; keep head pose unchanged."
*   **P-B-HELMET-COLOR:** "Replace the helmet to a blue or red safety helmet; keep the same helmet shape and realism."
*   **P-B-NO-HELMET:** "Remove the helmet and show natural hair or bald head, consistent with the person's appearance and lighting."
*   **P-B-HELMET-WEAR-WRONG:** "Make the helmet worn incorrectly: strap unfastened or dangling, helmet tilted slightly; keep head anatomy realistic."
*   **P-B-CLEANROOM-HAIR-OUT:** "Make small hair strands visible outside the cleanroom hood edge; subtle and realistic."
*   **P-B-HELMET-ON-TABLE:** "Place a safety helmet on the specified table/shelf area, no person change, realistic scale and shadow."
*   **P-B-PALLET-BAG:** "Add a white plastic bag wrapping on a pallet load, human-like silhouette risk but clearly a bag texture."
*   **P-B-TOWEL-HEAD:** "Add a towel wrapped around head/neck area, realistic fabric folds, no logo."
*   **P-B-HAIRNET:** "Add a transparent hairnet on the head, subtle mesh texture, no helmet."
*   **P-B-BAREHEAD-HAIRVAR:** "Adjust hair style (black hair / bald) while keeping face indistinct and unchanged."
*   **P-B-CAP-HOODIE:** "Replace with a cap or hoodie covering the head, realistic fabric, no branding."

#### <a id="catalog-F"></a> F (Safety Shoes)
*   **P-F-BOOT-OK:** "Replace footwear to high-top safety boots with steel toe, consistent scale and contact shadow."
*   **P-F-SHOE-LACE-OK:** "Replace footwear to low-top safety shoes with laces, realistic details, no logo."
*   **P-F-SHOE-VELCRO-OK:** "Replace footwear to safety shoes with velcro straps, realistic texture."
*   **P-F-NO-SAFETY-SHOE:** "Replace footwear to sandals/slippers or bare feet, realistic foot shape, no distortion."
*   **P-F-HEEL-STEP:** "Make the safety shoe worn as a slip-on with heel stepped on (open heel), subtle and realistic."
*   **P-F-SHOE-COVER:** "Add a transparent/white shoe cover over normal shoes, thin plastic texture."

#### <a id="catalog-H"></a> H (Safety Rules)
*   **P-H-PHONE-IN-HAND:** "Add a smartphone in the hand while walking, natural grip, realistic size, no screen text."
*   **P-H-HANDS-IN-POCKETS:** "Make both hands inside pants pockets while walking, natural arm pose, no extra limbs."
*   **P-H-JAYWALK:** "Adjust walking path to cross diagonally over the marked line area; keep foot placement plausible."
*   **P-H-NO-HANDRAIL:** "Remove hand contact with the handrail while on stairs; keep balance and pose natural."
*   **P-H-NO-LOOK:** "Make the head facing forward without looking left/right at crossing; keep face indistinct."
*   **P-H-PDA-SCANNER:** "Replace the handheld object with a PDA/scanner/radio, realistic industrial device, no text."
*   **P-H-SMALL-OBJECT:** "Replace the handheld object with a small item (wallet, notebook, bottle), realistic."
*   **P-H-POCKET-BRIEF:** "Show a brief hand-to-pocket motion (momentary), subtle, not fully hands-in-pockets."
*   **P-H-STICKY-MAT:** "Add sticky-mat cleaning action cues (foot stomping stance near mat), keep realistic posture."
*   **P-H-ADJUST-EARMUFF:** "Add hand adjusting earmuff or safety glasses, natural motion, no artifacts."
*   **P-H-TALK-TURN:** "Turn head to the side as if talking to someone, keep body direction unchanged."

#### <a id="catalog-I"></a> I (Restricted Area / Intrusion)
*   **P-I-FORKLIFT-IN-WALKWAY:** "Add a forklift entering the green walkway area, correct perspective and shadows."
*   **P-I-PERSON-IN-LANE:** "Place a walking person inside the forklift lane area, realistic scale, no clear face."
*   **P-I-WRONG-WAY:** "Change movement direction to the opposite arrow direction (person or vehicle), keep scene consistent."
*   **P-I-TAILGATE:** "Add a second person closely following behind through the access door, realistic spacing."
*   **P-I-BIKE-IN-ZONE:** "Add a bicycle or small e-bike entering the restricted zone, realistic motion blur if needed."
*   **P-I-AGV-IN-ZONE:** "Add an AGV/robot crossing the forbidden line, consistent industrial look."
*   **P-I-LOITER-DOOR:** "Add a person loitering near the restricted door area, standing posture, no clear face."
*   **P-I-HANDTRUCK:** "Add a manual hand truck in the walkway, permitted-looking scenario."
*   **P-I-CLEANER-CART:** "Add a cleaning cart being pushed, realistic cleaning equipment, no branding."
*   **P-I-CONTRACTOR-VEST:** "Change clothing to a contractor reflective vest (different color), no logo."
*   **P-I-SECURITY-PATROL:** "Change clothing to security uniform with badge shape but no readable text."
*   **P-I-ANIMAL-OUTDOOR:** "Add a dog/cat crossing outdoor/fence area, realistic scale and motion."

#### <a id="catalog-aug"></a>6.3 Augmentation Tag Glossary

| Tag | Meaning |
| :--- | :--- |
| H265 | H.264/H.265 compression artifacts |
| Noise | High-ISO grain (night) or transmission noise |
| MotionBlur | Fast motion or camera vibration blur |
| ExposureShift | Brightness/contrast/exposure changes |
| Glare | Light reflection on glossy surfaces |
| Lowlight / IR | Low-light or infrared B&W mode |
| Haze | Humidity fog or lens condensation |
| Cutout/Mosaic | Occlusion via cutout blocks or mosaic |
| Occlusion30/60 | 30%/60% occlusion by objects or crowd |
| Flicker | Light flickering or small flashes |
| Bloom | White bloom from IR source or insects |
| PoseVar | Pose variation (bend/squat/gesture) |
