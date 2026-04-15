# Source Ranking — Quality-First Dataset Selection

For a v1 training set, pick 2–4 sources that together cover the canonical classes with diverse scenes. **Volume is a tiebreaker, not a primary criterion.** Label noise caps achievable mAP; redundant near-duplicate images waste compute without improving generalization.

## Ranking criteria (in order)

### 1. Label provenance
| Tier | Description |
|---|---|
| 🥇 Human-verified / paper-authored | Academic datasets (arXiv published), author-verified splits (e.g. sh17, D-Fire). |
| 🥈 Roboflow top-rated | ≥50 downloads AND ≥1 star, workspace has multiple projects. |
| 🥉 Kaggle community | Mirror of a Roboflow export, or a community-labeled upload. Spot-check first. |
| ⚠️ Untrusted | Opaque class names (`"0"`, `"14"`, `mobile_dataset - v9 ...`), 0 downloads, solo author with no other projects. **Drop unless user insists.** |

### 2. Scene/lighting diversity
Look for:
- **Geographic variety** — not all one factory / one country.
- **Lighting** — mix of daylight, indoor, low-light.
- **Camera angle** — CCTV top-down vs. eye-level staged.
- **Subject pose/context** — workers in action vs. studio shots.

A 1K-image dataset from 50 scenes beats a 10K dataset that's all near-duplicates from one facility.

### 3. Class coverage (with negatives)
For detection, the canonical class set must include **negatives** to prevent false positives (e.g. `no_helmet`, `no_shoes`, `no_glove`). Sources that provide negatives explicitly (NO-Hardhat, no_shoes, barefoot) are more valuable than pure-positive sources.

A source's contribution to the v1 is scored by:
- How many canonical classes it covers.
- Whether it provides negatives.
- Whether it provides `person` alongside the detail classes (for downstream tracking).

### 4. Volume (tiebreaker only)
All else equal, prefer the larger source. But reject:
- Near-duplicate mirrors of other picked sources.
- Community-scraped Kaggle datasets that overlap a paper dataset.

## License considerations
Datasets propagate their license to trained models under some regimes (AGPL, CC-BY-NC-SA). When picking commercial-friendly sources:
- ✅ Apache-2.0, MIT, CC-BY-4.0, CC0, Public Domain
- ⚠️ CC-BY-NC-SA-4.0 — non-commercial only
- ⚠️ AGPL-3.0 — may force model to inherit AGPL (e.g. FPI-Det)
- ⚠️ competition-use — requires rules acceptance (e.g. Kaggle state-farm-distracted)

Record each picked source's license in a comment inside `00_data_preparation.yaml`.

## Exclusion checklist

Before accepting a source, verify it is **not**:

- [ ] A stub folder (README only, 0 images) — see `dataset_store/CLAUDE.md` for known stubs.
- [ ] Opaque-classed (`"0"`, `"14"`, or versioned-name like `"mobile_dataset - v9 2025-04-05"`).
- [ ] A near-duplicate of an already-picked source (check arXiv / Roboflow lineage).
- [ ] Flagged `"spot-check labels"` in the source registry in `dataset_store/CLAUDE.md`.

## Output of this step

For each picked source, record in `00_data_preparation.yaml` comments:
- Why picked (one phrase).
- Volume + key classes.
- License.

For dropped sources, record in a block comment at the top of the YAML: which ones, why, and under what condition you'd add them in v2.
