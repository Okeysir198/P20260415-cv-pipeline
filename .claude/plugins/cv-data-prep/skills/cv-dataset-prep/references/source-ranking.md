# Source Ranking — Quality-First Selection

Pick 2–4 sources covering all canonical classes. **Volume is a tiebreaker, not a criterion.**

## Ranking order

1. **Label provenance** — paper-authored / human-verified > Roboflow top-rated (≥50 downloads) > Kaggle community > untrusted (opaque names, 0 downloads)
2. **Negative coverage** — sources that provide explicit negative classes are more valuable than pure-positive sources
3. **Scene diversity** — geographic variety, lighting mix, camera angles
4. **Volume** — tiebreaker only; reject near-duplicate mirrors of already-picked sources

## License tiers

- ✅ Apache-2.0, MIT, CC-BY-4.0, CC0 — commercial-friendly
- ⚠️ CC-BY-NC-SA-4.0 — non-commercial only
- ⚠️ AGPL-3.0 — may propagate to trained model
- ⚠️ competition-use — requires rules acceptance before download

Record in the source's `license:` field in `00_data_preparation.yaml`.

## Exclusion checklist

Drop any source that is:
- [ ] A stub (README only, 0 images) — check `dataset_store/CLAUDE.md` for known stubs
- [ ] Opaque-classed (`"0"`, `"14"`, auto-generated name strings)
- [ ] Near-duplicate of an already-picked source
- [ ] Flagged "spot-check labels" in `dataset_store/CLAUDE.md`

## Recording decisions

- Picked sources: fill `license:`, `notes:`, `dropped_classes:` per source entry.
- Excluded sources: add to the top-level `held_back:` list with `name`, `reason`, `when`.
  Do not use YAML comments — `held_back:` appears in `DATASET_REPORT.md`.
