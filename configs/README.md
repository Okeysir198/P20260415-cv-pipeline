# configs/ — Cross-Feature Infrastructure

This directory holds only cross-feature infrastructure; **feature configs
are authoritative under `features/<name>/configs/`**. Features never fall
back to `_shared/` at runtime — copy a template into your feature if you
need to override it.

```
configs/
  _shared/   non-authoritative pipeline templates (01–04, HPO, export)
  _test/     CI smoke-test fixtures
  CLAUDE.md  authoritative schema reference for every phase YAML
  README.md  ← you are here
```

## Where to go

| You want to … | Look at |
|---|---|
| Understand the phase-YAML schema (00, 05, 06, 10, …) | [`CLAUDE.md`](CLAUDE.md) |
| See the end-to-end CLI workflow (labeled + raw-unlabeled) | [root `README.md`](../README.md) |
| Learn the feature folder layout + naming contract | [`features/README.md`](../features/README.md) |
| Scaffold a new feature | `bash scripts/new_feature.sh <category>-<name>` |
| Override any value at runtime | `--override training.lr=0.005` (see [root README](../README.md)) |

## Numbering

| # | Step | Config location |
|---|---|---|
| 00 | Data preparation | per-feature |
| 01 | Auto-annotate | `_shared/` |
| 02 | Annotation QA | `_shared/` |
| 03 | Generative augment | `_shared/` or per-feature override |
| 04 | Label Studio | `_shared/` |
| 05 | Data definition | per-feature |
| 06 | Training | per-feature |
| 07 | HPO | `_shared/` (file: `08_hyperparameter_tuning.yaml`) |
| 08 | Evaluation | CLI-only, no YAML |
| 09 | Export | `_shared/` |
| 10 | Inference + alerts | per-feature (carries `alerts:`, `tracker:`, `samples:`) |

## Key rules

1. **No inheritance** — each config file is complete and self-contained.
2. **Paths are relative from project root** — `../../dataset_store/`,
   `../../pretrained/`.
3. **`${var}` interpolation is in-file only** — no cross-file refs.
4. **`data.dataset_config: 05_data.yaml`** in a training config resolves
   relative to that training config's directory.

See [`CLAUDE.md`](CLAUDE.md) for the complete schema of every YAML.
