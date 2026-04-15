# CLAUDE.md — services/s18103_label_studio/

Off-the-shelf Label Studio container for human-in-the-loop annotation review. No custom code — configured entirely via docker-compose environment variables. CPU-only.

## Architecture

```
core/p04_label_studio/bridge.py  →  Label Studio (:18103)  ←  Browser (reviewer)
       ↑                              ↑
  dataset_store/              bind-mounted read-only at /datasets
```

Label Studio is the **human review** component:
1. `core/p01_auto_annotate/` generates auto-annotations via SAM3
2. `core/p02_annotation_qa/` flags problematic labels
3. `core/p04_label_studio/bridge.py` pushes pre-annotations into Label Studio for review
4. Reviewers correct annotations in the browser UI
5. `bridge.py export` pulls reviewed annotations back to YOLO `.txt` files

Label Studio itself makes **no outbound calls** to other services. The bridge CLI mediates all data flow.

## Running

```bash
cd services/s18103_label_studio && docker compose up -d
curl http://localhost:18103/health

# First visit: create admin account at http://localhost:18103
# Then use bridge CLI for project setup + import/export
```

## Port

Host `18103` → Container `8080`

## Volume Mounts

| Host Path | Container Path | Access | Purpose |
|-----------|---------------|--------|---------|
| Named volume `ls-data` | `/label-studio/data` | read-write | LS database (SQLite), user data, exports |
| `../../dataset_store` | `/datasets` | read-only | Dataset images served via local file storage |
| `../../tests/fixtures/data` | `/dataset_store/test_fixtures` | read-only | Test fixture images for E2E tests |

Data persists across restarts in `ls-data`. Use `docker compose down -v` to destroy.

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` | `true` | Enables serving files from local filesystem mounts |
| `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` | `/datasets` | Root directory for local file serving |
| `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK` | `false` | Allows open signup (no invite link required) |
| `LABEL_STUDIO_LEGACY_API_TOKEN_AUTH_ENABLED` | `true` | Required for bridge CLI token-based API auth |

No `.env` file needed — all config is in docker-compose.yaml.

## Bridge CLI Integration

All Label Studio interaction goes through `core/p04_label_studio/bridge.py`:

```bash
# Create project with class labels from data config
uv run python core/p04_label_studio/bridge.py setup --data-config configs/shoes_detection/05_data.yaml

# Import auto-annotations as pre-annotations
uv run python core/p04_label_studio/bridge.py import --data-config configs/shoes_detection/05_data.yaml

# Export reviewed annotations back to YOLO format
uv run python core/p04_label_studio/bridge.py export --data-config configs/shoes_detection/05_data.yaml
```

Config: `configs/_shared/04_label_studio.yaml` — URL, API key, import/export settings, label colors.

API key priority: `--api-key` CLI arg > `LS_API_KEY` env var > config file value.

## E2E Tests

Playwright-based tests in `tests/`:

```bash
cd services/s18103_label_studio
uv sync --extra playwright
uv run playwright install chromium
uv run pytest tests/ -v     # requires Label Studio running
```

8 ordered tests: signup/login → create project → import annotations → verify tasks → verify labels → export round-trip. Tests auto-skip if service is not running. Screenshots saved to `tests/screenshots/`.

## Gotchas

- **First visit requires account creation** — No default admin. First browser visit triggers signup. E2E tests handle this automatically.
- **Legacy token auth must be enabled** — `LABEL_STUDIO_LEGACY_API_TOKEN_AUTH_ENABLED=true` is required for the bridge CLI's token-based API to work.
- **Read-only dataset mount** — `dataset_store` is `:ro`. LS can display images but cannot modify source data. All writes go through bridge CLI export.
- **Images served via local file storage** — Images are NOT uploaded into LS. They're served directly from the `/datasets` mount, avoiding data duplication.
- **Test port** — Tests connect on port `8080` (container-internal). To run tests from the host, use the mapped port `18103`.
- **No GPU required** — Pure CPU service, standard container image.
