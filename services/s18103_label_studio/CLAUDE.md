# CLAUDE.md — services/s18103_label_studio/

Off-the-shelf Label Studio container for human-in-the-loop annotation review. No custom code — configured entirely via docker-compose environment variables. CPU-only.

## Architecture

```
core/p04_label_studio/bridge.py  →  Label Studio (:18103)  ←  Browser (reviewer)
       ↑                              ↑
  dataset_store/training_ready/    bind-mounted read-only at /datasets/training_ready
```

Label Studio is the **human review** component:
1. `core/p01_auto_annotate/` generates auto-annotations via SAM3
2. `core/p02_annotation_qa/` flags problematic labels
3. `core/p04_label_studio/bridge.py` pushes pre-annotations into Label Studio for review
4. Reviewers correct annotations in the browser UI
5. `bridge.py export` pulls reviewed annotations back to YOLO `.txt` files

Label Studio itself makes **no outbound calls** to other services. The bridge CLI mediates all data flow.

## Running (fresh install)

```bash
cd services/s18103_label_studio
docker compose up -d                 # start container
./bootstrap.sh                       # enable legacy tokens + print admin token
# ADMIN_EMAIL=admin@admin.com
# ADMIN_TOKEN=1fc75e70df...
export LS_API_KEY=<ADMIN_TOKEN>      # for bridge CLI
```

First-run admin is created from `LABEL_STUDIO_USERNAME` / `LABEL_STUDIO_PASSWORD` in `docker-compose.yaml` (default: `admin@admin.com` / `admin123`). The bootstrap only runs when the `ls-data` volume is fresh, i.e. on first start or after `docker compose down -v`. `bootstrap.sh` is idempotent — safe to re-run.

## Port

Host `18103` → Container `8080`.

## Volume Mounts

| Host Path | Container Path | Access | Purpose |
|-----------|---------------|--------|---------|
| Named volume `ls-data` | `/label-studio/data` | read-write | LS database (SQLite), user data, exports |
| `../../dataset_store` | `/datasets` | read-only | Full dataset tree (raw + training_ready + site_collected) |
| `../../tests/fixtures/data` | `/dataset_store/test_fixtures` | read-only | Test fixture images for E2E tests |

Data persists across restarts in `ls-data`. Use `docker compose down -v` to destroy.

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` | `true` | Enables serving files from local filesystem mounts |
| `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` | `/datasets/training_ready` | Root dir for local file serving (must NOT equal any per-dataset storage path — see gotcha below) |
| `LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK` | `false` | Allows open signup |
| `LABEL_STUDIO_LEGACY_API_TOKEN_AUTH_ENABLED` | `true` | Required, but not sufficient — see gotcha below |
| `LABEL_STUDIO_USERNAME` | `admin@admin.com` | Bootstrap admin email (first-run only) |
| `LABEL_STUDIO_PASSWORD` | `admin123` | Bootstrap admin password (first-run only; ≥8 chars required) |

No `.env` file needed — all config is in docker-compose.yaml.

## Bridge CLI Integration

All Label Studio interaction goes through `core/p04_label_studio/bridge.py`:

```bash
export LS_API_KEY=<token from bootstrap.sh>

# Create project with class labels + local-files storage connector
uv run core/p04_label_studio/bridge.py \
  --email admin@admin.com --password admin123 \
  setup --data-config features/<feature>/configs/05_data.yaml

# Import YOLO pre-annotations as tasks
uv run core/p04_label_studio/bridge.py \
  --email admin@admin.com --password admin123 \
  import --data-config features/<feature>/configs/05_data.yaml

# Export reviewed annotations back to YOLO format
uv run core/p04_label_studio/bridge.py \
  --email admin@admin.com --password admin123 \
  export --data-config features/<feature>/configs/05_data.yaml
```

Config: `configs/_shared/04_label_studio.yaml` — URL, API key, import/export settings, label colors.

**Auth model** — two layers:
- `LS_API_KEY` (or `--api-key`) — used for most endpoints (tasks, projects, annotations).
- `--email` / `--password` — required for `setup` because **creating a local-files storage connector needs a Django session** (token auth doesn't have that permission in LS 1.23+). Put the email/password flags **before** the subcommand (they're top-level args).

## E2E Tests

Playwright-based tests in `tests/`:

```bash
cd services/s18103_label_studio
uv sync --extra playwright
uv run playwright install chromium
uv run pytest tests/ -v     # requires Label Studio running
```

8 ordered tests: signup/login → create project → import annotations → verify tasks → verify labels → export round-trip. Tests auto-skip if service is not running. Screenshots saved to `tests/screenshots/`.

## Gotchas (LS 1.23+)

- **Legacy token auth needs both env var AND per-org DB flag.** `LABEL_STUDIO_LEGACY_API_TOKEN_AUTH_ENABLED=true` is necessary but not sufficient — each `Organization` row has a `legacy_api_tokens_enabled` field that defaults to `False`. `bootstrap.sh` flips it via Django shell. Without it, every request returns `401: "legacy token authentication has been disabled for this organization"`.

- **Storage path cannot equal DOCUMENT_ROOT.** LS rejects `POST /api/storages/localfiles/` with `"Absolute local path '/X' cannot be the same as LOCAL_FILES_DOCUMENT_ROOT='/X' by security reasons"`. Fix: DOCUMENT_ROOT is the **parent** (`/datasets/training_ready`), and per-dataset storage paths are sub-dirs (`/datasets/training_ready/<feature_name>`). The bridge's `setup` emits task URLs as `?d=<feature_name>/...` which then resolves against DOCUMENT_ROOT correctly.

- **Bootstrap admin only created on empty volume.** `LABEL_STUDIO_USERNAME` / `PASSWORD` env vars seed the DB only when `ls-data` is fresh. After `docker compose down -v` + `up -d`, you need to re-run `./bootstrap.sh` to re-enable legacy tokens on the new org.

- **Storage connector creation needs session auth.** The bridge's `setup` command auto-creates the connector when `--email` + `--password` are passed; the SDK-level token alone gets "No authenticated session available" and silently skips it.

- **Read-only dataset mount** — `dataset_store` is `:ro`. LS can display images but cannot modify source data. All writes go through bridge CLI export.

- **Images served via local file storage** — Images are NOT uploaded into LS. They're served directly from the `/datasets` mount, avoiding data duplication.

- **Test port** — Tests connect on port `8080` (container-internal). To run tests from the host, use the mapped port `18103`.

- **No GPU required** — Pure CPU service, standard container image.
