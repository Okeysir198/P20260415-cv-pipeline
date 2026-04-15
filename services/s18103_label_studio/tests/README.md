# Label Studio E2E Tests

Playwright-based tests validating the auto-annotation → Label Studio visualization workflow.

Uses the existing Label Studio container — creates a separate project (`e2e_fire_review`) for test isolation.

> **Note:** The tests connect to Label Studio on port **8080** (the container-internal default). The host-side port is **18103** — use `http://localhost:18103` for browser access. The `docker-compose.yaml` maps host port 18103 → container port 8080.

## Prerequisites

```bash
uv sync --extra playwright
uv run playwright install chromium
```

## Running

```bash
# Start Label Studio (if not already running)
cd services/s18103_label_studio && docker compose up -d && cd -

# Run tests (headless)
uv run pytest services/s18103_label_studio/tests/ -v

# Run tests (headed, for debugging)
uv run pytest services/s18103_label_studio/tests/ -v --headed

# Run with SAM3 full test (start SAM3 first)
cd services/s18100_sam3_service && docker compose up -d
uv run pytest services/s18103_label_studio/tests/ -v
```

## How It Works

1. **First run**: Signs up a test account via Playwright, extracts the API key, saves credentials to `.credentials.json`
2. **Subsequent runs**: Logs in with saved credentials, reuses API key
3. **Lite mode** (always): Uses fixture labels as mock auto-annotate output → imports → verifies visualization
4. **Full mode** (if SAM3 available): Runs real SAM3 auto-annotation → imports → verifies

## Credential Management

- Credentials are saved to `services/s18103_label_studio/tests/.credentials.json` (gitignored)
- Delete this file to force a fresh signup

## Screenshots

Test screenshots are saved to `services/s18103_label_studio/tests/screenshots/` for debugging.
