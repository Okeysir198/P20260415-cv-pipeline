# Phase 2 — Playwright-driven Label Studio sanity check

Goal: prove that browser-driven Label Studio automation works on this machine before handing off to `tests/test_p12_raw_pipeline.py`. The pytest runs an API-driven "accept all predictions" roundtrip — that covers the LS bridge and REST surface. What pytest can't exercise is the UI itself. This phase closes that gap with a brief browser ceremony.

Use the Playwright MCP tools under `mcp__plugin_playwright_playwright__*`.

## Strategy — minimum viable sanity (default)

Three tool calls are enough to prove the browser path works end-to-end:

1. `browser_navigate` → `http://localhost:18103/user/login/`
2. Fill & submit the login form.
3. `browser_navigate` → `http://localhost:18103/projects`, assert no 500/403.

If all three succeed, Phase 2 is done. The actual LS roundtrip happens in Phase 3's pytest.

## Strategy — extended per-task browser review (optional)

If the user explicitly asks for "full browser review" or "click through every task" (rare — usually only when debugging a UI-specific regression), use this extended procedure. It re-creates its own LS project so it doesn't conflict with pytest's project, and uses "3 UI submissions + API bulk-submit" to stay under 1 minute total.

## Inputs

- `LS_URL = http://localhost:18103`
- `LS_API_KEY` — already in env (Phase 1 prerequisite)
- Admin credentials for the UI: `admin@admin.com` / `admin123` (from docker-compose.yaml)
- For the extended procedure: you'll create a temporary project named `e2e_smoke_browser_review` and drive it yourself (separate from pytest's `test_raw_pipeline_review`).

## Part A — Browser-driven review (3 tasks) [extended only]

Action sequence using the Playwright MCP tools. Each step is one tool call. If any step errors, stop the phase and report.

1. `browser_navigate` → `http://localhost:18103/user/login/`
2. `browser_snapshot` — confirm the login form is present (look for `input[name="email"]`).
3. `browser_fill_form` with:
   - email field: `admin@admin.com`
   - password field: `admin123`
4. `browser_click` on the Submit/"Log In" button.
5. `browser_wait_for` → text "Projects" (dashboard loaded).
6. `browser_navigate` → `http://localhost:18103/projects/<PROJECT_ID>/data`
7. `browser_snapshot` — confirm the task queue is visible.
8. `browser_click` on the first task row (opens the labeling UI).
9. `browser_wait_for` → the canvas/annotation region is ready.
10. `browser_press_key` → `Ctrl+Enter` (standard LS submit-and-next shortcut). If that doesn't advance, fall back to clicking the "Submit" button via `browser_click`.
11. Repeat steps 9–10 two more times so 3 tasks are submitted through the UI.
12. `browser_close`.

If the UI shortcuts have changed in a newer LS version and Ctrl+Enter doesn't submit, use `browser_snapshot` to locate the visible "Submit" button and `browser_click` it. Do not give up and skip — re-read the snapshot and adapt.

## Part B — API bulk-submit (remaining tasks) [extended only]

After Part A, run this inline script to post an annotation to every remaining un-annotated task. The body copies each task's `predictions[0].result` into a new annotation (i.e. "accept the pre-annotation as-is"):

```bash
uv run python - <<'PY'
import os, requests
LS_URL = "http://localhost:18103"
PROJECT_ID = int(os.environ["PROJECT_ID"])
headers = {"Authorization": f"Token {os.environ['LS_API_KEY']}"}

# Page through all tasks
page, posted, skipped = 1, 0, 0
while True:
    r = requests.get(
        f"{LS_URL}/api/tasks",
        params={"project": PROJECT_ID, "page": page, "page_size": 100},
        headers=headers, timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    tasks = data.get("tasks") or data.get("results") or []
    if not tasks:
        break
    for t in tasks:
        if t.get("annotations"):
            skipped += 1
            continue
        preds = t.get("predictions") or []
        result = preds[0].get("result", []) if preds else []
        body = {"result": result, "task": t["id"]}
        resp = requests.post(
            f"{LS_URL}/api/tasks/{t['id']}/annotations/",
            json=body, headers=headers, timeout=30,
        )
        resp.raise_for_status()
        posted += 1
    page += 1
    if len(tasks) < 100:
        break

print(f"posted={posted} skipped_already_annotated={skipped}")
PY
```

Before running, set `PROJECT_ID` in the environment. On success, the script prints a line like `posted=97 skipped_already_annotated=3` — log it to the final report.

## Post-condition check

Before leaving Phase 5, verify the project's annotation count hits the task count:

```bash
curl -sf -H "Authorization: Token $LS_API_KEY" \
  "http://localhost:18103/api/projects/$PROJECT_ID/" \
  | uv run python -c "
import json, sys
p = json.load(sys.stdin)
print(f'tasks={p.get(\"task_number\",\"?\")} annotated={p.get(\"num_tasks_with_annotations\",\"?\")}')"
```

If `annotated < tasks`, there are tasks the API call missed — usually because LS has multi-page pagination and the while-loop broke early. Re-check the loop and re-run Part B.

## Why not click all 100?

- Time: ~5 s per click × 100 = 8+ minutes of Playwright.
- Flakiness: a single timing mismatch in the UI fails the whole phase.
- No added signal: the skill is already exercising the UI path 3 times; bulk-submit exercises the API path once. Together they cover both integrations.

If the user explicitly wants "every task reviewed through the browser", make it a parameter (`--ui-tasks all`) and warn about the time cost — but default to 3 + API.

## Troubleshooting

- **Login 403** — `bootstrap.sh` wasn't run; run it from `services/s18103_label_studio/` and re-export `LS_API_KEY`.
- **POST /annotations/ returns 400** — the `result` format doesn't match the project's labeling config. Inspect one prediction with `curl .../api/tasks/<id>` and mirror its `result` shape exactly.
- **Playwright times out on `wait_for` text** — LS UI text may have changed between versions. Use `browser_snapshot` to see the current DOM and adjust.
