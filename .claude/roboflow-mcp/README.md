# roboflow-mcp

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes the [Roboflow](https://roboflow.com) platform API as tools in [Claude Code](https://claude.ai/code). Manage datasets, trigger training runs, search Universe, and run inference — all from the CLI.

---

## Setup

**Requirements:** Python 3.10+, a Roboflow API key, Claude Code installed.

```bash
git clone https://github.com/nickedridge-wq/roboflow-mcp.git
cd roboflow-mcp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure Claude Code

**Option A — project-level** (recommended, checked into the repo):

```bash
claude mcp add roboflow \
  --env ROBOFLOW_API_KEY=your_api_key_here \
  -- /path/to/roboflow-mcp/venv/bin/python /path/to/roboflow-mcp/server.py
```

This writes a `.mcp.json` file in the current project directory.

**Option B — user-level** (available in all projects):

```bash
claude mcp add roboflow --scope user \
  --env ROBOFLOW_API_KEY=your_api_key_here \
  -- /path/to/roboflow-mcp/venv/bin/python /path/to/roboflow-mcp/server.py
```

Restart Claude Code — the `mcp__roboflow__*` tools will be available immediately.

---

## Tools

| Tool | Description |
|---|---|
| `list_workspaces` | Show workspace name, URL slug, and project count |
| `list_projects` | List all projects in a workspace with type and image counts |
| `get_project` | Get classes, annotation type, and metadata for a project |
| `list_versions` | List all dataset versions with image counts per split |
| `upload_image` | Upload an image and optional annotation to a project |
| `create_version` | Generate a new dataset version with preprocessing and augmentation |
| `download_dataset` | Download a version locally (yolov8, coco, voc, and more) |
| `download_universe_dataset` | Download a public dataset directly from Roboflow Universe |
| `search_universe` | Search Universe for public datasets and pre-trained models |
| `run_inference` | Run inference via a deployed model on a local file or URL |
| `get_model_metrics` | Fetch mAP, precision, and recall for a trained version |

---

## Example Workflows

### Find and download a public dataset

```
search_universe("hard hat detection")
→ pick a result, note workspace + project + version

download_universe_dataset(
  universe_workspace="roboflow-universe-projects",
  universe_project="hard-hat-universe",
  version_number=1,
  model_format="yolov8",
  location="./datasets/hard-hat"
)
```

### Upload images and generate a training version

```
upload_image(project_url="my-project", image_path="/data/img001.jpg",
             annotation_path="/data/img001.xml", split="train")

create_version(
  project_url="my-project",
  preprocessing={"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Stretch to"}},
  augmentation={"flip": {"horizontal": True}, "rotation": {"degrees": 15}}
)
```

### Run inference and check model performance

```
run_inference(project_url="my-project", version_number=3,
              image_path="/data/test.jpg", confidence=60)

get_model_metrics(project_url="my-project", version_number=3)
```

---

## Tests

```bash
python -m unittest test_server -v
```

21 tests covering output suppression, lazy init thread safety, input validation, null model guard, auth error propagation, and parameter contracts. No live API key required.

---

## Implementation Notes

- **Lazy authentication** — the Roboflow SDK authenticates once per session on first tool call, with double-checked locking for thread safety.
- **Output suppression** — the SDK prints to both stdout and stderr on init, which corrupts MCP's stdio transport. All SDK calls redirect both streams.
- `search_universe` and `get_model_metrics` call the Roboflow REST API directly for endpoints not exposed cleanly through the SDK.
