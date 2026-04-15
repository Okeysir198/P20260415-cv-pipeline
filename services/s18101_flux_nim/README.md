# Flux 2 Klein 4B NIM Service

NVIDIA NIM container for [Flux 2 Klein 4B](https://build.nvidia.com/black-forest-labs/flux.2-klein-4b) image generation. Pre-built container — no custom code.

## Architecture

- **Port**: 18101 (maps to internal 8000)
- **GPU**: 1x NVIDIA (~8GB VRAM)
- **Container**: `nvcr.io/nim/black-forest-labs/flux.2-klein-4b:latest`
- **Warm-up**: ~1-3 minutes on start (model loading + Triton compilation)

## API

The NIM container exposes the standard NVIDIA inference API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health/ready` | GET | Health check |
| `/v1/infer` | POST | Text-to-image / image-to-image generation |

### POST /v1/infer

**Text-to-image** (no `image` field):
```json
{
  "prompt": "a red fire truck parked in front of a building, photorealistic",
  "seed": 42,
  "steps": 4
}
```

**Image-to-image** (with `image` field):
```json
{
  "prompt": "same scene but during sunset",
  "image": ["data:image/png;base64,..."],
  "seed": 42,
  "steps": 4
}
```

Response:
```json
{
  "artifacts": [
    {"base64": "...", "finishReason": "SUCCESS", "seed": 42}
  ]
}
```

> **Note**: `finishReason` may be `CONTENT_FILTERED` if the safety filter triggers. In that case `base64` is empty. Use safe prompts and source images.

## Quick Start

```bash
cd services/s18101_flux_nim

# Create .env from example
cp .env.example .env
# Edit .env and set your NGC_API_KEY

docker compose up -d

# Wait ~1-3 minutes for warm-up, then:
curl http://localhost:18101/v1/health/ready
```

## Tests

Integration tests in `tests/` cover all endpoints (requires service running on `:18101`):

| File | Endpoint | Tests |
|------|----------|-------|
| `test00_health.py` | `GET /v1/health/ready` | Health check (2 tests) |
| `test01_text2img.py` | `POST /v1/infer` | Text-to-image: basic generation, dimensions, seed determinism, different seeds, visualization (5 tests) |
| `test02_img2img.py` | `POST /v1/infer` | Image-to-image: basic, from file, output dimensions, visualization (4 tests) |

```bash
# Start the service first
docker compose up -d

# Run tests (from this directory)
uv run pytest tests/ -v

# Run individual test files
uv run pytest tests/test00_health.py -v
uv run pytest tests/test01_text2img.py -v
uv run pytest tests/test02_img2img.py -v
```

Tests skip gracefully if the service is not running. Test data lives in `tests/data/`, visualizations are saved to `tests/outputs/`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NGC_API_KEY` | Yes | NVIDIA NGC API key for container authentication |
| `NVIDIA_VISIBLE_DEVICES` | No | GPU visibility (default: all) |

## Model Details

- **Model**: Flux 2 Klein 4B (Black Forest Labs)
- **Parameters**: 4B
- **License**: NVIDIA NIM EULA
- **Default steps**: 4 (fast inference)
- **Output**: 1024x1024 JPEG
- **Best for**: Text-to-image generation, img2img editing

## GB10 / Jetson Compatibility Notes

This service includes two workarounds for Blackwell GB10 / Jetson devices:

1. **`cuda_platform_patched.py`** — Patches sglang's NVML memory query which fails on GB10 because `nvmlDeviceGetMemoryInfo` raises `NVMLError_NotSupported`. The patched version falls back to `torch.cuda.get_device_properties()`.

2. **`TRITON_PTXAS_PATH`** — Triton's bundled ptxas is CUDA 12.8 and doesn't know `sm_121a` (Blackwell). This env var points Triton to the system ptxas (CUDA 13.0) which supports it.

These patches are only needed for GB10/Jetson. For standard datacenter GPUs (A100, H100, L40), you can use the official NVIDIA command directly:

```bash
# Login to NGC
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Create cache directory
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 $LOCAL_NIM_CACHE

# Run container directly (no patches needed for standard GPUs)
docker run -it --rm --name=nim-server \
  --runtime=nvidia --gpus='"device=0"' \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache/" \
  nvcr.io/nim/black-forest-labs/flux.2-klein-4b:latest
```

You can also run the container directly following NVIDIA's official documentation:

```bash
# Login to NGC
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# Create cache directory
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod 777 $LOCAL_NIM_CACHE

# Run container
docker run -it --rm --name=nim-server \
  --runtime=nvidia --gpus='"device=0"' \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache/" \
  nvcr.io/nim/black-forest-labs/flux.2-klein-4b:latest
```
