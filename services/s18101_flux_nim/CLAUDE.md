# CLAUDE.md — s18101_flux_nim

Flux 2 Klein 4B NIM service — pre-built NVIDIA container with GB10/Jetson patches.

## Critical: GB10/Jetson Requires Patches

This service has TWO patches that are **required** for GB10/Jetson, not optional:

| Patch | File | Error without it |
|-------|------|------------------|
| NVML memory query fallback | `cuda_platform_patched.py` volume mount | `NVMLError_NotSupported` at `get_device_total_memory` |
| Triton ptxas CUDA 13.0 | `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | `PTXAS error: Internal Triton PTX codegen error` |

If you remove either patch, the service will start but fail during warm-up with the errors above.

**Do NOT remove these patches for GB10/Jetson deployments.** For standard datacenter GPUs (A100, H100, L40), neither patch is needed — use the official NVIDIA docker run command instead (see README).

## Model Limitations

**Quality**: This NIM has a **hard limit of 4 steps** — images will be rough/low quality. This is a speed-optimized configuration, not suitable for production visuals.

| Limitation | Value | Notes |
|------------|-------|-------|
| Max steps | 4 | API rejects `steps > 4` with 422 error |
| Output size | 1024x1024 | Fixed |
| Best for | Testing, prototyping | Not production-quality images |

For higher quality, consider hosting the model yourself (no step limit) or using a different model.

## API

```
POST /v1/infer           Generate image (requires warm service)
GET  /v1/health/ready    Readiness check (200 when warm)
```

**Request** (`POST /v1/infer`):
```json
{
  "prompt": "a cat sitting on a red chair",
  "negative_prompt": "",
  "steps": 4,
  "seed": 42
}
```
**Response**: PNG image bytes (`Content-Type: image/png`). Save with: `open('out.png', 'wb').write(response.content)`. `steps` must be ≤ 4 (hard limit), default 4.

## Quick Commands

```bash
# First-time setup: create .env with NGC API key
echo "NGC_API_KEY=<your-key>" > .env

# From project root
cd services/s18101_flux_nim

# Restart service (after code/config changes)
docker compose down && docker compose up -d

# Follow logs for warm-up (takes 1-3 minutes)
docker logs -f s18101_flux_nim-flux-nim-1

# Check health
curl http://localhost:18101/v1/health/ready

# Run tests (service must be running)
uv run pytest tests/ -v
```

## Troubleshooting

| Symptom | Check | Solution |
|---------|-------|----------|
| Health check fails after 5 min | `docker logs ... \| grep -E "PTXAS\|NVMLError"` | Missing TRITON_PTXAS_PATH or cuda_platform patch |
| Container crashes on start | `docker logs ... \| tail -50` | Check NGC_API_KEY is set in .env |
| Tests skip | Service not running | Start with `docker compose up -d` |

## Warm-up Timeline

1. **0-30s**: Container starts, loads cached models
2. **30-120s**: Pipeline initialization (compiles Triton kernels)
3. **120-180s**: Warm-up (generates test images)
4. **Ready**: Health endpoint returns 200

Look for `"Pipeline warmup: done"` in logs to confirm readiness. Until then, requests return 503.

## Usage Example

```bash
# Generate image (service must be warm — check /v1/health/ready first)
curl -s http://localhost:18101/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a factory worker wearing a safety helmet", "steps": 4, "seed": 42}' \
  -o output.png

# Python
import httpx
resp = httpx.post("http://localhost:18101/v1/infer", json={
    "prompt": "fire and smoke in a warehouse",
    "steps": 4, "seed": 42
}, timeout=30)
open("output.png", "wb").write(resp.content)
```
