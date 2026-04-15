"""Service health check utilities for camera_edge tools."""

import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


_START_COMMANDS: dict[str, str] = {
    "SAM3 :18100": "cd services/s18100_sam3_service && docker compose up -d",
    "SAM3.1 :18106": "cd services/s18106_sam3_1_service && docker compose up -d",
    "Auto-Label :18104": "cd services/s18104_auto_label && docker compose up -d",
    "QA :18105": "cd services/s18105_annotation_quality_assessment && docker compose up -d",
    "Flux :18101": "cd services/s18101_flux_nim && docker compose up -d",
}


def _start_hint(name: str) -> str:
    """Return docker compose start command hint for a service name."""
    return _START_COMMANDS.get(name, "see services/ directory for the relevant docker compose file")


def check_service_health(
    url: str,
    timeout: float = 5.0,
    retries: int = 1,
) -> bool:
    """Return True if the service at *url* responds with a non-5xx status."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status < 500:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if attempt < retries - 1:
            time.sleep(1.0)

    return False


def require_services(
    services: dict[str, str],
    skip: bool = False,
) -> None:
    """Check all required services concurrently. Exits with code 1 if any are unreachable.

    Args:
        services: Dict mapping service name to health URL.
        skip: If True, skip all checks (--skip-health-check flag).
    """
    if skip:
        return

    print("Checking required services...")
    failed = []

    with ThreadPoolExecutor(max_workers=len(services)) as pool:
        futures = {
            pool.submit(check_service_health, url): (name, url)
            for name, url in services.items()
        }
        for future in as_completed(futures):
            name, url = futures[future]
            if future.result():
                print(f"  {name} ({url}) ... OK")
            else:
                print(f"  {name} ({url}) ... UNREACHABLE", file=sys.stderr)
                failed.append(name)

    if failed:
        print(file=sys.stderr)
        for name in failed:
            hint = _start_hint(name)
            print(f"ERROR: Required service '{name}' is not running.", file=sys.stderr)
            print(f"  Start with: {hint}", file=sys.stderr)
        print("  Or skip checks with: --skip-health-check", file=sys.stderr)
        sys.exit(1)

    print()
