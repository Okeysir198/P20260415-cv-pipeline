"""Pytest conftest for Label Studio E2E Playwright tests.

Session-scoped fixtures for health checks, credentials, and screenshot
capture. Uses the existing Label Studio container — no separate test compose.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests


# ---------------------------------------------------------------------------
# Helper functions (module-level)
# ---------------------------------------------------------------------------

def wait_for_health(url: str, timeout_s: int = 90, interval_s: int = 3) -> None:
    """Poll GET *url* until HTTP 200 or *timeout_s* exceeded."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(interval_s)
    msg = f"{url} not healthy after {timeout_s}s"
    if last_error:
        msg += f" (last error: {last_error})"
    raise TimeoutError(msg)


def is_label_studio_running() -> bool:
    """Check whether Label Studio is reachable at localhost:18103."""
    try:
        resp = requests.get("http://localhost:18103/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Credentials manager
# ---------------------------------------------------------------------------

_DEFAULT_EMAIL = "nthanhtrung198@gmail.com"
_DEFAULT_PASSWORD = "Trung123"

_CREDS_FILENAME = ".credentials.json"


@dataclass
class _CredentialsManager:
    """Manages E2E test credentials stored in a JSON file.

    On first run the file does not exist — default credentials are generated
    and ``needs_signup`` returns ``True``.  After ``save()`` is called the
    file is written and subsequent runs load from it.
    """

    path: Path
    email: str = _DEFAULT_EMAIL
    password: str = _DEFAULT_PASSWORD
    api_key: str = ""
    _loaded_from_file: bool = field(default=False, repr=False)

    @property
    def needs_signup(self) -> bool:
        return not self._loaded_from_file

    def load(self) -> None:
        """Load credentials from disk if the file exists."""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.email = data.get("email", _DEFAULT_EMAIL)
            self.password = data.get("password", _DEFAULT_PASSWORD)
            self.api_key = data.get("api_key", "")
            self._loaded_from_file = True

    def save(self, creds: dict[str, str] | None = None) -> None:
        """Persist credentials to disk."""
        if creds:
            self.email = creds.get("email", self.email)
            self.password = creds.get("password", self.password)
            self.api_key = creds.get("api_key", self.api_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"email": self.email, "password": self.password, "api_key": self.api_key},
                indent=2,
            )
            + "\n"
        )
        self._loaded_from_file = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the camera_edge project root."""
    return Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture(scope="session", autouse=True)
def label_studio_health() -> None:
    """Skip the entire session if Label Studio is not running.

    Start it manually: ``cd services/s18103_label_studio && docker compose up -d``
    """
    if not is_label_studio_running():
        pytest.skip(
            "Label Studio not running at localhost:8080. "
            "Start it with: cd services/s18103_label_studio && docker compose up -d"
        )


@pytest.fixture(scope="session")
def sam3_available() -> bool:
    """Check whether the SAM3 service is reachable (does NOT start it)."""
    try:
        resp = requests.get("http://localhost:18100/health", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def screenshots_dir() -> Path:
    """Create and return the directory for Playwright screenshots."""
    d = Path(__file__).resolve().parent / "screenshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def credentials_manager() -> _CredentialsManager:
    """Session-scoped credentials manager backed by ``.credentials.json``."""
    creds_path = Path(__file__).resolve().parent / _CREDS_FILENAME
    mgr = _CredentialsManager(path=creds_path)
    mgr.load()
    return mgr
