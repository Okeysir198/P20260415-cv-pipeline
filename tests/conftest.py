"""Shared pytest fixtures for the camera_edge test suite.

Provides session-scoped fixtures for the test dataset and common
configuration, so that individual test modules don't need to re-create
them.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for all tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config

DATA_CONFIG_PATH = ROOT / "configs" / "_test" / "05_data.yaml"
TRAIN_CONFIG_PATH = ROOT / "configs" / "_test" / "06_training.yaml"


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return ROOT


@pytest.fixture(scope="session")
def test_outputs_dir():
    """Return the shared test outputs directory."""
    outputs = ROOT / "tests" / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return outputs


@pytest.fixture(scope="session")
def data_config():
    """Loaded test_fire_100 data config dict."""
    return load_config(str(DATA_CONFIG_PATH))


@pytest.fixture(scope="session")
def train_config():
    """Loaded test_fire_100 training config dict."""
    return load_config(str(TRAIN_CONFIG_PATH))
