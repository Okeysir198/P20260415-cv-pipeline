"""Tests for GET /health endpoint."""

from __future__ import annotations

import requests

from conftest import SERVICE_URL, skip_no_service


@skip_no_service
class TestHealth:
    def test_health_returns_200(self):
        """Health endpoint returns 200 with expected fields."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "model" in data
        assert "dtype" in data
        assert "loaded" in data
        assert "sessions" in data

    def test_health_reports_loaded_models(self):
        """Health reports which models are loaded."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        loaded = data["loaded"]
        assert "text" in loaded
        assert "tracker" in loaded
        assert "video" in loaded
        assert all(isinstance(v, bool) for v in loaded.values())

    def test_health_reports_sessions(self):
        """Health reports session count and limit."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        sessions = data["sessions"]
        assert "active" in sessions
        assert "max" in sessions
        assert isinstance(sessions["active"], int)
        assert sessions["active"] >= 0

    def test_health_reports_sam3_1_model(self):
        """Health endpoint reports SAM3.1 model name."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        assert "facebook/sam3.1" in data.get("model", "")
