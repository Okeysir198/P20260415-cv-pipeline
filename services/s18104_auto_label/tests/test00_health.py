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
        assert "status" in data
        assert "sam3" in data
        assert "active_jobs" in data
        assert "active_video_sessions" in data

    def test_health_sam3_status(self):
        """Health reports SAM3 connectivity."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        assert isinstance(data["sam3"], str)
        assert len(data["sam3"]) > 0

    def test_health_overall_status(self):
        """Overall status is 'ok' when SAM3 is reachable, 'degraded' otherwise."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        if data["sam3"] == "ok":
            assert data["status"] == "ok"
        else:
            assert data["status"] == "degraded"
