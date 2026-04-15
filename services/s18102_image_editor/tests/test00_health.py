"""Image Editor service tests — GET /health."""

import requests

from conftest import SERVICE_URL, skip_no_service


@skip_no_service
class TestHealth:
    def test_health_returns_200(self):
        """Health endpoint returns 200."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        assert resp.status_code == 200

    def test_health_reports_downstream(self):
        """Health response includes downstream service status."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        assert "status" in data
        assert "flux_nim" in data
        assert "sam3" in data

    def test_health_status_values(self):
        """Health status is 'ok' or 'degraded'."""
        resp = requests.get(f"{SERVICE_URL}/health", timeout=10)
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
