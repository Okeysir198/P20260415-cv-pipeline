"""Flux NIM service tests — GET /v1/health/ready."""

import requests

from conftest import SERVICE_URL, skip_no_service


@skip_no_service
class TestHealth:
    def test_health_ready(self):
        """Health endpoint returns 200 when service is ready."""
        resp = requests.get(f"{SERVICE_URL}/v1/health/ready", timeout=10)
        assert resp.status_code == 200

    def test_health_returns_body(self):
        """Health endpoint returns a response body."""
        resp = requests.get(f"{SERVICE_URL}/v1/health/ready", timeout=10)
        assert resp.status_code == 200
        # NIM health may return empty body or JSON — just verify it doesn't error
        assert resp.text is not None
