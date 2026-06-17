"""
tests/smoke/test_api_smoke.py

Smoke test: "does the API start and return 200 on /health?"

This test is intentionally fragile to environment issues — that is its job.
No mocks are used. If model artifacts exist, model_loaded will be True.
If they don't exist, model_loaded will be False but /health should still return 200
(graceful degradation path).

Run with: pytest tests/smoke/ -v
"""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_health_endpoint_responds():
    """
    Import the real app and hit /health with zero mocking.
    The pipeline may or may not load depending on whether model artifacts exist.
    What matters is: the app starts, the endpoint responds, and the schema is present.
    """
    from src.api.api import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == 200, (
        f"Expected 200 from /health but got {response.status_code}. "
        f"Response body: {response.text}"
    )


def test_health_response_has_expected_keys():
    """Schema regression: all required fields must be present."""
    from src.api.api import app

    with TestClient(app, raise_server_exceptions=False) as client:
        data = client.get("/health").json()

    for field in ("status", "model_loaded", "model_version", "uptime_seconds", "timestamp"):
        assert field in data, f"Missing key '{field}' in /health response"


def test_test_endpoint_is_reachable():
    """/test is the lightest possible reachability check."""
    from src.api.api import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/test")

    assert response.status_code == 200
