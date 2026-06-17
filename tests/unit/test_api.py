"""
tests/unit/test_api.py

Unit tests for the FastAPI application (src/api/api.py).

Strategy:
- TestClient runs the app synchronously without starting a real HTTP server.
- _initialize_pipeline is patched in the lifespan so no model is loaded from disk.
- The global `pipeline` module-level variable is patched with a MagicMock whose
  .run() returns a controlled one-row DataFrame.
- Tests that need different pipeline responses patch `pipeline` inline.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_mock(
    label: str = "positive",
    confidence: float = 0.94,
    text: str = "Works perfectly.",
) -> MagicMock:
    """Return a MagicMock pipeline whose .run() yields a single-row DataFrame."""
    mock = MagicMock()
    mock.batch_separator = "|||"
    mock.run.return_value = pd.DataFrame([{
        "text": text,
        "label": label,
        "confidence": confidence,
        "scores": {
            "negative": round(1 - confidence, 4),
            "neutral": 0.02,
            "positive": confidence,
        },
        "aspect": None,
    }])
    return mock


# ---------------------------------------------------------------------------
# Module-scoped client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    One TestClient per module — created after patching _initialize_pipeline
    so the lifespan context manager does not attempt to load model weights.
    """
    mock_pipe = _make_pipeline_mock()

    with patch("src.api.api._initialize_pipeline", return_value=True), \
         patch("src.api.api.pipeline", mock_pipe), \
         patch("src.api.api.prediction_logger") as mock_logger:
        mock_logger.log_prediction.return_value = None
        from src.api.api import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    data = client.get("/health").json()
    for field in ("status", "model_loaded", "model_version", "uptime_seconds", "timestamp"):
        assert field in data, f"Missing field in health response: {field}"


def test_test_endpoint_returns_200(client):
    response = client.get("/test")
    assert response.status_code == 200


def test_test_endpoint_has_message(client):
    data = client.get("/test").json()
    assert "message" in data


# ---------------------------------------------------------------------------
# Single predict — request validation
# ---------------------------------------------------------------------------

def test_predict_missing_title_returns_422(client):
    """title is required — omitting it should trigger Pydantic 422."""
    response = client.post("/predict", json={"text": "No title"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Single predict — happy path
# ---------------------------------------------------------------------------

def test_predict_valid_request_returns_200(client):
    mock_pipe = _make_pipeline_mock(label="positive", confidence=0.94)
    with patch("src.api.api.pipeline", mock_pipe), \
         patch("src.api.api.startup_time", 0.0):
        response = client.post(
            "/predict",
            json={"title": "Great purchase", "text": "Arrived on time and works perfectly."},
        )
    assert response.status_code == 200


def test_predict_response_has_required_fields(client):
    mock_pipe = _make_pipeline_mock()
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post(
            "/predict",
            json={"title": "Great purchase", "text": "Works perfectly."},
        ).json()
    for field in ("label", "confidence", "scores", "flags", "processing_time_ms", "model_version"):
        assert field in body, f"Missing field: {field}"


def test_predict_label_is_valid_sentiment(client):
    mock_pipe = _make_pipeline_mock(label="positive")
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post(
            "/predict",
            json={"title": "Good", "text": "Works well."},
        ).json()
    assert body["label"] in ("positive", "neutral", "negative")


def test_predict_scores_has_three_classes(client):
    mock_pipe = _make_pipeline_mock()
    with patch("src.api.api.pipeline", mock_pipe):
        scores = client.post(
            "/predict",
            json={"title": "Good", "text": "Works well."},
        ).json()["scores"]
    assert set(scores.keys()) == {"negative", "neutral", "positive"}


# ---------------------------------------------------------------------------
# Single predict — flag detection
# ---------------------------------------------------------------------------

def test_predict_short_text_sets_short_review_flag(client):
    """text=None means the cleaned text will be just the title (<10 chars)."""
    mock_pipe = _make_pipeline_mock(text="Bad", confidence=0.88)
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post("/predict", json={"title": "Bad", "text": None}).json()
    assert body["flags"]["short_review"] is True


def test_predict_high_confidence_does_not_set_low_confidence_flag(client):
    mock_pipe = _make_pipeline_mock(confidence=0.94, text="Great product overall.")
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post("/predict", json={"title": "Great", "text": "Great product overall."}).json()
    assert body["flags"]["low_confidence"] is False


def test_predict_low_confidence_sets_flag(client):
    mock_pipe = _make_pipeline_mock(confidence=0.50, text="This product is okay I guess.")
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post(
            "/predict",
            json={"title": "Meh", "text": "This product is okay I guess."},
        ).json()
    assert body["flags"]["low_confidence"] is True


# ---------------------------------------------------------------------------
# Batch submission (async job queue)
# ---------------------------------------------------------------------------

def test_batch_submit_returns_202(client):
    mock_pipe = _make_pipeline_mock()
    with patch("src.api.api.pipeline", mock_pipe):
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"title": "Good", "text": "Works great"},
                    {"title": "Bad", "text": "Broke immediately"},
                ]
            },
        )
    assert response.status_code == 202


def test_batch_submit_response_has_job_id(client):
    mock_pipe = _make_pipeline_mock()
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post(
            "/predict/batch",
            json={"items": [{"title": "Test", "text": "body"}]},
        ).json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["item_count"] == 1


def test_batch_submit_correct_item_count(client):
    mock_pipe = _make_pipeline_mock()
    with patch("src.api.api.pipeline", mock_pipe):
        body = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"title": "A", "text": "one"},
                    {"title": "B", "text": "two"},
                    {"title": "C", "text": "three"},
                ]
            },
        ).json()
    assert body["item_count"] == 3


# ---------------------------------------------------------------------------
# Batch polling — not found
# ---------------------------------------------------------------------------

def test_batch_poll_unknown_job_returns_404(client):
    response = client.get("/predict/batch/nonexistent-job-id-xyz")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200
