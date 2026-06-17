"""
tests/conftest.py — Shared fixtures for the entire ReviewSentinel test suite.

Design decisions:
- session-scoped fixtures (tiny_df, minimal_config, mock_tokenizer) are created
  once per pytest run, keeping CI fast.
- function-scoped fixtures (qm, conflict_logger) give every test a fresh SQLite
  :memory: database so there is zero bleed between tests.
- bert-tiny is used in place of distilbert-base-uncased for all tokenizer /
  model fixtures: architecturally identical, weighs almost nothing, no GPU needed.
"""

import pytest
import torch
import pandas as pd
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from src.llm_judge import QueueEntry
from src.llm_judge.queue_manager import QueueManager
from src.llm_judge.conflict_logger import ConflictLogger


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_df():
    """
    Minimal review DataFrame matching the ingestion schema:
    columns = [rating, title, text] with a balanced class distribution
    (ratings 1–5, one of each) so it passes the DataValidator's imbalance check.
    """
    return pd.DataFrame({
        "rating": [5, 1, 3, 2, 4],
        "title":  ["Great!", "Terrible", "Okay", "Bad packaging", "Works well"],
        "text": [
            "Loved it, would buy again.",
            "Fell apart after one use.",
            "Nothing special, does the job.",
            "Arrived crushed.",
            "Solid product for the price.",
        ],
    })


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minimal_config():
    """
    Minimal pipeline config dict that mirrors the shape of pipeline_params.yaml.
    All external services point to localhost; SQLite queue uses :memory:.
    Never reads YAML from disk — safe to use in any CI environment.
    """
    return {
        "clean_data_bert": {
            "model_name": "prajjwal1/bert-tiny",
            "max_len": 64,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test",
        },
        "llm_judge": {
            "confidence_window": {"lower": 0.40, "upper": 0.60},
            "ollama": {
                "base_url": "http://localhost:11434",
                "model_name": "mistral",
            },
            "queue": {
                "db_path": ":memory:",
                "batch_size": 10,
                "max_age_minutes": 5,
            },
        },
        "orchestration": {
            "quality_gate": {
                "min_f1_improvement": 0.01,
                "baseline_metric": "f1",
                "first_run_auto_deploy": True,
            },
            "validation": {
                "max_null_ratio": 0.10,
                "max_class_imbalance": 0.80,
                "min_row_count": 3,
            },
        },
        "monitoring": {
            "drift": {
                "alert_threshold": 0.30,
                "retrain_threshold": 0.50,
            }
        },
    }


# ---------------------------------------------------------------------------
# Model / tokenizer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_tokenizer():
    """
    Real bert-tiny tokenizer (session-scoped so the download happens once).
    Used wherever tests need a functioning tokenizer without full DistilBERT.
    """
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


@pytest.fixture
def mock_model():
    """
    Fake model that returns deterministic logits without loading any weights.
    Logits are biased toward class 2 (positive) so label assertions are stable.
    Shape: [batch=1, num_labels=3].
    """
    model = MagicMock()
    model.return_value.logits = torch.tensor([[0.1, 0.2, 2.5]])
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qm():
    """
    Fresh QueueManager backed by SQLite :memory: for each test function.
    No disk I/O, no teardown logic, safe for parallel test execution.
    """
    return QueueManager(db_path=":memory:")


@pytest.fixture
def conflict_logger():
    """
    Fresh ConflictLogger backed by SQLite :memory: for each test function.
    """
    return ConflictLogger(db_path=":memory:")


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_queue_entry(
    confidence: float = 0.51,
    prediction: str = "positive",
    entry_id: str | None = None,
) -> QueueEntry:
    """
    Factory helper — not a fixture so callers can create multiple distinct
    entries inside a single test without needing parametrize.
    """
    import uuid
    from datetime import datetime, timezone

    return QueueEntry(
        entry_id=entry_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_text="The screen cracked on day two.",
        raw_title="Broke quickly",
        raw_text="The screen cracked on day two.",
        model_prediction=prediction,
        model_confidence=confidence,
        model_probabilities={
            "negative": round(1 - confidence, 4),
            "neutral": 0.0,
            "positive": confidence,
        },
        model_version="test-v1",
        status="pending",
    )
