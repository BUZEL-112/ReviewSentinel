"""
tests/unit/test_drift_thresholds.py

Unit tests for evaluate_drift_task (src/orchestration/monitoring_tasks.py).

The task is decorated with @task from Prefect. We call the underlying function
via `.fn` so no Prefect server or event loop is required.

DriftResult is imported from src/monitoring/drift_monitor.py.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.orchestration.monitoring_tasks import evaluate_drift_task, DriftAction


# ---------------------------------------------------------------------------
# Minimal DriftResult stub
# ---------------------------------------------------------------------------

def make_drift_result(fraction: float) -> MagicMock:
    """Create a minimal DriftResult-like object with a drift_fraction attribute."""
    result = MagicMock()
    result.drift_fraction = fraction
    result.summary = f"Drift fraction: {fraction:.2f}"
    result.mlflow_run_id = "test-run-id"
    result.report_path = "artifacts/drift/report.html"
    return result


# ---------------------------------------------------------------------------
# Boundary definitions (must match monitoring_tasks.py defaults)
# ---------------------------------------------------------------------------
ALERT_THRESH  = 0.30
RETRAIN_THRESH = 0.50


# ---------------------------------------------------------------------------
# None / missing drift result
# ---------------------------------------------------------------------------

def test_none_drift_result_returns_none_action():
    """A None drift_result should return DriftAction.NONE without crashing."""
    config = {"monitoring": {"drift": {"alert_threshold": ALERT_THRESH, "retrain_threshold": RETRAIN_THRESH}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(None, config_path="configs/pipeline_params.yaml")
    assert result == DriftAction.NONE


# ---------------------------------------------------------------------------
# Below alert threshold → NONE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [0.0, 0.10, 0.29])
def test_below_alert_threshold_returns_none(fraction):
    config = {"monitoring": {"drift": {"alert_threshold": ALERT_THRESH, "retrain_threshold": RETRAIN_THRESH}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(fraction),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.NONE


# ---------------------------------------------------------------------------
# At / above alert threshold but below retrain → ALERT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [0.30, 0.35, 0.49])
def test_at_or_above_alert_threshold_returns_alert(fraction):
    config = {"monitoring": {"drift": {"alert_threshold": ALERT_THRESH, "retrain_threshold": RETRAIN_THRESH}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(fraction),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.ALERT


# ---------------------------------------------------------------------------
# At / above retrain threshold → TRIGGER_RETRAINING
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction", [0.50, 0.75, 1.0])
def test_at_or_above_retrain_threshold_triggers_retraining(fraction):
    config = {"monitoring": {"drift": {"alert_threshold": ALERT_THRESH, "retrain_threshold": RETRAIN_THRESH}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(fraction),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.TRIGGER_RETRAINING


# ---------------------------------------------------------------------------
# Exact boundary values
# ---------------------------------------------------------------------------

def test_exactly_at_alert_threshold_is_alert():
    """0.30 is the alert threshold — should return ALERT (>=)."""
    config = {"monitoring": {"drift": {"alert_threshold": 0.30, "retrain_threshold": 0.50}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(0.30),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.ALERT


def test_just_below_alert_threshold_is_none():
    """0.299... is just below — should return NONE."""
    config = {"monitoring": {"drift": {"alert_threshold": 0.30, "retrain_threshold": 0.50}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(0.2999),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.NONE


def test_exactly_at_retrain_threshold_triggers_retraining():
    """0.50 is the retrain threshold — should return TRIGGER_RETRAINING (>=)."""
    config = {"monitoring": {"drift": {"alert_threshold": 0.30, "retrain_threshold": 0.50}}}
    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=config):
        result = evaluate_drift_task.fn(
            make_drift_result(0.50),
            config_path="configs/pipeline_params.yaml",
        )
    assert result == DriftAction.TRIGGER_RETRAINING
