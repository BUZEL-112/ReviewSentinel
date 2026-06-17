"""
tests/unit/test_quality_gate.py

Unit tests for quality_gate_task (src/orchestration/flows.py).

Prefect tasks expose the underlying Python function via `.fn`, which lets us
call it directly without a running Prefect server or event loop.

MLflow's search_runs is patched to return a controlled DataFrame (or empty
DataFrame) for each test scenario.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _config(
    min_improvement: float = 0.01,
    first_run_auto_deploy: bool = True,
) -> dict:
    return {
        "orchestration": {
            "quality_gate": {
                "min_f1_improvement": min_improvement,
                "baseline_metric": "f1",
                "first_run_auto_deploy": first_run_auto_deploy,
            }
        },
        "mlflow": {"experiment_name": "test"},
    }


def _baseline_runs(f1: float) -> pd.DataFrame:
    """Simulate the DataFrame returned by mlflow.search_runs for a run with a given F1."""
    return pd.DataFrame([{
        "run_id": "abc123",
        "metrics.eval_f1": f1,
    }])


def run_gate(new_f1: float, baseline_f1: float | None, **config_kwargs) -> bool:
    """
    Execute quality_gate_task.fn with controlled config and MLflow data.

    Args:
        new_f1:       The F1 score from the new training run.
        baseline_f1:  The F1 score from the production baseline (or None for first run).
        **config_kwargs: Forwarded to _config().
    """
    from src.orchestration.flows import quality_gate_task

    cfg = _config(**config_kwargs)

    runs_df = pd.DataFrame() if baseline_f1 is None else _baseline_runs(baseline_f1)

    with patch("src.orchestration.flows.mlflow") as mock_mlflow, \
         patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=cfg):
        mock_mlflow.search_runs.return_value = runs_df
        mock_mlflow.set_experiment.return_value = None
        return quality_gate_task.fn({"f1": new_f1}, config_path="configs/pipeline_params.yaml")


# ---------------------------------------------------------------------------
# First-run (no baseline) scenarios
# ---------------------------------------------------------------------------

def test_first_run_auto_deploy_enabled_returns_true():
    assert run_gate(new_f1=0.72, baseline_f1=None, first_run_auto_deploy=True) is True


def test_first_run_auto_deploy_disabled_returns_false():
    assert run_gate(new_f1=0.72, baseline_f1=None, first_run_auto_deploy=False) is False


# ---------------------------------------------------------------------------
# Improvement scenarios
# ---------------------------------------------------------------------------

def test_sufficient_improvement_passes():
    """0.02 improvement > 0.01 threshold — gate should open."""
    assert run_gate(new_f1=0.80, baseline_f1=0.78) is True


def test_insufficient_improvement_fails():
    """0.005 improvement < 0.01 threshold — gate should reject."""
    assert run_gate(new_f1=0.785, baseline_f1=0.78) is False


def test_regression_fails():
    """New model is worse than baseline — gate must reject."""
    assert run_gate(new_f1=0.70, baseline_f1=0.80) is False


def test_exact_threshold_passes():
    """Improvement == threshold (0.01) should pass (>= comparison)."""
    assert run_gate(new_f1=0.79, baseline_f1=0.78, min_improvement=0.01) is True


def test_just_below_threshold_fails():
    """Improvement 0.009 < 0.01 threshold — gate should reject."""
    assert run_gate(new_f1=0.7899, baseline_f1=0.78, min_improvement=0.01) is False


# ---------------------------------------------------------------------------
# Custom threshold scenarios
# ---------------------------------------------------------------------------

def test_large_improvement_custom_threshold():
    """0.05 improvement with 0.05 threshold — exactly at boundary, should pass."""
    assert run_gate(new_f1=0.85, baseline_f1=0.80, min_improvement=0.05) is True


def test_zero_threshold_always_passes_if_no_regression():
    """A zero min_improvement threshold — any non-negative change should pass."""
    assert run_gate(new_f1=0.80, baseline_f1=0.80, min_improvement=0.0) is True


# ---------------------------------------------------------------------------
# No-improvement edge cases
# ---------------------------------------------------------------------------

def test_identical_scores_at_nonzero_threshold_fails():
    """Same score as baseline with a 0.01 threshold — should fail."""
    assert run_gate(new_f1=0.80, baseline_f1=0.80, min_improvement=0.01) is False
