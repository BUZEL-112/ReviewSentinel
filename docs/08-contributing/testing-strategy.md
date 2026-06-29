# Testing Strategy

We maintain three distinct test suites to ensure the reliability of the ReviewSentinel pipeline and API.

## 1. Unit Tests (`tests/unit/`)
Unit tests guard the core logic and edge cases of individual functions and classes. They run entirely locally, mocking out external dependencies like MLflow, Prefect runtimes, or database connections.

**Key areas covered:**
- **`test_quality_gate.py`**: Calls `quality_gate_task.fn` directly with a patched `mlflow.search_runs`. It covers the regression case (new F1 < baseline), the exact-boundary case (improvement == threshold), and the first-run auto-deploy flag. This guards against accidentally loosening the gate.
- **`test_drift_thresholds.py`**: Calls `evaluate_drift_task.fn` with stubbed `DriftResult` values. It catches changes to threshold evaluation that would cause the flow to silently skip retraining when drift occurs.

**Command:** `pytest tests/unit/ -v` (Run this before every commit!)

## 2. Integration Tests (`tests/integration/`)
Integration tests verify the end-to-end flows. They test the actual handoffs between components: (Train → Infer → Judge). These tests spin up local test databases or rely on a dedicated local test configuration.

**Command:** `pytest tests/integration/ -v`

## 3. Smoke Tests (`tests/smoke/`)
Smoke tests verify the deployed infrastructure. They run against a live system (either Docker Compose or a local Kubernetes deployment) and send actual HTTP requests to the API endpoints to ensure everything is wired correctly.

**Command:** `pytest tests/smoke/ -v`
