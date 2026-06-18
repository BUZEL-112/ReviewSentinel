# ADR 002: Prefect over Airflow and Celery

**Date:** June 2026  
**Status:** Accepted

## Context

ReviewSentinel requires an orchestration engine to run background processes:
1. **Training Flow:** A heavy, multi-hour pipeline that runs weekly or when triggered by drift.
2. **Monitoring Flow:** A lightweight job running every Monday to compute data drift.
3. **LLM Judge Flow:** A recurring job (every 4 hours) that processes a queue of uncertain predictions.

We evaluated three primary orchestration tools:
- **Apache Airflow:** The industry standard for data orchestration.
- **Celery (+ Celery Beat):** The standard Python background task queue.
- **Prefect (2.x):** A modern, Python-native orchestrator.

## Decision

We chose **Prefect**.

## Rationale

1. **Python Native Execution:** Unlike Airflow, which requires DAGs to be written in a specific declarative style and heavily penalises passing data between tasks, Prefect allows us to use native Python `def` functions. We can pass pandas DataFrames directly between Prefect `@task`s without writing them to intermediate storage.
2. **Dynamic Workflows:** The training flow contains a Quality Gate. If the new model fails the gate, the flow must halt. Prefect's native Python `if/else` control flow handles this trivially.
3. **State Observability vs. Celery:** While Celery is great for firing off thousands of independent tasks, it lacks a native UI for visualizing a multi-step pipeline (Ingest → Clean → Train → Evaluate). Prefect provides a central UI for pipeline observability.
4. **Local Development:** Prefect 2.x doesn't require a heavy database setup just to run a script locally. A developer can run `python src/orchestration/flows.py` and it executes locally, while still reporting to the tracking server if available.

## Consequences

- **Positive:** Developer experience is excellent. Flow code looks like standard Python code.
- **Negative:** We must maintain a Prefect Server and Prefect Worker container in our deployment topology.
- **Negative:** Prefect's background scheduler relies on the worker process remaining alive; if the worker crashes silently, scheduled flows will miss their triggers. We rely on Kubernetes/Docker restart policies to mitigate this.
