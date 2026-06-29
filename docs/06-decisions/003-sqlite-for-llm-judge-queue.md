# ADR 003: SQLite for LLM Judge Queue

**Date:** June 2026  
**Status:** Accepted

## Context

When the primary DistilBERT model returns a prediction with a confidence score between 0.40 and 0.60, the prediction is considered "uncertain." We want to send these uncertain predictions to a larger, more capable LLM (Mistral 7B) for a "second opinion."

Because LLM inference is slow (seconds per request), we cannot block the main FastAPI prediction response while waiting for the LLM. We must decouple the process: the API must queue the review, and a background worker must dequeue and process it asynchronously.

We evaluated queue technologies:
1. **Redis / RabbitMQ:** Standard message brokers.
2. **PostgreSQL:** Using row-level locking (e.g., `FOR UPDATE SKIP LOCKED`).
3. **In-memory Python `queue.Queue`:** Fast, but volatile.
4. **SQLite:** A file-based SQL database.

## Decision

We chose **SQLite** for both the queue (`artifacts/llm_judge/review_queue.db`) and the resulting conflict log (`artifacts/llm_judge/conflicts.db`).

## Rationale

1. **Zero External Dependencies:** Redis or RabbitMQ would require adding another stateful container to our infrastructure. Since we already have MLflow, MinIO, and Prefect, we wanted to minimize operational surface area.
2. **Durability:** An in-memory queue would lose pending reviews if the API container restarted. SQLite provides durability to disk.
3. **Concurrency Profile:** The queue has a single writer (the FastAPI process) and a single consumer (the Prefect scheduled worker that runs every 4 hours). SQLite handles single-writer / single-reader concurrency perfectly well with WAL (Write-Ahead Logging) enabled.
4. **Volume:** We expect < 10% of traffic to hit the uncertainty window. At a volume of a few thousand records a day, SQLite performance is indistinguishable from PostgreSQL.

## Consequences

- **Positive:** Operations are trivial. If the queue breaks, you can inspect it by simply opening the `.db` file with any SQLite viewer.
- **Negative:** If we scale the FastAPI application horizontally to multiple pods, SQLite's single-writer lock could become a bottleneck under heavy load. If the system scales significantly, this queue should be migrated to PostgreSQL or Redis.
