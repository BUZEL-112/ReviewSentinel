# Local Docker Guide

This document provides detailed operational instructions for running ReviewSentinel locally using Docker Compose. 

If you are just getting started, see [Getting Started](getting-started.md) for the initial setup steps.

---

## Service Architecture

When you run `docker-compose -f docker/docker-compose.yaml up`, Docker Compose brings up the following inter-connected services on your machine:

| Service Name | Port | Description |
|--------------|------|-------------|
| **nginx** | `80` (HTTP) | Reverse proxy routing `/api/*` to FastAPI. Acts as the main entrypoint. |
| **api** | `8000` | The FastAPI application serving the prediction endpoints. |
| **mlflow** | `5000` | MLflow tracking server for recording metrics, parameters, and model weights. |
| **prefect-server** | `4200` | The Prefect orchestration UI and API. |
| **prefect-worker** | *(No port)* | A background process that executes flows (training, monitoring, LLM Judge). |
| **minio** | `9001` (Console) | S3-compatible object storage. Stores MLflow model artifacts. |
| **ollama** | `11434` | Local LLM engine hosting the Mistral 7B judge model. |

---

## Triggering Flows Locally

The `prefect-worker` container is responsible for running all background tasks. By default, it listens for scheduled flows. However, during local development, you will frequently want to trigger flows manually.

### Triggering the Training Flow

To run the complete data ingestion, preprocessing, training, and deployment pipeline:

```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py
```

### Triggering the Drift Monitor

To manually run the Evidently AI drift analysis against the prediction logs:

```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py monitor
```

### Triggering the LLM Judge

To manually force the LLM Judge to process any pending items in the SQLite queue:

```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py judge
```

> [!TIP]
> Why run these *inside* the worker container instead of on your host? Inside the container, DNS names like `http://mlflow:5000` resolve correctly. If you run them on your host, the scripts will attempt to reach `http://mlflow:5000` and fail, triggering a fallback to `localhost` which may or may not succeed depending on your environment variables.

---

## Hot-Reloading for Development

If you are actively editing the Python code in `src/` and want the API to reload automatically when you save a file, you should use the development override file.

```bash
# Stop any running instances
docker-compose -f docker/docker-compose.yaml down

# Start with the dev override
docker-compose -f docker/docker-compose.yaml -f docker/docker-compose.dev.yaml up --build
```

The dev override typically passes the `--reload` flag to Uvicorn and bind-mounts your local `src/` directory into the container.

---

## Teardown and Reset

### Safe Stop (Preserve Data)

To stop the services while preserving your databases, prediction logs, and MinIO artifacts:

```bash
docker-compose -f docker/docker-compose.yaml down
```

### Hard Reset (Delete All Data)

If you have corrupted your MLflow database or want to simulate a fresh install:

```bash
# WARNING: This deletes all Docker named volumes. 
# You will lose your MLflow history and MinIO artifacts.
docker-compose -f docker/docker-compose.yaml down -v

# You may also want to delete host-mounted files:
rm -rf artifacts/models/* artifacts/evaluation/* artifacts/monitoring/reports/*
```

---

## Troubleshooting Docker

See [Troubleshooting](../operations/troubleshooting.md) for full details on common Docker issues, including:
- The MinIO startup race condition
- Prefect worker deployment failures
- Ollama first-boot timeouts
- `PYTHONPATH` resolution errors
