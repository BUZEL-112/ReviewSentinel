# Getting Started

This guide covers getting ReviewSentinel running on your local machine using Docker Compose. It is intended for developers who need to run the API, trigger training flows, or explore the dashboards locally.

If you are looking to deploy to a Kubernetes cluster, see [Deploying to Kind](../operations/deploying-to-kind.md).

---

## 1. Prerequisites

You must have the following tools installed on your local machine:

| Tool | Minimum Version | Verify Command |
|------|-----------------|----------------|
| **Git** | 2.x | `git --version` |
| **Docker** | 24.x | `docker --version` |
| **Docker Compose** | 2.x | `docker-compose version` |

*(Optional, for Kubernetes deployments only)*
- `kubectl` 1.28+
- `kind` 0.20+
- `make`

---

## 2. Clone and Configure

Clone the repository and set up your local environment variables.

```bash
# Clone the repository
git clone https://github.com/BUZEL-112/ReviewSentinel.git
cd ReviewSentinel

# Create your local environment file
cp .env.example .env
```

Open `.env` in your text editor. At a minimum, you must set secure values for the MinIO credentials. Do not use default credentials on any system connected to the internet.

```ini
# Edit these in .env
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=your_secure_password
```

---

## 3. Start the Stack

ReviewSentinel runs entirely within Docker Compose for local development. This includes the FastAPI application, Prefect orchestration, MLflow tracking, MinIO artifact storage, and an Ollama LLM node.

```bash
# Build images and start all services in detached mode
docker-compose -f docker/docker-compose.yaml up --build -d
```

You can view the startup logs to monitor progress:
```bash
docker-compose -f docker/docker-compose.yaml logs -f
```

Wait until you see messages indicating that `api`, `mlflow`, and `prefect-server` are running. Note that MinIO may restart once or twice as its initialization job runs (this is normal; see [Troubleshooting](../operations/troubleshooting.md)).

---

## 4. Verify Services

Once the stack is up, verify that the core services are responding.

### API Health
```bash
curl http://localhost:8000/health
```
*Expected output: A JSON object with `"status": "healthy"` or `"status": "unhealthy"` depending on whether a model has been trained yet.*

### Dashboards
Open these in your browser:
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow:** [http://localhost:5000](http://localhost:5000)
- **Prefect:** [http://localhost:4200](http://localhost:4200)

---

## 5. Make a Prediction

The API is ready immediately, but if you have not run a training job, it will not have a model loaded. (It returns a 503 error instructing you to wait for the self-healing training job to complete).

If a model is loaded, you can test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Broke after one week",
    "text": "Complete waste of money. The battery died and it stopped charging."
  }'
```

*Expected output: A JSON object containing `"label": "negative"` and confidence scores.*

---

## Next Steps

- **Want to train a model?** Read the [Local Docker Guide](local-docker.md) to learn how to trigger the Prefect training flow.
- **Integrating the API?** Read the full [API Reference](../api/reference.md).
- **Want to contribute?** Read the [Contributing Guide](contributing.md).
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
