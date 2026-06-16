# ReviewSentinel

A production-grade MLOps platform for real-time customer review sentiment analysis. ReviewSentinel combines a fine-tuned DistilBERT classifier with an end-to-end operational stack — orchestrated training, automated drift monitoring, LLM-powered active learning, semantic search, and self-healing infrastructure — deployable locally via Docker Compose or Kubernetes (Kind).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Running with Docker Compose](#running-with-docker-compose)
  - [Running with Kubernetes (Kind)](#running-with-kubernetes-kind)
- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Drift Monitoring](#drift-monitoring)
- [Training Pipeline](#training-pipeline)
  - [Pipeline Stages](#pipeline-stages)
  - [Quality Gate](#quality-gate)
  - [Triggering a Training Run](#triggering-a-training-run)
- [Explainability (SHAP)](#explainability-shap)
- [Semantic Search](#semantic-search)
- [Drift Monitoring](#drift-monitoring-1)
- [LLM Judge & Active Learning](#llm-judge--active-learning)
- [Self-Healing Watchdog](#self-healing-watchdog)
- [Configuration Reference](#configuration-reference)
- [Makefile Reference](#makefile-reference)
- [License](#license)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        NGINX (Reverse Proxy)                 │
│                         :80 / :30080                         │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
       ┌───────▼───────┐               ┌──────▼──────┐
       │   FastAPI      │               │   MLflow    │
       │   :8000        │               │   :5000     │
       │ ┌────────────┐ │               └─────────────┘
       │ │DistilBERT  │ │                     │
       │ │Inference   │ │               ┌─────▼─────┐
       │ ├────────────┤ │               │   MinIO    │
       │ │SHAP        │ │               │   :9000    │
       │ │Explainer   │ │               │  (S3 Store)│
       │ ├────────────┤ │               └───────────-┘
       │ │Semantic    │ │
       │ │Search      │ │               ┌───────────┐
       │ │(FAISS)     │ │               │ Prefect   │
       │ ├────────────┤ │               │ Server    │
       │ │LLM Judge   │◄├──────────────►│ :4200     │
       │ │Queue       │ │               ├───────────┤
       │ └────────────┘ │               │ Prefect   │
       └───────────────-┘               │ Worker    │
                                        └─────┬─────┘
                                              │
                                        ┌─────▼─────┐
                                        │  Ollama    │
                                        │  :11434    │
                                        │ (Mistral)  │
                                        └───────────-┘
```

---

## Key Features

| Feature | Description |
|---|---|
| **DistilBERT Sentiment Classifier** | Three-class (negative / neutral / positive) fine-tuned transformer with softmax confidence scores |
| **Async Inference API** | FastAPI with thread-pool executor, batch job queue, and Prometheus metrics |
| **SHAP Explainability** | Per-token attribution explaining *why* a prediction was made |
| **Semantic Search** | FAISS-powered nearest-neighbor retrieval with sentiment alignment signals |
| **Drift Monitoring** | Evidently AI statistical tests with automated alert/retrain thresholds |
| **LLM Judge (Active Learning)** | Low-confidence predictions are re-evaluated by a local Mistral 7B via Ollama |
| **Prefect Orchestration** | Scheduled training, drift monitoring, LLM judge, and watchdog flows |
| **Quality Gate** | New models must beat the production F1 baseline before deployment |
| **Self-Healing Watchdog** | Polls API health; triggers retraining if the model is missing |
| **MLflow Experiment Tracking** | Parameters, metrics, and artifacts logged per run with production tagging |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | `distilbert-base-uncased` via HuggingFace Transformers |
| Inference API | FastAPI + Uvicorn |
| Explainability | SHAP (Partition Explainer) |
| Semantic Search | `all-MiniLM-L6-v2` + FAISS (`IndexFlatIP`) |
| LLM Judge | Ollama (Mistral 7B) |
| Orchestration | Prefect 2.x |
| Drift Detection | Evidently AI |
| Experiment Tracking | MLflow |
| Object Storage | MinIO (S3-compatible) |
| Observability | Prometheus client (`/metrics` endpoint) |
| Containerization | Docker / Docker Compose |
| Kubernetes | Kind (local) |
| Reverse Proxy | NGINX |
| Language | Python 3.10+ |

---

## Project Structure

```
ReviewSentinel/
├── configs/
│   ├── config.yaml              # Core: data sources, model hyperparams, paths
│   ├── model_params.yaml        # Model architecture parameters
│   └── pipeline_params.yaml     # Orchestration hub: all pipeline/flow config
├── docker/
│   ├── api/                     # API Dockerfile
│   ├── mlflow/                  # MLflow Dockerfile
│   ├── minio/                   # MinIO init script
│   ├── nginx/                   # NGINX reverse proxy config
│   ├── prefect/                 # Prefect server + worker Dockerfile
│   ├── docker-compose.yaml      # Full local stack
│   ├── docker-compose.dev.yaml  # Dev overrides (hot-reload, debug logs)
│   └── docker-compose.prod.yaml # Production overrides
├── k8s/
│   ├── 01-base.yaml             # Secrets, ConfigMaps, PVCs
│   ├── 02-minio.yaml            # MinIO Deployment + init Job
│   ├── 03-core.yaml             # MLflow, Prefect Server, Prefect Worker
│   └── 04-app.yaml              # API, Search Index Builder, NGINX
├── src/
│   ├── api/api.py               # FastAPI application
│   ├── data/                    # Data loading and BERT preprocessing
│   ├── models/                  # Model building, training, evaluation
│   ├── pipeline/                # Train, evaluate, and inference pipelines
│   ├── orchestration/           # Prefect flows, tasks, validation, watchdog
│   ├── monitoring/              # Drift monitoring, reference store, report server
│   ├── explainability/          # SHAP explainer
│   ├── llm_judge/               # LLM judge, queue manager, conflict logger
│   ├── search/                  # Sentence encoder, FAISS indexer, searcher
│   └── utils/                   # Logger, exception handler, config parser
├── scripts/
│   ├── run_flow.py              # Manual Prefect flow trigger
│   ├── build_search_index.py    # Standalone FAISS index builder
│   └── ci/                      # CI quality gate scripts
├── kind-config.yaml             # Kind cluster configuration
├── prefect.yaml                 # Prefect deployment definitions
├── Makefile                     # Project automation targets
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment variable template
```

---

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** (v2+)
- **Python 3.10+** (for local development)
- **Kind** (for Kubernetes deployment): [Install Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
- **kubectl** (for Kubernetes deployment)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BUZEL-112/ReviewSentinel.git
   cd ReviewSentinel
   ```

2. **Create your environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Configure `.env`:**
   ```dotenv
   MINIO_ROOT_USER=reviewsentinel
   MINIO_ROOT_PASSWORD=<your-secure-password>
   MLFLOW_EXPERIMENT_NAME=sentiment_analysis_experiment
   PREFECT_API_URL=http://prefect-server:4200/api
   MODEL_DIR=artifacts/models/distilbert-finetuned/
   LOG_LEVEL=INFO
   ```

### Running with Docker Compose

Start the full stack (API, MLflow, Prefect, MinIO, Ollama, NGINX):

```bash
docker-compose -f docker/docker-compose.yaml up --build -d
```

For development with hot-reloading:

```bash
docker-compose -f docker/docker-compose.yaml -f docker/docker-compose.dev.yaml up --build
```

**Service URLs:**

| Service | URL | Description |
|---|---|---|
| NGINX (entrypoint) | `http://localhost/` | Reverse proxy |
| FastAPI (direct) | `http://localhost:8000/` | API + Swagger docs at `/docs` |
| MLflow UI | `http://localhost:5000/` | Experiment tracking |
| Prefect UI | `http://localhost:4200/` | Pipeline orchestration |
| MinIO Console | `http://localhost:9001/` | S3 bucket management |

**Trigger a training run:**

```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py
```

**Teardown:**

```bash
# Preserve data volumes
docker-compose -f docker/docker-compose.yaml down

# Full reset (deletes all data)
docker-compose -f docker/docker-compose.yaml down -v
```

### Running with Kubernetes (Kind)

The project includes a Kind cluster configuration that mounts host directories for data, artifacts, and configs.

1. **Create the cluster and deploy:**
   ```bash
   make up
   ```
   This generates the Kind config, creates a cluster, and applies all Kubernetes manifests.

2. **Watch pod status:**
   ```bash
   make logs
   ```

3. **Verify all pods are running:**
   ```bash
   kubectl get pods
   ```

   Expected output:
   ```
   NAME                              READY   STATUS      RESTARTS   AGE
   api-585b87bb5c-xxxxx              1/1     Running     0          5m
   minio-bfb4968c5-xxxxx             1/1     Running     0          5m
   minio-init-xxxxx                  0/1     Completed   0          5m
   mlflow-74c9ff95b8-xxxxx           1/1     Running     0          5m
   nginx-5578bbf586-xxxxx            1/1     Running     0          5m
   prefect-server-58c75569f7-xxxxx   1/1     Running     0          5m
   prefect-worker-7d96548f7-xxxxx    1/1     Running     0          5m
   search-index-builder-xxxxx        0/1     Completed   0          5m
   ```

4. **Access the API** via the NodePort: `http://localhost:30080/`

5. **Tear down the cluster:**
   ```bash
   make down
   ```

**Kubernetes Resources:**

| Manifest | Resources |
|---|---|
| `01-base.yaml` | Secrets, ConfigMap, PVCs (MinIO, MLflow, Prefect) |
| `02-minio.yaml` | MinIO Deployment + Service, bucket init Job |
| `03-core.yaml` | MLflow, Prefect Server, Prefect Worker |
| `04-app.yaml` | API Deployment, Search Index Builder Job, NGINX (NodePort :30080) |

---

## API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "unknown",
  "model_dir": "artifacts/models/distilbert",
  "aspect_enabled": false,
  "uptime_seconds": 124.5,
  "timestamp": "2026-06-16T20:00:00.000000+00:00"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Terrible quality",
    "text": "The packaging was broken and the product looks cheap.",
    "include_explanation": true,
    "include_similar_reviews": true,
    "similar_reviews_count": 3
  }'
```

**Response:**

```json
{
  "text": "Terrible quality The packaging was broken and the product looks cheap.",
  "label": "negative",
  "confidence": 0.98,
  "scores": {
    "negative": 0.98,
    "neutral": 0.01,
    "positive": 0.01
  },
  "aspect": null,
  "flags": {
    "short_review": false,
    "low_confidence": false,
    "possible_sarcasm": false
  },
  "processing_time_ms": 45.2,
  "model_version": "unknown",
  "explanation": {
    "predicted_class": "negative",
    "target_class_explained": "negative",
    "baseline_probability": 0.3333,
    "tokens": [
      { "token": "broken", "shap_value": 0.45, "direction": "toward" },
      { "token": "Terrible", "shap_value": 0.38, "direction": "toward" },
      { "token": "cheap", "shap_value": 0.25, "direction": "toward" }
    ]
  },
  "similar_reviews": {
    "results": [ ... ],
    "alignment_rate": 0.85,
    "alignment_signal": "STRONG_SUPPORT"
  }
}
```

**Edge-Case Flags:**

| Flag | Trigger |
|---|---|
| `short_review` | Input text < 10 characters |
| `low_confidence` | Winning class probability < 0.60 |
| `possible_sarcasm` | Heuristic pattern match (e.g., "just wonderful", "oh great") |

### Batch Prediction

Submit a batch asynchronously (returns HTTP 202):

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "items": [
      { "title": "Great product!", "text": "Exactly what I needed." },
      { "title": "Awful experience", "text": "Broke on day one." }
    ]
  }'
```

```json
{ "job_id": "abc-123", "status": "queued", "item_count": 2 }
```

Poll for results:

```bash
curl http://localhost:8000/predict/batch/abc-123
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Exposed metrics:
- `sentiment_predictions_total` — counter by endpoint and label
- `sentiment_prediction_errors_total` — counter by endpoint
- `sentiment_prediction_latency_seconds` — histogram by endpoint
- `sentiment_confidence_score` — histogram of confidence distribution
- `sentiment_batch_jobs_active` — gauge of active batch jobs

### Drift Monitoring

| Endpoint | Description |
|---|---|
| `GET /monitoring/drift/latest` | JSON metadata about the most recent drift evaluation |
| `GET /monitoring/drift/report` | Full interactive HTML Evidently drift report |

---

## Training Pipeline

The training pipeline is fully orchestrated by Prefect and runs as a DAG of retryable tasks with data validation and a production quality gate.

### Pipeline Stages

```
Load Data → Validate Data → Clean Data → Train Model → Evaluate → Quality Gate → Deploy → Rebuild Search Index
```

1. **Load Data** — Downloads and parses Amazon review data from JSONL (HuggingFace Hub streaming).
2. **Validate Data** — Great Expectations-style checks: schema, null ratios, target bounds, class imbalance.
3. **Clean Data** — BERT tokenization via `distilbert-base-uncased` with train/val/test splits.
4. **Train Model** — Fine-tunes DistilBERT for 3-class sentiment classification. Logs to MLflow.
5. **Evaluate Model** — Computes accuracy, precision, recall, and F1-score on the held-out test set.
6. **Quality Gate** — Compares new F1 against the production baseline in MLflow. Blocks deployment if the improvement threshold is not met.
7. **Deploy Model** — Tags the MLflow run as `is_production=true`, saves model artifacts, and stores a reference dataset for drift monitoring.
8. **Rebuild Search Index** — Rebuilds the FAISS semantic search index to reflect the latest training corpus.

### Quality Gate

The quality gate prevents deploying degraded models:
- Queries MLflow for the current production model's F1 score.
- Requires a configurable minimum improvement (default: 1 percentage point).
- On the first run with no baseline, auto-deployment is enabled by default.

### Triggering a Training Run

**Via Prefect (Docker Compose):**

```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py
```

**Via Prefect (Kubernetes):**

```bash
kubectl exec -it deploy/prefect-worker -- python scripts/run_flow.py
```

**Scheduled runs** are defined in `prefect.yaml`:

| Deployment | Schedule | Description |
|---|---|---|
| `weekly-training-pipeline` | `0 2 * * 1` (Mon 2 AM UTC) | Weekly retraining |
| `weekly-drift-monitoring` | `0 3 * * 1` (Mon 3 AM UTC) | Weekly drift analysis |
| `llm-judge-processing` | `0 */4 * * *` (Every 4 hours) | Process LLM judge queue |
| `proactive-health-watchdog` | Every 5 minutes | Self-healing health check |

The training pipeline also responds to `model.missing` events emitted by the API, enabling reactive self-healing.

---

## Explainability (SHAP)

When `include_explanation: true` is set on a `/predict` request, the API generates per-token SHAP attributions using a Partition Explainer. This reveals which specific words drove the classification — not just *what* the model predicted, but *why*.

Use case: product managers triage negative feedback by vocabulary clusters (e.g., "broken", "slow", "expensive") rather than just sentiment counts.

Configuration in `configs/config.yaml`:

```yaml
explainability:
  enabled: true
  max_evals: 500
  top_k_tokens: 10
```

---

## Semantic Search

FAISS-powered nearest-neighbor retrieval finds historically similar reviews and compares their known sentiment against the current prediction.

**Architecture:**
- **Encoder:** `all-MiniLM-L6-v2` (sentence-transformers) generates dense float32 embeddings.
- **Index:** `faiss.IndexFlatIP` with `IndexIDMap` for exact inner-product search.
- **Graceful degradation:** If the index is missing, the API continues to function and returns `null` for similar reviews.

**Alignment Signals:**
- `STRONG_SUPPORT` — ≥ 80% of similar past reviews share the same sentiment.
- `CONTRADICTS` — < 40% match, flagging a potential anomaly or edge case.

**Request:**

```json
{
  "title": "Battery dies too fast",
  "text": "I like the design but it doesn't last a full day.",
  "include_similar_reviews": true,
  "similar_reviews_count": 3
}
```

The FAISS index is automatically rebuilt after every successful training pipeline run.

---

## Drift Monitoring

Models degrade silently as vocabulary, user behavior, and product categories evolve. ReviewSentinel uses **Evidently AI** to detect this drift before business metrics are impacted.

**Monitored dimensions:**
- Text length distributions (character and word count)
- Vocabulary richness (out-of-vocabulary rate)
- Sentiment score drift
- Predicted label distribution shift

**Automatic thresholds:**

| Threshold | Action |
|---|---|
| > 30% of features drifted | Alert |
| > 50% of features drifted | Trigger automatic retraining |

**Schedule:** Every Monday at 3:00 AM UTC, analyzing prediction logs from the prior 7 days.

---

## LLM Judge & Active Learning

When the primary DistilBERT classifier produces a prediction with confidence in the uncertainty window (default: 0.40–0.60), the review is queued for a second opinion from a locally-hosted **Mistral 7B** model via Ollama.

**Workflow:**
1. Low-confidence predictions are enqueued to a SQLite-backed queue.
2. The `llm-judge-processing` Prefect flow dequeues and evaluates batches every 4 hours.
3. If the LLM judge **disagrees** with the model, the conflict is logged.
4. Before the next drift retraining, logged conflicts are exported and ingested into the training dataset.

This creates a continuous active learning loop: the model automatically improves on the exact edge cases it finds most difficult.

**Configuration in `configs/pipeline_params.yaml`:**

```yaml
llm_judge:
  confidence_window:
    lower: 0.40
    upper: 0.60
  ollama:
    base_url: "http://ollama:11434"
    model_name: "mistral"
    temperature: 0.1
    timeout_seconds: 30
  queue:
    db_path: "artifacts/llm_judge/review_queue.db"
    batch_size: 50
  conflicts:
    db_path: "artifacts/llm_judge/conflicts.db"
    export_on_retrain: true
```

---

## Self-Healing Watchdog

A Prefect flow (`proactive-health-watchdog`) polls the API's `/health` endpoint every 5 minutes. If the model is not loaded:

1. A time-bucketed idempotency key prevents duplicate triggers within the same hour.
2. The watchdog triggers the `weekly-training-pipeline` deployment to retrain and redeploy the model.

Additionally, the API itself emits a `model.missing` Prefect event (rate-limited to once per 5 minutes) when inference is attempted without a loaded model. The training pipeline deployment is configured with a trigger that listens for this event, providing a secondary self-healing path.

---

## Configuration Reference

| File | Purpose |
|---|---|
| `configs/config.yaml` | Data sources, preprocessing params, DistilBERT hyperparameters, MLflow, explainability |
| `configs/model_params.yaml` | Model architecture parameters |
| `configs/pipeline_params.yaml` | Orchestration hub — training, evaluation, inference, MLflow, Prefect, monitoring, LLM judge, semantic search |
| `.env` | Runtime environment variables (MinIO credentials, model dir, log level) |
| `prefect.yaml` | Prefect deployment definitions and schedules |

---

## Makefile Reference

```bash
make help     # Show available commands
make up       # Generate config, create Kind cluster, apply k8s manifests
make down     # Tear down the Kind cluster
make clean    # Remove generated configuration files
make apply    # Re-apply Kubernetes manifests
make logs     # Watch pod status (kubectl get pods -w)
```

---

## License

This project is provided as-is for educational and demonstration purposes.
