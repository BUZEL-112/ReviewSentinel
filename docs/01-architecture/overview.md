# Architecture Overview

ReviewSentinel is composed of two logically separate paths that share infrastructure but have entirely distinct triggers, data flows, and responsibilities: the **training path** and the **inference path**.

Understanding which path a component belongs to is the fastest way to orient yourself in the codebase.

---

## The Two Paths

### Training Path

**Trigger:** Prefect schedule (currently manual or weekly via `training_flow`) or automatic retraining triggered by the drift monitor.

**Purpose:** Take raw Amazon review data, preprocess it, fine-tune DistilBERT, evaluate the result, compare against the current production model, and promote only if the new model is meaningfully better.

```
Amazon JSONL.gz
  → LoadData          (download + parse)
  → DataValidator     (Great Expectations checks: schema, nulls, class balance)
  → CleanDataBERT     (label mapping, URL removal, tokenisation, train/val/test split)
  → ModelTrainer      (HuggingFace Trainer, MLflow experiment logging)
  → ModelEvaluator    (weighted F1, accuracy, per-class metrics on test split)
  → Quality Gate      (compare new F1 vs. production baseline in MLflow)
    ├─ PASS → deploy_model_task (tag MLflow run as is_production=true, copy to artifacts/best_model)
    │         → rebuild_search_index_task (rebuild FAISS index from new training corpus)
    └─ FAIL → pipeline halts; current production model unchanged
```

### Inference Path

**Trigger:** HTTP POST to `/predict` (single) or `/predict/batch` (async batch job).

**Purpose:** Accept a raw review (title + optional body text), apply identical preprocessing to training, run sentiment classification, optionally add explainability and semantic search enrichment, queue uncertain predictions for LLM Judge review.

```
POST /predict
  → InferencePipeline._build_texts()   (URL removal + whitespace collapse + concatenate)
  → InferencePipeline._predict_sentiment()  (tokenise → DistilBERT forward pass → softmax)
  → [Optional] SentimentExplainer.explain()  (SHAP values via KernelExplainer)
  → [Optional] SemanticSearcher.search()     (FAISS ANN → alignment signal)
  → [If 0.40 ≤ confidence ≤ 0.60] → LLM Judge SQLite queue
  → PredictionLogger.log_prediction()        (JSONL append for drift monitoring)
  → Return PredictionResult (JSON)
```

---

## Component Responsibilities

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| **FastAPI** | Python, Uvicorn | Prediction API, health checks, Prometheus metrics endpoint, batch job queue |
| **InferencePipeline** | HuggingFace Transformers | Load model from disk, tokenise inputs, run forward pass, return softmax scores |
| **MLflow** | MLflow server + PostgreSQL backend | Experiment tracking, run parameter/metric logging, model artifact storage (backed by MinIO), production model tagging |
| **Prefect Server** | Prefect 2.x | Flow run scheduling, state history, UI |
| **Prefect Worker** | Prefect 2.x | Executes flows: training, monitoring, LLM Judge processing |
| **MinIO** | S3-compatible object store | MLflow artifact backend (model weights, tokeniser files) |
| **Nginx** | Reverse proxy | Routes `/api/*` traffic to FastAPI; serves static assets |
| **Evidently AI** | Python library | Drift report generation from prediction logs vs. training reference |
| **FAISS Index** | `IndexFlatIP` | Exact inner-product nearest-neighbor search for semantic review retrieval |
| **SQLite (LLM Judge Queue)** | SQLite | Durable queue for uncertain predictions pending LLM second opinion |
| **SQLite (Conflicts DB)** | SQLite | Stores LLM Judge vs. primary model disagreements for active learning |
| **Ollama** | Local LLM server | Hosts Mistral 7B (`nemotron-3-nano:4b`) for LLM Judge second opinions |
| **SetFit Aspect Model** | SetFit / sentence-transformers | Optional aspect classification (product quality, delivery, packaging, etc.) |

---

## Communication Protocols

```
FastAPI ←─HTTP──────► Prefect Server API       (emit Prefect events on model.missing)
FastAPI ←─disk──────► InferencePipeline        (loads model from artifacts/models/distilbert)
FastAPI ←─disk──────► FAISS Index              (artifacts/search/)
FastAPI ←─SQLite────► LLM Judge Queue          (artifacts/llm_judge/review_queue.db)
Prefect Worker ──HTTP──► MLflow                (experiment logging, run tagging)
Prefect Worker ──S3────► MinIO                 (artifact upload via boto3)
Prefect Worker ──HTTP──► Ollama                (LLM inference via REST API)
MLflow ──S3────────────► MinIO                 (model artifact storage)
Evidently AI ──disk─────► Prediction Logs      (artifacts/monitoring/prediction_log.jsonl)
Prometheus ──HTTP scrape─► /metrics            (Grafana or any Prometheus-compatible system)
```

---

## Storage Layout

```
artifacts/
├── models/
│   ├── distilbert/          # Active training output (tokenizer + model weights)
│   └── best_model/          # Latest quality-gate-approved model
├── search/
│   ├── faiss.index          # FAISS flat index
│   └── metadata.pkl         # Review metadata keyed by FAISS ID
├── monitoring/
│   ├── prediction_log.jsonl # Append-only prediction log (rotates at 100MB)
│   └── reports/             # Evidently HTML reports (last 12 retained)
├── llm_judge/
│   ├── review_queue.db      # SQLite queue for uncertain predictions
│   └── conflicts.db         # SQLite log of LLM Judge vs. model disagreements
└── evaluation/
    └── metrics.json         # Latest evaluation metrics from ModelEvaluator
```

---

## Deployment Topologies

ReviewSentinel supports two deployment modes. See [Component Diagram](component-diagram.md) for a visual comparison.

| Aspect | Docker Compose | Kubernetes (Kind) |
|--------|---------------|-------------------|
| Service discovery | Docker network DNS (`mlflow:5000`) | Kubernetes Service DNS (`mlflow:5000`) |
| Persistent storage | Named Docker volumes | PersistentVolumeClaims + `hostPath` mounts |
| Secrets | `.env` file | Kubernetes `Secret` objects |
| Entry point | `docker-compose.yaml` | `k8s/` manifests via `make up` |
| Recommended for | Local development, CI | Local Kubernetes testing, staging |

---

## Key Source Files

| File | Purpose |
|------|---------|
| [`src/api/api.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/api/api.py) | FastAPI application, all endpoints |
| [`src/pipeline/inference_pipeline.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/pipeline/inference_pipeline.py) | Model loading + inference logic |
| [`src/orchestration/flows.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/orchestration/flows.py) | All Prefect flows and tasks |
| [`src/monitoring/drift_monitor.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/monitoring/drift_monitor.py) | Evidently drift report generation |
| [`src/llm_judge/judge.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/llm_judge/judge.py) | Ollama LLM inference for second opinions |
| [`src/search/searcher.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/search/searcher.py) | FAISS search + alignment signal |
| [`configs/pipeline_params.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/configs/pipeline_params.yaml) | Master config for all flows and subsystems |
| [`configs/config.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/configs/config.yaml) | Data, model, and training hyperparameters |
