# ReviewSentinel

ReviewSentinel is a fine-tuned DistilBERT classifier for three-class review sentiment (negative / neutral / positive), built around an MLOps pipeline rather than a one-shot script. The system handles quality-gated model promotion, scheduled drift monitoring with auto-retraining, and an async LLM second-opinion loop for low-confidence predictions — all orchestrated with Prefect and tracked in MLflow. This is a portfolio project trained on public Amazon review data; the infrastructure choices are sized for a single-node demo environment, not for multi-region production load.

[![CI](https://github.com/BUZEL-112/ReviewSentinel/actions/workflows/pr-checks.yaml/badge.svg)](https://github.com/BUZEL-112/ReviewSentinel/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
Amazon Review JSONL
        │
        ▼
  LoadData ──► CleanDataBERT
                     │
                     ▼
               ModelTrainer (DistilBERT fine-tune)
                     │
                     ▼
              ModelEvaluator
                     │
              Quality Gate: F1 ≥ baseline + 0.01?
             /                         \
           YES                          NO
            │                           │
   MLflow tag (is_production=true)   Blocked — no deploy
   FAISS flat-index rebuild
            │
            ▼
   ┌─── Inference (/predict) ────────────────────────────────────┐
   │  InferencePipeline                                          │
   │    ├─ DistilBERT → sentiment score                          │
   │    ├─ SetFit → aspect label                                 │
   │    ├─ SHAP → token attributions                             │
   │    └─ FAISS → top-k similar reviews                         │
   │         │                                                   │
   │   confidence 0.40–0.60?                                     │
   │         │ YES                                               │
   │   SQLite queue → Mistral 7B (Ollama)                        │
   │         │ CONFLICT                                          │
   │   conflicts.db ──► next training set                        │
   └─────────────────────────────────────────────────────────────┘
            │
   Prediction log (JSONL)
            │
   Evidently AI drift report (weekly, Wasserstein)
            │  drift_fraction ≥ 0.50?
            └─► trigger_retraining_task() ──► ModelTrainer (loop)
```

Async boundary: the LLM judge runs in a separate Prefect flow (`judge_processing_flow`) on a 4-hour cron. The training and monitoring flows are independent schedules; training does not block serving.

---

## Design Decisions

**Quality gate queries MLflow rather than a local file.**
`quality_gate_task` (`src/orchestration/flows.py`) calls `mlflow.search_runs(filter_string="tags.is_production='true'")` to find the current baseline. The alternative — reading a local metrics JSON — would break under any deployment topology where the worker and the model registry are not on the same filesystem. The tradeoff: the gate has a hard dependency on MLflow being reachable at promotion time; a network partition silently defaults to `first_run_auto_deploy` behavior.

**Drift triggers retraining through a threshold pair, not a single cutoff.**
`evaluate_drift_task` (`src/orchestration/monitoring_tasks.py`) maps `drift_fraction` to a three-way enum: `NONE` (< 0.30), `ALERT` (0.30–0.49), `TRIGGER_RETRAINING` (≥ 0.50), configured in `pipeline_params.yaml`. A single threshold would force a choice between noisy alerts and missed drift. The tradeoff: the alert band (0.30–0.49) produces notifications with no automatic action, so it only matters if someone is watching.

**LLM judge uses SQLite, not a message broker.**
The judge queue (`artifacts/llm_judge/review_queue.db`) is a SQLite table with `status IN ('pending', 'processing', 'done', 'failed')`. This gives atomic dequeue without running a Redis or RabbitMQ sidecar. The tradeoff: SQLite write-ahead log means only one writer at a time; if the judge flow is ever parallelized across workers, this becomes a bottleneck. See [ADR 003](docs/decisions/003-sqlite-for-llm-judge-queue.md).

**FAISS flat index (IndexFlatIP) instead of HNSW.**
At the current corpus size (≤ 50 000 reviews, configurable via `semantic_search.build.max_index_rows`), exact inner-product search is fast enough that approximate search provides no latency benefit and removes the need to reason about recall@k degradation after index updates. The tradeoff: a flat index does not support incremental inserts; the index is fully rebuilt after every successful deployment. See [ADR 004](docs/decisions/004-faiss-flat-index-over-hnsw.md).

**Three-tier prompt parsing for LLM output.**
`src/llm_judge/` applies regex extraction, then JSON parse, then a keyword fallback before marking a response as unparseable. This was necessary because Mistral 7B via Ollama produces inconsistent output formatting across temperature settings. Falling through to the keyword tier instead of raising means the judge batch does not halt on a single malformed response. The tradeoff: keyword matching can produce false positives for edge-case phrasing. See [ADR 007](docs/decisions/007-three-tier-prompt-parsing.md).

**MinIO instead of AWS S3.**
All MLflow artifacts are stored in a MinIO bucket (`reviewsentinel-artifacts`) behind an S3-compatible endpoint. This keeps the stack fully local without boto3 credential management. The tradeoff: any production migration requires updating `MLFLOW_S3_ENDPOINT_URL` and real IAM credentials; there is no IAM policy or bucket versioning in the demo. See [ADR 005](docs/decisions/005-minio-over-s3.md).

---

## Quickstart

**Prerequisites:** Docker, Docker Compose. Full environment details in [Getting Started](docs/guides/getting-started.md).

```bash
git clone https://github.com/BUZEL-112/ReviewSentinel.git
cd ReviewSentinel

cp .env.example .env   # fill in credentials

docker-compose -f docker/docker-compose.yaml up --build -d

# Confirm the API is live
curl http://localhost:8000/health
curl http://localhost:30080/api/health
# Run a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Broke after one week", "text": "Complete waste of money."}'
```

For the full training pipeline (loads data, trains DistilBERT, runs quality gate, rebuilds FAISS index), see [Training Guide](docs/ml/training-guide.md). For the Kubernetes path, see [Deploying to Kind](docs/operations/deploying-to-kind.md).

---

## Project Structure

```
src/
  api/           FastAPI app — /predict, /health, /metrics endpoints
  data/          Data loading, BERT-specific cleaning, aspect dataset builder
  models/        DistilBERT trainer, SetFit aspect model, evaluator, quality gate logic
  pipeline/      Training, evaluation, and inference pipeline orchestrators
  orchestration/ Prefect flows and tasks — training, monitoring, judge processing
  monitoring/    Evidently drift monitor, reference data store, report server
  llm_judge/     SQLite queue manager, Ollama client, prompt builder, conflict logger
  search/        Sentence encoder (MiniLM), FAISS index builder, searcher
  explainability/SHAP token attribution wrapper
  utils/         Shared logger, MinIO upload helper
configs/         pipeline_params.yaml — single config read by every subsystem
tests/           unit/, integration/, smoke/ test suites
docs/            ADRs, guides, API reference, ML model/data cards, runbooks
docker/          Compose file and per-service Dockerfiles
k8s/             Kubernetes manifests (Kind-compatible)
```

---

## Stack

Model: DistilBERT (HuggingFace Transformers), SetFit (aspect), all-MiniLM-L6-v2 (embeddings) | Serving: FastAPI + Uvicorn, SHAP, FAISS-cpu | Orchestration: Prefect 2.x, Evidently AI, MLflow 2.x | LLM: Mistral 7B via Ollama | Storage: MinIO (S3-compatible), SQLite | Infra: Docker Compose, Kind (local Kubernetes), Prometheus

---

## Testing

**`tests/unit/test_quality_gate.py`** — calls `quality_gate_task.fn` directly (bypassing the Prefect runtime) with a patched `mlflow.search_runs`. Covers the regression case (new F1 < baseline), the exact-boundary case (improvement == threshold), and the first-run auto-deploy flag. This is the guard against accidentally loosening or inverting the gate comparison operator.

**`tests/unit/test_drift_thresholds.py`** — calls `evaluate_drift_task.fn` with a stubbed `DriftResult` at values spanning both boundaries (0.30 alert, 0.50 retrain). Catches any change to threshold evaluation logic that would cause the monitoring flow to silently skip retraining when drift exceeds the configured limit.

Run the unit suite: `pytest tests/unit/ -v`

---

## Known Limitations

- **No online FAISS updates.** The index is rebuilt from scratch on every deployment; adding a single review requires a full rebuild of up to 50 000 vectors.
- **SQLite judge queue is single-writer.** Parallelizing the `judge_processing_flow` across multiple Prefect workers will cause lock contention.
- **Drift monitor requires ≥ 50 prediction log entries** (`min_current_samples` in config) before it emits a report; a low-traffic deployment will never trigger monitoring.
- **LLM judge timeout is synchronous per batch.** A slow or unavailable Ollama instance blocks the entire flow for `timeout_seconds × batch_size` before failing.
- **No auth on internal services.** MLflow, Prefect, and MinIO run without authentication in the Docker Compose stack; the API has no token validation. See [Authentication](docs/api/authentication.md) for the production gap analysis.

---

## Possible Next Steps

- **Streaming conflict export** — replace the `export_on_retrain` batch dump with a CDC-style append to the training set so conflicts are incorporated on the next scheduled run without a manual trigger.
- **Incremental FAISS index via HNSW** — switch `IndexFlatIP` to `IndexHNSWFlat` with a nightly merge step to avoid full rebuilds as the corpus grows past 50k reviews.
- **Judge result parity test** — add a regression fixture of 20 known LLM responses and assert the three-tier parser assigns the correct label to each, catching prompt or regex changes that silently degrade parse accuracy.
