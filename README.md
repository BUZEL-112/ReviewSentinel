# ReviewSentinel

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



ReviewSentinel is a fine-tuned DistilBERT classifier for three-class review sentiment (negative / neutral / positive), built around an MLOps pipeline rather than a one-shot script. The system handles quality-gated model promotion, scheduled drift monitoring with auto-retraining, and an async LLM second-opinion loop for low-confidence predictions.

## Quickstart

ReviewSentinel runs entirely within Docker Compose for local development.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BUZEL-112/ReviewSentinel.git
   cd ReviewSentinel
   ```

2. **Set up your local environment:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set secure values for `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`.

3. **Start the stack:**
   ```bash
   docker-compose -f docker/docker-compose.yaml up --build -d
   ```
   This spins up FastAPI, Prefect, MLflow, MinIO, and an Ollama LLM node. Wait until the `api`, `mlflow`, and `prefect-server` are fully running.

4. **Verify Health:**
   ```bash
   curl http://localhost:8000/health
   ```

## Usage Example

If a model is loaded and the API is ready, you can make a single-review sentiment prediction via a POST request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Broke after one week",
    "text": "Complete waste of money. The battery died and it stopped charging."
  }'
```

*Expected Response Summary:*
```json
{
  "text": "broke after one week complete waste of money the battery died and it stopped charging",
  "label": "negative",
  "confidence": 0.9832
}
```

## Tech Stack

* **Model**: Fine-tuned DistilBERT (Sentiment) and Ollama with Mistral 7B (LLM Judge)
* **API**: FastAPI (served via Uvicorn)
* **Orchestration & Workflows**: Prefect
* **Experiment Tracking**: MLflow
* **Object Storage**: MinIO
* **Vector Search**: FAISS
* **Infrastructure**: Docker Compose, Kubernetes (Kind for local k8s)

## Features

1. **Real-time Inference**: High-performance FastAPI endpoints for single and batch predictions.
2. **Quality-Gated Model Promotion**: Automated testing and CI/CD validation before models hit production.
3. **Automated Drift Monitoring**: Built-in Evidently AI integration to detect data drift and automatically trigger retraining.
4. **LLM Second-Opinion Loop**: Background asynchronous review for uncertain or low-confidence predictions using a local Mistral 7B LLM.
5. **SHAP Explanations**: Token-level attribution to understand model decisions.

## Configurations

The project relies on two main configuration layers:
1. **Environment Variables**: Managed via `.env` (copy from `.env.example`). This is used for sensitive credentials such as `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`.
2. **Pipeline Parameters**: Managed via `configs/config.yaml` and `configs/pipeline_params.yaml`. These dictate model hyperparameters, drift thresholds, and pipeline settings.

## API Reference

The API runs locally on port `8000`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check and model loading status |
| GET | `/metrics` | Prometheus metrics exposition |
| POST | `/predict` | Single-review sentiment prediction |
| POST | `/predict/batch` | Submit async batch job |
| GET | `/predict/batch/{job_id}` | Poll batch job status |
| GET | `/monitoring/drift/latest` | Latest drift evaluation metadata |
| GET | `/monitoring/drift/report` | Full Evidently HTML drift report |

Detailed interactive Swagger documentation is available locally at `http://localhost:8000/docs`.

## Troubleshooting

- **API pod crashlooping**: If you have never trained a model, the API will return 503s but it should *not* crashloop. If the pod is restarting constantly, check its logs. It likely cannot reach MLflow.
- **Prefect worker cannot find files**: The worker container relies on host volumes. If using Docker Desktop on macOS/Windows, ensure file sharing is permitted for the local repository directory.
- **MinIO Restarting**: MinIO requires an initialization job to create artifact buckets. If this fails, MinIO might restart. Wait for the `minio-init` job to complete successfully.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.