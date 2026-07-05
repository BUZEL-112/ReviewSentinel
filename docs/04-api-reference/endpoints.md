# API Reference

**Base URL:** `http://localhost:8000` (Docker Compose) | `http://localhost:30080/api` (Kubernetes)  
**Authentication:** None (see [Authentication](authentication.md))  
**Content-Type:** `application/json`  
**Interactive Docs:** `GET /docs` (Swagger UI) | `GET /redoc` (ReDoc)

---

## Endpoints

| Method | Path | Tag | Description |
|--------|------|-----|-------------|
| GET | `/health` | Health | Service health check |
| GET | `/metrics` | Observability | Prometheus metrics |
| GET | `/test` | Health | Lightweight reachability check |
| POST | `/predict` | Inference | Single-review sentiment prediction |
| POST | `/predict/batch` | Inference | Submit async batch job |
| GET | `/predict/batch/{job_id}` | Inference | Poll batch job status |
| GET | `/monitoring/drift/latest` | Monitoring | Latest drift evaluation metadata |
| GET | `/monitoring/drift/report` | Monitoring | Full Evidently HTML drift report |

---

## GET `/health`

Returns the API's current operational status, model load state, and uptime. Used by Kubernetes liveness and readiness probes.

### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "model_dir": "artifacts/models/distilbert",
  "aspect_enabled": false,
  "uptime_seconds": 3726.4,
  "timestamp": "2026-06-17T10:30:00.000000+00:00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` if model loaded, `"unhealthy"` otherwise |
| `model_loaded` | boolean | Whether `InferencePipeline` initialised successfully |
| `model_version` | string | Content of `VERSION` file at project root, or `"unknown"` |
| `model_dir` | string | Filesystem path where the model was loaded from |
| `aspect_enabled` | boolean | Whether the SetFit aspect classifier is active |
| `uptime_seconds` | float | Seconds since API process started |
| `timestamp` | string | UTC ISO-8601 timestamp |

### Status Codes
- `200 OK` — always returned (check `model_loaded` to distinguish healthy vs. degraded)

---

## GET `/metrics`

Returns Prometheus metrics in the standard text exposition format. Intended for scraping by Prometheus or a compatible system.

### Exposed Metrics

| Metric name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `sentiment_predictions_total` | Counter | `endpoint`, `label` | Total predictions served |
| `sentiment_prediction_errors_total` | Counter | `endpoint` | Total prediction errors |
| `sentiment_prediction_latency_seconds` | Histogram | `endpoint` | End-to-end latency |
| `sentiment_confidence_score` | Histogram | — | Distribution of confidence scores |
| `sentiment_batch_jobs_active` | Gauge | — | Currently processing batch jobs |

---

## POST `/predict`

Single-review sentiment prediction. The model forward pass runs in a thread pool so the event loop is never blocked.

### Request Body

```json
{
  "title": "Broke after one week",
  "text": "Complete waste of money. The battery died and it stopped charging.",
  "include_explanation": false,
  "include_similar_reviews": false,
  "similar_reviews_count": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `title` | string | ✅ | — | Review title or headline |
| `text` | string | ❌ | `null` | Review body text |
| `include_explanation` | boolean | ❌ | `false` | Include SHAP token attributions |
| `include_similar_reviews` | boolean | ❌ | `false` | Include semantically similar historical reviews |
| `similar_reviews_count` | integer | ❌ | `5` | Number of similar reviews (max 10) |

> [!NOTE]
> `text` is optional. The model works on `title` alone, but predictions are generally more reliable when `text` is provided. The two fields are concatenated as `"{title} {text}"` before inference.

### Response Body

```json
{
  "text": "broke after one week complete waste of money the battery died and it stopped charging",
  "label": "negative",
  "confidence": 0.9832,
  "scores": {
    "negative": 0.9832,
    "neutral": 0.0112,
    "positive": 0.0056
  },
  "aspect": null,
  "flags": {
    "short_review": false,
    "low_confidence": false,
    "possible_sarcasm": false
  },
  "processing_time_ms": 42.7,
  "model_version": "1.0.0",
  "explanation": null,
  "similar_reviews": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Cleaned input text passed to the model (after URL removal + whitespace collapse) |
| `label` | string | Predicted sentiment: `negative` \| `neutral` \| `positive` |
| `confidence` | float | Softmax probability for the winning class (0.0–1.0) |
| `scores` | object | Full three-class probability distribution |
| `scores.negative` | float | Probability of negative sentiment |
| `scores.neutral` | float | Probability of neutral sentiment |
| `scores.positive` | float | Probability of positive sentiment |
| `aspect` | object \| null | Aspect classification if SetFit model is enabled; `null` otherwise |
| `aspect.label` | string | Predicted aspect category (e.g., `quality`, `delivery`, `packaging`) |
| `flags` | object | Edge-case quality flags (see below) |
| `flags.short_review` | boolean | `true` if cleaned text is shorter than 10 characters |
| `flags.low_confidence` | boolean | `true` if `confidence < 0.60` |
| `flags.possible_sarcasm` | boolean | `true` if a sarcasm heuristic pattern matched the text |
| `processing_time_ms` | float | Milliseconds from request receipt to response |
| `model_version` | string | Version tag of the loaded model |
| `explanation` | object \| null | SHAP explanation (if `include_explanation: true`) |
| `similar_reviews` | object \| null | Semantic search results (if `include_similar_reviews: true`) |

### Explanation Object (when `include_explanation: true`)

```json
{
  "predicted_class": "negative",
  "target_class_explained": "negative",
  "baseline_probability": 0.3333,
  "tokens": [
    {"token": "broke", "shap_value": 0.45, "direction": "toward"},
    {"token": "waste", "shap_value": 0.38, "direction": "toward"},
    {"token": "love", "shap_value": 0.12, "direction": "against"}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted_class` | string | The class the model predicted |
| `target_class_explained` | string | The class being explained (always equals `predicted_class`) |
| `baseline_probability` | float | Uniform prior: `1/3 = 0.3333` for three classes |
| `tokens[].token` | string | WordPiece token from the input |
| `tokens[].shap_value` | float | SHAP attribution magnitude (higher = stronger contribution) |
| `tokens[].direction` | string | `"toward"` = pushes toward the predicted class; `"against"` = pushes away |

> [!NOTE]
> SHAP explanations use `KernelExplainer`, which is model-agnostic but slow (up to 1–2 seconds). Only enable `include_explanation` when the explanation data is actually needed. Latency for unexplained requests is typically 40–100ms.

### Similar Reviews Object (when `include_similar_reviews: true`)

```json
{
  "results": [
    {
      "rank": 1,
      "similarity_score": 0.923,
      "clean_text": "battery died after three days",
      "raw_title": "Terrible battery life",
      "raw_text": "Bought this last month and the battery is already dead.",
      "known_sentiment": "negative",
      "rating": 1.0,
      "sentiment_alignment": true
    }
  ],
  "query_text": "broke after one week...",
  "top_k": 3,
  "alignment_rate": 0.95,
  "alignment_signal": "STRONG_SUPPORT",
  "search_latency_ms": 12.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `alignment_signal` | string | `STRONG_SUPPORT` (≥80% match) \| `MIXED` (40–80%) \| `CONTRADICTS` (<40%) |
| `alignment_rate` | float | Fraction of retrieved reviews whose known sentiment matches the prediction |
| `results[].similarity_score` | float | Cosine similarity (inner product of normalised vectors, 0.0–1.0) |
| `results[].sentiment_alignment` | boolean | Whether this review's known sentiment matches the current prediction |

**Interpreting alignment signals:**
- `STRONG_SUPPORT` — historical evidence strongly agrees with the prediction
- `MIXED` — the similar reviews are split; treat the prediction with more caution
- `CONTRADICTS` — historical similar reviews disagree; consider flagging for human review

### Status Codes
- `200 OK` — prediction succeeded
- `422 Unprocessable Entity` — request body validation failed (missing `title`, wrong type)
- `500 Internal Server Error` — model forward pass failed
- `503 Service Unavailable` — model not loaded (see [Error Codes](error-handling.md))

---

## POST `/predict/batch`

Submits a batch of reviews for asynchronous processing. Returns immediately with a `job_id`. Poll `GET /predict/batch/{job_id}` for results.

### Request Body

```json
{
  "items": [
    {"title": "Great product", "text": "Works perfectly."},
    {"title": "Broke immediately", "text": "Stopped working after one day."}
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `items` | array | ✅ | List of prediction requests (minimum 1 item) |
| `items[].title` | string | ✅ | Review title |
| `items[].text` | string | ❌ | Review body text |
| `items[].include_explanation` | boolean | ❌ | SHAP explanation per item |

### Response (`202 Accepted`)

```json
{
  "job_id": "a3f8c2d1-4e7b-4a2c-b123-456789abcdef",
  "status": "queued",
  "item_count": 2
}
```

Poll the returned `job_id` at `GET /predict/batch/{job_id}`.

---

## GET `/predict/batch/{job_id}`

Polls the status and result of a submitted batch job.

### Response

```json
{
  "job_id": "a3f8c2d1-4e7b-4a2c-b123-456789abcdef",
  "status": "done",
  "predictions": [ ... ],
  "error": null
}
```

| Field | `status` value | Meaning |
|-------|----------------|---------|
| `job_id` | any | The job identifier |
| `status` | `queued` | Job accepted but not yet started |
| `status` | `processing` | Inference in progress |
| `status` | `done` | Results available in `predictions` |
| `status` | `failed` | An error occurred; check `error` field |

> [!WARNING]
> Batch job results are stored **in-memory** only. They are lost if the API process restarts. The in-memory store does not have a TTL or eviction policy — very long-running APIs with high batch throughput will see memory grow. For production use cases, persist results to a database.

### Status Codes
- `200 OK` — job found
- `404 Not Found` — `job_id` does not exist (never submitted, or API restarted)

---

## GET `/monitoring/drift/latest`

Returns JSON metadata about the most recent drift evaluation.

```json
{
  "timestamp": "2026-06-16T03:00:00Z",
  "report_path": "artifacts/monitoring/reports/drift_20260616_030000.html",
  "n_drifted_features": 2,
  "total_features": 4,
  "drift_fraction": 0.50,
  "action_taken": "TRIGGER_RETRAINING"
}
```

---

## GET `/monitoring/drift/report`

Serves the latest Evidently AI HTML drift report as an HTML response. Open directly in a browser.

---

## Common Patterns

### Check model is loaded before sending predictions

```python
import requests

health = requests.get("http://localhost:8000/health").json()
if not health["model_loaded"]:
    raise RuntimeError("Model not ready — check API logs")
```

### Single prediction with explanation

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Terrible quality",
    "text": "The packaging was broken and the product looks cheap.",
    "include_explanation": true
  }'
```

### Batch job — submit and poll

```python
import requests, time

# Submit
resp = requests.post("http://localhost:8000/predict/batch", json={
    "items": [
        {"title": "Amazing product", "text": "Best purchase I've made."},
        {"title": "Broken on arrival", "text": "DOA. Very disappointed."}
    ]
})
job_id = resp.json()["job_id"]

# Poll until done
while True:
    result = requests.get(f"http://localhost:8000/predict/batch/{job_id}").json()
    if result["status"] in ("done", "failed"):
        break
    time.sleep(0.5)

print(result["predictions"])
```
