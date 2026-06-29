# Inference API Guide

This guide details how to interact with the ReviewSentinel REST API.

## `/predict` (Single Prediction, Sync)
Performs sentiment prediction synchronously.

**Request Body:**
```json
{
  "title": "Broke after one week",
  "text": "Complete waste of money."
}
```

**Response Body:**
```json
{
  "text": "broke after one week complete waste of money",
  "label": "negative",
  "confidence": 0.9832,
  "scores": {
    "negative": 0.9832,
    "neutral": 0.0112,
    "positive": 0.0056
  }
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Broke after one week", "text": "Complete waste of money."}'
```

**Common Errors:**
- `422 Unprocessable Entity`: Missing the required `title` field.
- `503 Service Unavailable`: Model is not currently loaded.

---

## `/predict/batch` (Async Job Queue)
Submits a batch of reviews for asynchronous processing.

**Request Body:**
```json
{
  "items": [
    {"title": "Great product", "text": "Works perfectly."},
    {"title": "Broke immediately", "text": "Stopped working."}
  ]
}
```

**Response Body (`202 Accepted`):**
```json
{
  "job_id": "a3f8c2d1-4e7b-4a2c-b123-456789abcdef",
  "status": "queued",
  "item_count": 2
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"title": "Great"}]}'
```
*Note: Poll `GET /predict/batch/{job_id}` to retrieve results.*

**Common Errors:**
- `422 Unprocessable Entity`: Empty items list.

---

## `/health` (Liveness)
Returns the API's operational status.

**Response Body:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3726.4
}
```

**Example cURL:**
```bash
curl http://localhost:8000/health
```

---

## `/metrics` (Prometheus)
Exposes prometheus metrics.

**Example cURL:**
```bash
curl http://localhost:8000/metrics
```
