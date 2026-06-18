# Error Codes

This document details the non-200 HTTP responses that the ReviewSentinel API may return, what causes them, and how an integrating client should handle them.

---

## `422 Unprocessable Entity`

Returned when the request body fails Pydantic schema validation.

**Causes:**
- Missing a required field (e.g., omitting `title` in a `/predict` request)
- Invalid types (e.g., passing a string for `similar_reviews_count`)
- Value constraints violated (e.g., passing `similar_reviews_count: 20` when the max is 10)

**Response Format:**
ReviewSentinel uses FastAPI's default Pydantic error format.

```json
{
  "detail": [
    {
      "loc": ["body", "title"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Client Handling:**
Do not retry 422 errors automatically. The client must fix the invalid request payload before submitting again. Use the `loc` array to identify exactly which field caused the rejection.

---

## `503 Service Unavailable`

Returned when the API process is running, but the `InferencePipeline` could not load a model from disk.

**Causes:**
- The API was started in a fresh environment where no training pipeline has ever completed.
- The model artifacts directory (`artifacts/models/distilbert`) was deleted or corrupted.

**Self-Healing Trigger:**
When the API throws a 503, it automatically emits a `model.missing` event to the Prefect orchestrator (with a 5-minute rate limit to prevent event spam). If Prefect is configured correctly, this event triggers an automated training job to generate a model.

**Response Format:**
```json
{
  "detail": "Inference pipeline unavailable. A self-healing training job has been triggered. Please try again in a few minutes."
}
```

**Client Handling:**
The client should apply exponential backoff or alert a human. If the self-healing training job was triggered, the API will become available in approximately 5–15 minutes (depending on the dataset size and training configuration).

---

## `404 Not Found`

Returned when a requested resource does not exist. The most common occurrence is polling an invalid batch job ID.

**Causes:**
- Polling `GET /predict/batch/{job_id}` with an ID that was never submitted.
- Polling a `job_id` after the API process has restarted. Batch jobs are stored **in-memory** and do not survive process restarts.

**Response Format:**
```json
{
  "detail": "Job 'a3f8c2d1-4e7b-4a2c-b123-456789abcdef' not found."
}
```

**Client Handling:**
If polling a batch job returns 404, the client must resubmit the original `POST /predict/batch` request to get a new `job_id`.

---

## `500 Internal Server Error`

Returned when an unhandled exception occurs inside the application during request processing.

**Causes:**
- The model forward pass failed (e.g., out of memory, or a malformed tensor).
- A filesystem error occurred while attempting to write to the prediction log or the LLM Judge SQLite queue.
- An unexpected bug in the code.

**Response Format:**
```json
{
  "detail": "Internal server error message detailing the exception."
}
```

**Client Handling:**
These errors represent a system failure. The client may attempt a single retry, but sustained 500 errors require human intervention by the infrastructure or ML engineering team. Monitor the `sentiment_prediction_errors_total` Prometheus metric to catch these.
