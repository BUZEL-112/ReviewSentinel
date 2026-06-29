# Monitoring Alerts

ReviewSentinel exposes Prometheus metrics at `/metrics`. You should set up alerts to catch operational anomalies before they affect users.

## Recommended Alerts

### `prediction_latency_p99 > 500ms`
**What it means:** The 99th percentile of API responses is taking over half a second.
**Why it happens:** This often indicates that the system is CPU-starved, or if the LLM judge is processing a huge backlog on the same node, it's stealing cycles from the API.
**What to do:** Check Ollama queue depth. If the LLM judge is blocked, you may see high latency on `/predict`. See the [Runbooks](runbooks.md) for scaling or restarting the judge.

### `sentiment_prediction_errors_total > 5 / minute`
**What it means:** The API is returning `500 Internal Server Error` at an elevated rate.
**Why it happens:** Usually a symptom of OOM (Out of Memory) kills, a corrupted model file, or a network partition to MLflow preventing the model from downloading.
**What to do:** Check API container logs immediately.

### `sentiment_batch_jobs_active > 100`
**What it means:** The async batch queue is backing up.
**Why it happens:** Consumers are submitting batch predictions faster than the single-threaded background worker can drain them.
**What to do:** Temporarily rate limit upstream clients or investigate node CPU starvation.
