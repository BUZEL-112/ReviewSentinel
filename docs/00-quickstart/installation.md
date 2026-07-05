# Getting Started

This guide covers getting ReviewSentinel running on your local machine using Docker Compose. It is intended for developers who need to run the API, trigger training flows, or explore the dashboards locally.

If you are looking to deploy to a Kubernetes cluster, see [Deploying to Kind](../02-guides/kubernetes-deployment.md).

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

Wait until you see messages indicating that `api`, `mlflow`, and `prefect-server` are running. Note that MinIO may restart once or twice as its initialization job runs (this is normal; see [Troubleshooting](../05-operations/troubleshooting.md)).

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

- **Want to train a model?** Read the [Local Docker Guide](../02-guides/local-development.md#local-docker-guide) to learn how to trigger the Prefect training flow.
- **Integrating the API?** Read the full [API Reference](../04-api-reference/endpoints.md).
- **Want to contribute?** Read the [Contributing Guide](../08-contributing/README.md).
