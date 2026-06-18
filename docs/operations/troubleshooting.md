# Troubleshooting Guide

This document captures known operational quirks, race conditions, and common errors encountered when running ReviewSentinel.

---

## 1. MinIO Init Race Condition

**Symptom:** The training pipeline fails immediately with `botocore.exceptions.ClientError: An error occurred (404) when calling the HeadBucket operation: Not Found`.

**Cause:** MLflow and Boto3 expect the `reviewsentinel-artifacts` bucket to exist in MinIO before they can upload artifacts. In the Docker Compose and Kubernetes deployments, a `minio-init` container attempts to create this bucket on startup. If the Prefect Worker starts a flow before this init container finishes, the flow crashes.

**Resolution:**
In `src/orchestration/flows.py`, there is a `setup_minio_bucket` task that runs at the very beginning of the flow to check and create the bucket if missing. If you are bypassing the flow and running `train_model.py` directly, you must ensure the bucket exists first via the MinIO console (`http://localhost:9001`) or the `mc` CLI.

---

## 2. Ollama Healthcheck Failure on First Boot

**Symptom:** The `ollama` container logs show `Error: context deadline exceeded` or the LLM Judge flow fails with a connection timeout.

**Cause:** The first time the Ollama container boots, it must download the `nemotron-3-nano:4b` weights (approx. 2.5GB). Depending on your internet speed, this download can exceed Docker's healthcheck timeout or the Prefect flow's request timeout.

**Resolution:**
Wait for the download to finish. You can verify the status by tailing the Ollama logs:
```bash
docker logs reviewsentinel-ollama-1 -f
```
Once the model is pulled, subsequent boots take seconds.

---

## 3. Prefect Worker Cannot Find `prefect.yaml`

**Symptom:** `prefect-worker` crashes with `FileNotFoundError: No such file or directory: 'prefect.yaml'`.

**Cause:** The Prefect worker executes flows using the working directory context. In Docker/Kubernetes, the worker container expects the codebase to be mounted at `/app` or for the current working directory to be the repository root.

**Resolution:**
Ensure the Docker Compose bind mounts or Kubernetes `hostPath` mounts are correctly mapping your local repository into the container's working directory.

---

## 4. `ModuleNotFoundError: No module named 'src'`

**Symptom:** Running scripts locally (outside Docker) fails with import errors.

**Cause:** Python cannot resolve the `src` module if you run scripts from inside a subdirectory without modifying the path.

**Resolution:**
Run all scripts from the repository root, ensuring the root is in your `PYTHONPATH`:
```bash
# Correct
PYTHONPATH=. python scripts/run_flow.py

# Incorrect
cd scripts/
python run_flow.py
```

Inside the Docker containers, `PYTHONPATH=/app` is set globally in the Dockerfile to prevent this.

---

## 5. MLflow URI Unreachable Outside Docker

**Symptom:** Running `train_model.py` on your host machine fails with `requests.exceptions.ConnectionError: HTTPConnectionPool(host='mlflow', port=5000)`.

**Cause:** The `config.yaml` sets `mlflow.tracking_uri` to `http://mlflow:5000`. This DNS name only resolves *inside* the Docker network or Kubernetes cluster.

**Resolution:**
The `ModelTrainer` class has a fallback mechanism (lines 55–63 in `train_model.py`) that catches the DNS error and retries with `http://localhost:5000`. If it is still failing, ensure you have port-forwarded MLflow to `localhost:5000` (if using Kubernetes) or that the Docker Compose stack is running.
