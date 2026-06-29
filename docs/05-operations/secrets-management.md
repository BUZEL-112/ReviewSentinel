# Secrets Management

This document is the canonical single source of truth for all secrets in the ReviewSentinel pipeline.

## Required Credentials
The system relies on the following credentials to function:
1. **MinIO:** `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` (used as S3-compatible backend for MLflow artifacts).
2. **MLflow:** (Currently unauthenticated in local dev, but requires `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` if enabled in production).
3. **Prefect:** `PREFECT_API_URL` and optionally `PREFECT_API_KEY` if communicating with Prefect Cloud instead of a local Orion server.
4. **Ollama:** Not strictly authenticated, but requires internal network access controls.

## Where They Live

### Docker Compose
In the Docker Compose environment, secrets are injected via the `.env` file at the root of the project. A `.env.example` is provided to template this out. The `.env` file is explicitly ignored in `.gitignore`.

### Kubernetes (Kind / k3s)
In Kubernetes deployments, secrets are stored as native `Secret` objects. They should be created before applying the core manifests:
```bash
kubectl create secret generic minio-credentials \
  --from-literal=root-user=admin \
  --from-literal=root-password=supersecret
```
They are then mounted into the API and Prefect pods via environment variables.

## Production Rotation Strategy
**TODO: implement external secrets provider**
Currently, secrets must be manually rotated by deleting the Kubernetes `Secret` object, recreating it, and rolling the deployments. In the future, we will integrate with HashiCorp Vault or AWS Secrets Manager to handle automatic rotation.
