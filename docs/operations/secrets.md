# CI/CD & Cluster Secrets Reference

This document catalogs all operational secrets required by the GitHub Actions pipelines and the deployment clusters. The actual secret values must never be committed to the repository.

---

## GitHub Actions Secrets

These secrets must be configured in the GitHub repository settings (**Settings > Secrets and variables > Actions**).

| Secret Name | Purpose | Used By | Expiry / Verification |
|-------------|---------|---------|-----------------------|
| `GITOPS_PAT` | Fine-grained PAT with `contents:write` for the repo. Allows automated workflows to push commits that ArgoCD detects. | `update-image-tag.yaml` | Expires based on PAT creation date. Needs manual rotation. |
| `MLFLOW_TRACKING_URI` | URL of the remote MLflow tracking server used to fetch production baseline metrics. | `pr-checks.yaml`, `scheduled-eval.yaml` | Static infrastructure URL. |
| `AWS_ACCESS_KEY_ID` | Access key for MinIO/S3 object storage to pull MLflow artifacts. | `pr-checks.yaml`, `scheduled-eval.yaml` | Controlled by MinIO root user settings. |
| `AWS_SECRET_ACCESS_KEY` | Secret key for MinIO/S3 object storage. | `pr-checks.yaml`, `scheduled-eval.yaml` | Controlled by MinIO root user settings. |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint URL for the S3-compatible MinIO instance. | `pr-checks.yaml`, `scheduled-eval.yaml` | Static infrastructure URL. |
| `PREFECT_API_URL` | URL for the Prefect orchestration API. | `scheduled-eval.yaml` | Static infrastructure URL. |
| `PREFECT_API_KEY` | Authentication key for triggering Prefect flows via API. | `scheduled-eval.yaml` | Managed in Prefect Cloud/Server. |
| `PREFECT_TRAINING_DEPLOYMENT_ID`| UUID of the Prefect deployment that executes the model training flow. | `scheduled-eval.yaml` | Found in the Prefect UI. |
| `SLACK_WEBHOOK_URL` | Webhook URL for posting CI/CD and evaluation status updates to Slack. | `scheduled-eval.yaml`, `notify-slack` | Standard Slack webhook format. |
| `CODECOV_TOKEN` | Token for uploading coverage reports to Codecov. | `pr-checks.yaml` | Managed in Codecov. |

---

## Local Environment Secrets (`.env`)

For local Docker Compose development, secrets are managed in a `.env` file at the repository root.

1. Copy the template: `cp .env.example .env`
2. Populate the values.

| Variable | Purpose | How to generate |
|----------|---------|-----------------|
| `MINIO_ROOT_USER` | Admin username for MinIO. | Choose a username (e.g., `admin`). |
| `MINIO_ROOT_PASSWORD` | Admin password for MinIO. | Generate a strong password (minimum 8 chars). |
| `POSTGRES_USER` | (If using Postgres) Database user. | Choose a username. |
| `POSTGRES_PASSWORD`| (If using Postgres) Database password. | Generate a strong password. |

---

## Kubernetes Cluster Secrets

These secrets are managed externally and must be applied to the cluster (manually, via SealedSecrets, or via a Vault injector) before deploying the Helm charts or applying the manifests.

### Core Application Secrets

To deploy to Kind using the provided manifests, you must create a `reviewsentinel-secrets` Secret in the `default` namespace.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: reviewsentinel-secrets
type: Opaque
stringData:
  # Base64 encoding is NOT required when using stringData
  AWS_ACCESS_KEY_ID: "your-minio-root-user"
  AWS_SECRET_ACCESS_KEY: "your-minio-root-password"
```

Apply it via:
```bash
kubectl apply -f path/to/your/secret.yaml
```

**Verification:** If this secret is missing or incorrect, the `minio` pod will fail to start, and the `prefect-worker` pod will throw boto3 Access Denied errors during training.

### ArgoCD Secrets

If using ArgoCD for GitOps deployments:

| Secret Name | Namespace | Purpose |
|-------------|-----------|---------|
| `reviewsentinel-repo-creds` | `argocd` | Stores the Git token (often the same as `GITOPS_PAT`) so ArgoCD can fetch private repository contents. Requires the label `argocd.argoproj.io/secret-type=repository`. |
| `argocd-notifications-secret` | `argocd` | Stores the Slack Bot Token used by `argocd-notifications-cm` for dispatching sync status alerts. |
