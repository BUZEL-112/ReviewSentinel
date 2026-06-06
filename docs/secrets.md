# CI/CD Secrets Reference

This document catalogs all the operational secrets required by the GitHub Actions pipelines and the cluster itself. The actual secret values must never be committed to the repository.

## GitHub Actions Secrets

These secrets must be configured in the GitHub repository settings (Settings > Secrets and variables > Actions).

| Secret Name | Purpose | Used By | Expiry / Notes |
|-------------|---------|---------|----------------|
| `GITOPS_PAT` | Fine-grained Personal Access Token with `contents:write` permission for the repository. Allows automated workflows to push commits that ArgoCD will detect. | `update-image-tag.yaml` | Expires based on PAT creation date. Needs manual rotation. |
| `MLFLOW_TRACKING_URI` | Full URL of the remote MLflow tracking server used to fetch production baseline metrics. | `pr-checks.yaml`, `scheduled-eval.yaml` | Static infrastructure URL. |
| `AWS_ACCESS_KEY_ID` | Access key for MinIO / S3 object storage to pull MLflow artifacts. | `pr-checks.yaml`, `scheduled-eval.yaml` | Controlled by MinIO root user settings. |
| `AWS_SECRET_ACCESS_KEY` | Secret key for MinIO / S3 object storage. | `pr-checks.yaml`, `scheduled-eval.yaml` | Controlled by MinIO root user settings. |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint URL for the S3-compatible MinIO instance. | `pr-checks.yaml`, `scheduled-eval.yaml` | Static infrastructure URL. |
| `PREFECT_API_URL` | URL for the Prefect orchestration API. | `scheduled-eval.yaml` | Static infrastructure URL. |
| `PREFECT_API_KEY` | Authentication key for triggering Prefect flows via API. | `scheduled-eval.yaml` | Managed in Prefect Cloud/Server. |
| `PREFECT_TRAINING_DEPLOYMENT_ID` | UUID of the Prefect deployment that executes the model training flow. | `scheduled-eval.yaml` | Found in the Prefect UI. |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL for posting CI/CD and evaluation status updates to a Slack channel. | `scheduled-eval.yaml`, `notify-slack` | Standard Slack webhook format. |
| `CODECOV_TOKEN` | Authentication token for uploading coverage reports to Codecov. | `pr-checks.yaml` | Managed in Codecov. |

## Kubernetes Cluster Secrets

These secrets are managed externally and must be applied manually or via a secret management tool before the Helm chart or ArgoCD can successfully deploy the workloads.

See `helm/environments/production/secrets.yaml.example` for details on how to format the primary `reviewsentinel-secrets` Secret containing application runtime configuration (e.g., PostgreSQL passwords, JWT keys).

### ArgoCD Secrets

| Secret Name | Namespace | Purpose |
|-------------|-----------|---------|
| `reviewsentinel-repo-creds` | `argocd` | Stores the Git token (often the same as `GITOPS_PAT`) so ArgoCD can fetch private repository contents. Requires the label `argocd.argoproj.io/secret-type=repository`. |
| `argocd-notifications-secret` | `argocd` | Stores the Slack Bot Token used by `argocd-notifications-cm` for dispatching sync status alerts. |
