#!/bin/bash
set -euo pipefail

# Load environment variables from project root .env file
source "$(git rev-parse --show-toplevel)/.env"

NAMESPACE="reviewsentinel"

echo "Creating Kubernetes secrets in namespace: $NAMESPACE"

# Main application secrets
kubectl create secret generic reviewsentinel-secrets \
  --namespace="$NAMESPACE" \
  --from-literal=minio-root-user="$MINIO_ROOT_USER" \
  --from-literal=minio-root-password="$MINIO_ROOT_PASSWORD" \
  --dry-run=client -o yaml | kubectl apply -f -

# MLflow database credentials (if using Postgres instead of SQLite)
kubectl create secret generic mlflow-secrets \
  --namespace="$NAMESPACE" \
  --from-literal=tracking-uri="http://mlflow:5000" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secrets created successfully."
echo "Verify with: kubectl get secrets -n $NAMESPACE"