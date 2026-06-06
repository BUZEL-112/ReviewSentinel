#!/bin/bash
set -e

echo "Waiting for MLflow to be ready..."
# A simple wait loop checking the MLflow health endpoint
until curl -s http://mlflow:5000/health > /dev/null; do
  echo "MLflow is unavailable - sleeping"
  sleep 2
done

echo "MLflow is up - executing command"
# Start uvicorn
exec uvicorn src.api.api:app --host 0.0.0.0 --port 8000 --workers 2 "$@"
