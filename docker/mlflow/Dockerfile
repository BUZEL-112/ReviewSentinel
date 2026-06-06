FROM python:3.10-slim

# Install MLflow and S3/MinIO support
RUN pip install --no-cache-dir mlflow boto3
RUN apt-get update && apt-get install -y curl
EXPOSE 5000

# Start MLflow server using SQLite as backend and MinIO as artifact root
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://reviewsentinel-artifacts/mlflow \
    --host 0.0.0.0 \
    --port 5000 \
    --allowed-hosts mlflow:5000,mlflow,localhost:5000,127.0.0.1
