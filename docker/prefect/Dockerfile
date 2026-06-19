FROM python:3.10-slim

WORKDIR /app

# Install dependencies including Prefect and boto3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt boto3
# RUN pip install -r requirements.txt boto3
RUN apt-get update && apt-get install -y curl
COPY . .

# Copy and set permissions for entrypoint
COPY docker/prefect/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/app/entrypoint.sh"]
