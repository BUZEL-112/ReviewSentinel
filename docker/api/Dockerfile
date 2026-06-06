# Use a multi-stage build to keep the final image clean
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install boto3

FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN apt-get update && apt-get install -y curl
# Copy and set permissions for entrypoint
COPY docker/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/app/entrypoint.sh"]
