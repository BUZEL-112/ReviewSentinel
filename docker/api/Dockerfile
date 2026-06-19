# Use official PyTorch image as base to avoid compiling heavy ML packages
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 1. Set Timezone explicitly to prevent Pendulum/Prefect crashes
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 2. Add tzdata to your apt-get install
RUN apt-get update && apt-get install -y --no-install-recommends curl tzdata && rm -rf /var/lib/apt/lists/*

# Copy the custom API requirements
COPY docker/api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Copy and set permissions for entrypoint
COPY docker/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app
ENTRYPOINT ["/app/entrypoint.sh"]