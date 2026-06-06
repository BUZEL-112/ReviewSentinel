#!/bin/bash
set -e

# If this container is running the worker
if [ "$1" = "worker" ]; then
    echo "Waiting for Prefect server to be ready..."
    # The URL defaults to prefect-server:4200/api
    until curl -s $PREFECT_API_URL/health > /dev/null; do
        echo "Prefect server is unavailable - sleeping"
        sleep 2
    done

    echo "Creating Prefect work pool..."
    prefect work-pool create default-agent-pool --type process || true

    echo "Registering Prefect deployments..."
    prefect deploy --all

    echo "Starting Prefect worker..."
    exec prefect worker start --pool default-agent-pool

# If this container is running the server
elif [ "$1" = "server" ]; then
    echo "Starting Prefect server..."
    exec prefect server start --host 0.0.0.0
else
    # Allow running arbitrary commands (like bash)
    exec "$@"
fi
