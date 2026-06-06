#!/bin/sh
sleep 2
# Use a custom alias to bypass any cached default credentials
mc alias set myminio http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc mb myminio/reviewsentinel-artifacts || echo "Bucket may already exist"
