# ADR 005: MinIO over AWS S3

**Date:** June 2026  
**Status:** Accepted

## Context

MLflow requires an "artifact store" to save the actual file outputs of a training run (the gigabytes of model weights, the tokeniser vocabularies, and tensorboard logs). 

While MLflow can use the local filesystem (`file:///`), this breaks down as soon as the system is distributed. A Prefect Worker running in one Kubernetes Pod cannot easily write files to a filesystem that the FastAPI application in another Pod can read, unless a persistent shared network volume (like NFS) is configured, which is operationally brittle.

We needed a centralized object store. The industry standard is AWS S3.

## Decision

We chose **MinIO** running as a local service within our cluster/docker-compose, rather than provision a real AWS S3 bucket.

## Rationale

1. **S3 Compatibility:** MinIO implements the exact AWS S3 API. Python tools (like `boto3` and `mlflow`) do not know they are talking to MinIO instead of AWS. They just require the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and an overriding `ENDPOINT_URL`.
2. **Cost & Portability:** Provisioning real AWS S3 buckets for local development or CI/CD pipelines incurs cost and requires managing cloud IAM credentials for every developer. MinIO allows developers to run the entire stack locally, disconnected from the internet, at zero cost.
3. **Migration Path:** Because MinIO is API-compatible, migrating to real AWS S3 in the future requires exactly zero code changes. We only need to change the `MLFLOW_S3_ENDPOINT_URL` environment variable to point to AWS.

## Consequences

- **Positive:** True local development parity with production. No cloud bills for CI/CD artifact storage.
- **Negative:** We must manage the storage state of the MinIO container. In Kubernetes, this means managing a PersistentVolumeClaim (PVC).
- **Negative:** MinIO initialization can be flaky on first boot (bucket creation race conditions), which requires operational workarounds (like the `minio-init` Job in our Kubernetes manifests).
