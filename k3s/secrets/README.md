# Secrets

Required secrets for the ReviewSentinel cluster.

## Creating secrets

Ensure `.env` is populated at the project root, then:

    ./k8s/secrets/create-secrets.sh

## Required variables in .env

- MINIO_ROOT_USER
- MINIO_ROOT_PASSWORD  
- HCLOUD_TOKEN

## Secrets created

- `reviewsentinel-secrets` — main application credentials
- `mlflow-secrets` — MLflow configuration

## Production note

For production, use sealed-secrets or Vault + the vault-agent-injector instead of
this script approach. Sealed Secrets encrypts Secrets with a cluster key and allows
committing the encrypted form to git.
