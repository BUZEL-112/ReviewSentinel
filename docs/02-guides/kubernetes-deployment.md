# Deploying to Kind

This runbook details how to provision a local Kubernetes cluster using Kind (Kubernetes in Docker) and deploy the ReviewSentinel stack to it. This is the recommended environment for testing Kubernetes manifests and CI/CD pipelines locally.

See [ADR 006](../decisions/006-kind-for-local-k8s.md) for why we use Kind instead of Minikube.

---

## 1. Prerequisites

1. Docker installed and running.
2. `kubectl` installed (v1.28+).
3. `kind` installed (v0.20+).
4. `make` installed.

---

## 2. Provision the Cluster

We provide a Makefile target that builds the cluster, applies a custom configuration to map host ports, and mounts your local source directories into the cluster nodes.

```bash
make up
```

**What `make up` does:**
1. Calls `kind create cluster --config k8s/kind-config.yaml`
2. The config maps localhost port `30080` to the cluster's NodePort.
3. The config mounts your local `data/` and `artifacts/` directories into the Kind node at `/data` and `/artifacts`.
4. Applies all manifests in the `k8s/` directory in order.

---

## 3. Verify Deployment

Kubernetes deployments are asynchronous. The pods will take a minute or two to pull images and start. Check their status:

```bash
kubectl get pods -w
```

You are looking for all pods to reach the `Running` state, and the `prefect-init` job to reach the `Completed` state.

### Verifying MinIO Init
MinIO requires an initialization job to create the artifact buckets. If this fails, training will crash.
```bash
kubectl get jobs
# minio-init should show completions: 1/1
```

---

## 4. Port Forwarding

The API is exposed via a NodePort and is accessible directly at `http://localhost:30080/api`.

All other internal services require port forwarding:

**MLflow UI:**
```bash
kubectl port-forward svc/mlflow 5000:5000
# Access at http://localhost:5000
```

**Prefect UI:**
```bash
kubectl port-forward svc/prefect-server 4200:4200
# Access at http://localhost:4200
```

**MinIO Console:**
```bash
kubectl port-forward svc/minio 9001:9001
# Access at http://localhost:9001
```

---

## 5. Teardown

To destroy the cluster completely (this deletes all data not saved to the mounted `data/` or `artifacts/` directories):

```bash
make down
```

Which executes `kind delete cluster --name reviewsentinel`.

---

## Common Issues

**API pod crashlooping:**
If you have never trained a model, the API will return 503s but it should *not* crashloop. If the pod is restarting constantly, check its logs: `kubectl logs -l app=api`. It likely cannot reach MLflow.

**Prefect worker cannot find files:**
The `prefect-worker` pod relies on `hostPath` mounts to access your local `data/` and `artifacts/` directories. If you are on macOS/Windows and Docker Desktop does not have permission to share the repository directory, these mounts will be empty inside the cluster, and training will fail with `FileNotFoundError`. Ensure file sharing is enabled in Docker Desktop settings.
