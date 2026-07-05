# Component Diagram

ReviewSentinel can be deployed in two topologies: **Docker Compose** (local development) and **Kubernetes via Kind** (local cluster). Both topologies run the same services, but differ in service discovery, storage, and secrets management.

---

## Docker Compose Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                         Host Machine                            │
│                                                                 │
│  ┌────────────┐   HTTP :80    ┌──────────────────────────────┐  │
│  │  Browser / │──────────────►│           Nginx              │  │
│  │  API Client│               │    (reverse proxy)           │  │
│  └────────────┘               └──────────┬───────────────────┘  │
│                                          │ /api/* → :8000        │
│                               ┌──────────▼───────────────────┐  │
│                               │          FastAPI             │  │
│                               │     (uvicorn :8000)          │  │
│                               └──┬───────┬──────────┬────────┘  │
│                      disk r/w    │       │          │disk r/w   │
│                   (artifacts/)   │       │          │(queue.db) │
│                        ┌─────────▼─┐  ┌──▼────┐  ┌─▼────────┐  │
│                        │DistilBERT │  │ FAISS │  │  SQLite  │  │
│                        │  weights  │  │ Index │  │  Queue   │  │
│                        └───────────┘  └───────┘  └──────────┘  │
│                                                                 │
│  ┌─────────────┐  HTTP :5000   ┌─────────────────────────────┐  │
│  │   MLflow    │◄──────────────│      Prefect Worker         │  │
│  │   server    │               │  (runs training/monitor/    │  │
│  └──────┬──────┘               │   judge flows)              │  │
│         │ S3 protocol          └──────────────────────────────┘  │
│  ┌──────▼──────┐  HTTP :4200   ┌─────────────────────────────┐  │
│  │    MinIO    │               │      Prefect Server         │  │
│  │ :9000/:9001 │               │         :4200               │  │
│  └─────────────┘               └─────────────────────────────┘  │
│                                                                 │
│                        HTTP :11434   ┌─────────────────────┐    │
│                               ┌──────►      Ollama          │    │
│                               │      │  (Mistral 7B :11434)│    │
│                     [Prefect  │      └─────────────────────┘    │
│                      Worker]──┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Docker Compose Service Names and Ports

| Service | Internal hostname | External port | Config file |
|---------|------------------|---------------|-------------|
| FastAPI | `api` | `8000` | `docker/api/` |
| Nginx | `nginx` | `80` | `docker/nginx/` |
| MLflow | `mlflow` | `5000` | `docker/mlflow/` |
| Prefect Server | `prefect-server` | `4200` | `docker/prefect/` |
| Prefect Worker | `prefect-worker` | — (no inbound) | `docker/prefect/` |
| MinIO | `minio` | `9000` (API), `9001` (console) | `docker/minio/` |
| Ollama | `ollama` | `11434` | inline in compose |

Service-to-service communication uses Docker's internal network DNS. For example, MLflow's tracking URI inside containers is `http://mlflow:5000` — this is why running training scripts directly on the host (outside Docker) requires falling back to `http://localhost:5000`.

### Persistent Storage (Docker Compose)

| Volume name | Mounted by | Contains |
|-------------|-----------|---------|
| `mlflow-data` | mlflow | MLflow SQLite metadata DB |
| `minio-data` | minio | All MLflow artifact objects (model weights, tokeniser) |
| `prefect-data` | prefect-server | Prefect SQLite DB (run history, deployments) |

`artifacts/` and `data/` directories are bind-mounted from the host filesystem, not named volumes. This means local training output persists across `docker-compose down` and is directly accessible without `docker exec`.

---

## Kubernetes (Kind) Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kind Cluster (Docker)                      │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    default namespace                       │ │
│  │                                                            │ │
│  │  ┌──────────────┐  NodePort :30080  ┌───────────────────┐ │ │
│  │  │ nginx-pod    │◄──────────────────│  (host browser)   │ │ │
│  │  │ ClusterIP svc│──/api/*──────────►│                   │ │ │
│  │  └──────┬───────┘                   └───────────────────┘ │ │
│  │         │                                                  │ │
│  │  ┌──────▼───────┐  ┌────────────┐  ┌────────────────────┐ │ │
│  │  │   api-pod    │  │mlflow-pod  │  │ prefect-server-pod │ │ │
│  │  │  ClusterIP   │  │ClusterIP   │  │    ClusterIP       │ │ │
│  │  │   :8000      │  │ :5000      │  │     :4200          │ │ │
│  │  └──────────────┘  └─────┬──────┘  └────────────────────┘ │ │
│  │                          │S3                               │ │
│  │  ┌───────────────────┐   │         ┌────────────────────┐ │ │
│  │  │  prefect-worker   │───┘         │   minio-pod        │ │ │
│  │  │  (no service)     │◄────────────│   ClusterIP        │ │ │
│  │  └───────────────────┘             │  :9000/:9001       │ │ │
│  │                                    └────────────────────┘ │ │
│  │  ┌───────────────────┐                                     │ │
│  │  │   ollama-pod      │                                     │ │
│  │  │   ClusterIP :11434│                                     │ │
│  │  └───────────────────┘                                     │ │
│  │                                                            │ │
│  │  PersistentVolumeClaims:                                   │ │
│  │  mlflow-data-pvc  ── hostPath: /mlflow                     │ │
│  │  prefect-data-pvc ── hostPath: /root/.prefect              │ │
│  │  minio-data-pvc   ── hostPath: /minio                      │ │
│  │                                                            │ │
│  │  hostPath mounts (via Kind node → host):                   │ │
│  │  ${PROJECT_ROOT}/data      → /data      (in worker pod)    │ │
│  │  ${PROJECT_ROOT}/artifacts → /artifacts (in worker pod)    │ │
│  │  ${PROJECT_ROOT}/configs   → /configs   (in worker pod)    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Kubernetes Manifest Files

| File | Contents |
|------|---------|
| [`k8s/01-base.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/k8s/01-base.yaml) | Namespace, ConfigMap, Secrets, PVCs |
| [`k8s/02-minio.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/k8s/02-minio.yaml) | MinIO Deployment + Service + init Job |
| [`k8s/03-core.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/k8s/03-core.yaml) | MLflow, Prefect Server, Prefect Worker, prefect-init Job |
| [`k8s/04-app.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/k8s/04-app.yaml) | FastAPI Deployment + Service + Nginx Deployment + NodePort Service |
| [`k8s/05-ollama.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/k8s/05-ollama.yaml) | Ollama Deployment + Service |

### Key Differences: Docker Compose vs. Kubernetes

| Aspect | Docker Compose | Kubernetes (Kind) |
|--------|---------------|-------------------|
| **Service discovery** | Docker DNS (`mlflow`, `ollama`) | Kubernetes Service DNS (same names) |
| **Secrets** | `.env` file on host | `reviewsentinel-secrets` K8s Secret |
| **Config** | `docker-compose.yaml` env vars | `reviewsentinel-configs` ConfigMap |
| **External access** | Direct port bindings | NodePort `:30080` for Nginx |
| **Storage** | Named volumes + bind mounts | PVCs (hostPath) + hostPath mounts |
| **Prefect init** | docker-compose `depends_on` | Kubernetes `prefect-init` Job |
| **Image source** | Built locally | `ghcr.io/buzel-112/reviewsentinel-*:latest` |

> [!NOTE]
> The Kubernetes deployment uses pre-built images from GHCR. If you modify source code, you must rebuild and push the images before the changes take effect in the cluster. The Docker Compose setup builds images locally on `docker-compose up --build`.

### Accessing Services in Kind

The Kubernetes cluster exposes one external NodePort: Nginx on `:30080`. All other services require `kubectl port-forward`:

```bash
# FastAPI direct
kubectl port-forward svc/api 8000:8000

# MLflow UI
kubectl port-forward svc/mlflow 5000:5000

# Prefect UI
kubectl port-forward svc/prefect-server 4200:4200

# MinIO console
kubectl port-forward svc/minio 9001:9001
```
