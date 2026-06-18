# Operational Runbooks

This document contains standard operating procedures for the ReviewSentinel cluster.

---

## 1. Rotating Secrets Without Downtime

When `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` for MinIO must be rotated:

1. **Update MinIO via CLI:**
   ```bash
   mc alias set myminio http://localhost:9001 $OLD_USER $OLD_PASS
   mc admin user add myminio $NEW_USER $NEW_PASS
   mc admin policy attach myminio consoleAdmin --user $NEW_USER
   ```
2. **Update Kubernetes Secret:**
   Update the `reviewsentinel-secrets` object in Kubernetes with the new credentials.
   ```bash
   kubectl apply -f new-secret.yaml
   ```
3. **Roll Deployments:**
   The `prefect-worker` pod reads the secret as environment variables on boot. You must restart the deployment to pick up the new secret.
   ```bash
   kubectl rollout restart deployment prefect-worker
   ```
4. **Decommission old user:**
   Wait 10 minutes to ensure no active training jobs are using the old credentials, then:
   ```bash
   mc admin user disable myminio $OLD_USER
   ```

---

## 2. Manually Triggering a Training Flow

If you need to force a model retrain (e.g., after modifying `config.yaml`):

### Option A: Prefect UI
1. Open the Prefect UI (`http://localhost:4200`).
2. Navigate to **Deployments** > **ReviewSentinel Training Pipeline**.
3. Click **Run** > **Quick Run**.

### Option B: CLI (Inside Worker Container)
If Prefect Server is down or unreachable, you can bypass the scheduler and run the flow directly in the worker context:
```bash
# Docker Compose
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py

# Kubernetes
kubectl exec -it deployment/prefect-worker -- python scripts/run_flow.py
```

---

## 3. Verifying the LLM Judge Queue is Draining

The LLM Judge processes uncertain predictions in batches every 4 hours. To verify it is not falling behind:

1. Connect to the SQLite queue database:
   ```bash
   sqlite3 artifacts/llm_judge/review_queue.db
   ```
2. Check the pending count:
   ```sql
   SELECT COUNT(*) FROM queue WHERE status = 'pending';
   ```
3. If the count exceeds your daily uncertainty volume (e.g., > 1000), check the `prefect-worker` logs. The Ollama container may be down or timing out.

---

## 4. Scaling the API Deployment

If prediction latency increases under load, you can scale the FastAPI stateless deployment.

**Kubernetes:**
```bash
kubectl scale deployment api --replicas=3
```

**Docker Compose:**
```bash
docker-compose -f docker/docker-compose.yaml up -d --scale api=3
```

*(Note: Nginx automatically load-balances across the scaled API replicas).*

---

## 5. Rolling Back a Failed Deployment

If a newly deployed API container is failing (e.g., crashing on boot due to a missing dependency):

```bash
kubectl rollout undo deployment api
```
This instantly reverts the `api` Deployment to the previous ReplicaSet.
