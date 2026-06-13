#!/bin/bash
echo "Patching generated K8s manifests for ReviewSentinel..."

# 1. Inject Prefect Execution Arguments
sed -i '/image: .*prefect-server/a \          args: ["server"]' k8s/base/prefect-server-deployment.yaml
sed -i '/image: .*prefect-worker/a \          args: ["worker"]' k8s/base/prefect-worker-deployment.yaml

# 2. Inject Watchdog Internal DNS Routing
sed -i '/PREFECT_API_URL/a \            - name: API_URL\n              value: "http://api:8000"\n            - name: API_HOST\n              value: "http://api:8000"\n            - name: REVIEW_API_URL\n              value: "http://api:8000"' k8s/base/prefect-worker-deployment.yaml

# 3. Strip empty config volume mounts hiding the baked-in files
sed -i '/name: .*configs/,/mountPath: \/app\/configs/d' k8s/base/*deployment.yaml

# 4. Expose Nginx Gateway to the Public IP
sed -i 's/type: ClusterIP/type: NodePort/' k8s/base/nginx-service.yaml

# 5. Fix Nginx Deployment (Remove hostPort, inject ConfigMap mounts)
sed -i '/hostPort: 80/d' k8s/base/nginx-deployment.yaml
sed -i '/image: .*nginx/a \          volumeMounts:\n            - name: nginx-config-volume\n              mountPath: /etc/nginx/nginx.conf\n              subPath: nginx.conf' k8s/base/nginx-deployment.yaml
sed -i '/containers:/i \      volumes:\n        - name: nginx-config-volume\n          configMap:\n            name: nginx-config' k8s/base/nginx-deployment.yaml

# 6. Convert minio-init script from a Deployment into a single-run Job
sed -i 's/kind: Deployment/kind: Job/' k8s/base/minio-init-deployment.yaml
sed -i 's/apiVersion: apps\/v1/apiVersion: batch\/v1/' k8s/base/minio-init-deployment.yaml
sed -i '/replicas: 1/d' k8s/base/minio-init-deployment.yaml
sed -i '/containers:/i \      restartPolicy: OnFailure' k8s/base/minio-init-deployment.yaml

echo "All K8s base files successfully patched."
