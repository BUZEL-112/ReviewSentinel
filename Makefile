# Variables
PROJECT_ROOT := $(CURDIR)
CLUSTER_NAME := kind

# Phony targets are commands that don't represent physical files
.PHONY: help config up down clean apply logs

help:
	@echo "🚀 ReviewSentinel Infrastructure Commands"
	@echo "---------------------------------------"
	@echo "  make up      - Generate config, start cluster, and apply manifests"
	@echo "  make down    - Tear down the Kind cluster"
	@echo "  make clean   - Remove generated configuration files"
	@echo "  make apply   - Apply Kubernetes manifests (k8s/)"
	@echo "  make logs    - Watch the status of all pods"

config:
	@echo "📂 Generating cluster config for path: $(PROJECT_ROOT)"
	@PROJECT_ROOT=$(PROJECT_ROOT) envsubst < kind-config.yaml.template > kind-config.yaml

up: clean config down
	@echo "🏗️  Creating new cluster..."
	@kind create cluster --config kind-config.yaml
	@echo "✅ Cluster is ready. Applying manifests..."
	@kubectl apply -f k8s/
	@echo "🎉 Deployment complete! Run 'make logs' to watch the startup."

down:
	@echo "🗑️  Tearing down cluster..."
	@kind delete cluster --name $(CLUSTER_NAME) 2>/dev/null || true

clean:
	@echo "🧹 Cleaning up generated files..."
	@rm -f kind-config.yaml

apply:
	@kubectl apply -f k8s/

logs:
	@kubectl get pods -w


forward-all:
	kubectl port-forward svc/prefect-server 4200:4200 &
	kubectl port-forward svc/mlflow 5000:5000 &
	kubectl port-forward svc/minio 9001:9001 &
	kubectl port-forward svc/api 8000:8000 > /dev/null 2>&1 &
	
stop-all:
	pkill -f "kubectl port-forward"
# .PHONY: tf-vars tf-init tf-plan tf-apply tf-destroy tf-output tf-kubeconfig install-kubectl install-helm k8s-setup k8s-deploy k8s-status k8s-logs k8s-shell k8s-secrets

# # Auto-resolve KUBECONFIG — use env var if already set, else default to project root kubeconfig.
# # Every kubectl call in every target works without needing 'export KUBECONFIG=...' in your shell.
# export KUBECONFIG ?= $(CURDIR)/kubeconfig

# # Safe first-time setup: only copies the example if terraform.tfvars does NOT already exist.
# # This prevents accidentally overwriting real credentials with placeholder values.
# tf-vars:
# 	@if [ ! -f terraform/environments/production/terraform.tfvars ]; then \
# 		cp terraform/environments/production/terraform.tfvars.example \
# 		   terraform/environments/production/terraform.tfvars; \
# 		echo "Created terraform.tfvars from example — edit it with your real values."; \
# 	else \
# 		echo "terraform.tfvars already exists — skipping copy to protect your credentials."; \
# 	fi

# tf-init:
# 	cd terraform/environments/production && terraform init

# tf-plan:
# 	cd terraform/environments/production && terraform plan

# tf-apply:
# 	cd terraform/environments/production && terraform apply -auto-approve

# tf-destroy:
# 	./terraform/scripts/destroy-confirm.sh

# tf-output:
# 	cd terraform/environments/production && terraform output

# # install-kubectl: downloads and installs the latest stable kubectl to /usr/local/bin.
# # Safe to run multiple times — skips if kubectl is already present.
# install-kubectl:
# 	@if ! command -v kubectl &>/dev/null; then \
# 		echo "Installing kubectl..."; \
# 		KUBE_VER=$$(curl -Ls https://dl.k8s.io/release/stable.txt); \
# 		curl -Lo /tmp/kubectl "https://dl.k8s.io/release/$$KUBE_VER/bin/linux/amd64/kubectl"; \
# 		install -o root -g root -m 0755 /tmp/kubectl /usr/local/bin/kubectl; \
# 		rm /tmp/kubectl; \
# 		echo "kubectl $$(kubectl version --client --short 2>/dev/null) installed."; \
# 	else \
# 		echo "kubectl already installed: $$(kubectl version --client --short 2>/dev/null)"; \
# 	fi

# # install-helm: downloads and installs Helm 3. Safe to run multiple times.
# install-helm:
# 	@if ! command -v helm &>/dev/null; then \
# 		echo "Installing Helm 3..."; \
# 		curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash; \
# 		echo "Helm $$(helm version --short) installed."; \
# 	else \
# 		echo "helm already installed: $$(helm version --short)"; \
# 	fi

# tf-kubeconfig: install-kubectl
# 	@chmod +x terraform/scripts/get-kubeconfig.sh
# 	./terraform/scripts/get-kubeconfig.sh
# 	@echo ""
# 	@echo "=== Testing cluster connection ==="
# 	@KUBECONFIG=$$PWD/kubeconfig kubectl get nodes || echo "Cluster not ready yet — retry in 30s"
# 	@echo ""
# 	@echo "To use kubectl in your shell, run:"
# 	@echo "  export KUBECONFIG=$$PWD/kubeconfig"

# k8s-setup: install-helm
# 	@echo "=== Step 1/5: Ensuring script permissions ==="
# 	@find . -name "*.sh" -not -path "*/venv/*" -exec chmod +x {} \;
# 	@echo "=== Step 2/5: Creating namespaces ==="
# 	kubectl apply -f k8s/namespaces/
# 	@echo "=== Step 3/5: Creating secrets ==="
# 	./k8s/secrets/create-secrets.sh
# 	@echo "=== Step 4/5: Adding Helm repos ==="
# 	helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx || true
# 	helm repo add jetstack https://charts.jetstack.io || true
# 	helm repo update
# 	@echo "=== Step 5/5: Installing ingress-nginx and cert-manager ==="
# 	helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
# 		--namespace ingress-nginx --create-namespace \
# 		--values k8s/cluster/ingress/ingress-nginx-values.yaml \
# 		--wait --timeout 3m
# 	helm upgrade --install cert-manager jetstack/cert-manager \
# 		--namespace cert-manager --create-namespace \
# 		--version v1.14.4 --set installCRDs=true \
# 		--wait --timeout 3m
# 	kubectl apply -f k8s/cluster/cert-manager/issuer.yaml
# 	@echo "=== k8s-setup complete ==="

# k8s-deploy:
# 	kubectl apply -k k8s/overlays/production/

# k8s-status:
# 	kubectl get all -n reviewsentinel
# 	kubectl get ingress -n reviewsentinel
# 	kubectl get pvc -n reviewsentinel

# k8s-logs:
# 	kubectl logs -n reviewsentinel -l app=reviewsentinel-api --tail=100 -f

# k8s-shell:
# 	kubectl exec -it -n reviewsentinel \
# 		$$(kubectl get pod -n reviewsentinel -l app=reviewsentinel-api -o jsonpath='{.items[0].metadata.name}') \
# 		-- /bin/bash

# k8s-secrets:
# 	./k8s/secrets/create-secrets.sh

# HELM_CHART := helm/reviewsentinel
# RELEASE_NAME := reviewsentinel
# NAMESPACE := reviewsentinel

# .PHONY: helm-lint helm-template helm-install helm-upgrade helm-diff helm-status helm-rollback helm-uninstall

# helm-lint:
# 	helm lint $(HELM_CHART) -f helm/environments/production/values.yaml

# helm-template:
# 	helm template $(RELEASE_NAME) $(HELM_CHART) \
# 	  -f helm/environments/production/values.yaml \
# 	  --namespace $(NAMESPACE)

# helm-install:
# 	helm install $(RELEASE_NAME) $(HELM_CHART) \
# 	  -f helm/environments/production/values.yaml \
# 	  --namespace $(NAMESPACE) \
# 	  --create-namespace \
# 	  --wait

# helm-upgrade:
# 	helm upgrade $(RELEASE_NAME) $(HELM_CHART) \
# 	  -f helm/environments/production/values.yaml \
# 	  --namespace $(NAMESPACE) \
# 	  --wait \
# 	  --atomic         # rolls back automatically if upgrade fails

# helm-diff:
# 	helm diff upgrade $(RELEASE_NAME) $(HELM_CHART) \
# 	  -f helm/environments/production/values.yaml \
# 	  --namespace $(NAMESPACE)

# helm-status:
# 	helm status $(RELEASE_NAME) --namespace $(NAMESPACE)

# helm-rollback:
# 	helm rollback $(RELEASE_NAME) --namespace $(NAMESPACE)

# helm-uninstall:
# 	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)
# 	@echo "Note: PVCs with helm.sh/resource-policy: keep are NOT deleted."
# 	@echo "Delete manually with: kubectl delete pvc -l app.kubernetes.io/instance=$(RELEASE_NAME) -n $(NAMESPACE)"

# ARGOCD_NAMESPACE := argocd
# # Use --port-forward so the CLI tunnels directly to the argocd-server pod.
# # argocd.127.0.0.1.nip.io resolves to 127.0.0.1 (loopback), not the ArgoCD pod,
# # so connecting via the ingress hostname fails when running on the same host as k3s.
# ARGOCD_PF_OPTS := --port-forward --port-forward-namespace $(ARGOCD_NAMESPACE) --grpc-web

# .PHONY: install-argocd-cli argocd-bootstrap argocd-login argocd-status argocd-sync argocd-diff argocd-rollback argocd-password argocd-apply-apps

# # install-argocd-cli: downloads the argocd CLI matching the server version. Safe to run multiple times.
# install-argocd-cli:
# 	@if ! command -v argocd &>/dev/null; then \
# 		echo "Installing ArgoCD CLI..."; \
# 		ARGOCD_VER=$$(curl -Ls https://raw.githubusercontent.com/argoproj/argo-cd/stable/VERSION); \
# 		curl -sL -o /tmp/argocd "https://github.com/argoproj/argo-cd/releases/download/v$${ARGOCD_VER}/argocd-linux-amd64"; \
# 		install -o root -g root -m 0755 /tmp/argocd /usr/local/bin/argocd; \
# 		rm /tmp/argocd; \
# 		echo "argocd $$(argocd version --client --short 2>/dev/null) installed."; \
# 	else \
# 		echo "argocd CLI already installed: $$(argocd version --client --short 2>/dev/null)"; \
# 	fi

# argocd-bootstrap: install-argocd-cli
# 	@chmod +x gitops/argocd/install/bootstrap.sh
# 	./gitops/argocd/install/bootstrap.sh

# argocd-login:
# 	@PASS=$$(kubectl -n $(ARGOCD_NAMESPACE) get secret argocd-initial-admin-secret \
# 	    -o jsonpath="{.data.password}" 2>/dev/null | base64 -d); \
# 	if [ -z "$$PASS" ]; then echo "ERROR: argocd-initial-admin-secret not found. Run: make argocd-bootstrap"; exit 1; fi; \
# 	echo "Starting port-forward to argocd-server:80 on localhost:18080..."; \
# 	kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & \
# 	PF_PID=$$!; \
# 	sleep 2; \
# 	argocd login localhost:18080 \
# 	  --username admin \
# 	  --password "$$PASS" \
# 	  --plaintext; \
# 	kill $$PF_PID 2>/dev/null; true

# argocd-status:
# 	@kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & PF=$$!; sleep 2; \
# 	argocd app list --server localhost:18080 --plaintext; \
# 	argocd app get reviewsentinel-production --server localhost:18080 --plaintext 2>/dev/null || true; \
# 	kill $$PF 2>/dev/null; true

# argocd-sync:
# 	@kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & PF=$$!; sleep 2; \
# 	argocd app sync reviewsentinel-production --prune --server localhost:18080 --plaintext; \
# 	kill $$PF 2>/dev/null; true

# argocd-diff:
# 	@kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & PF=$$!; sleep 2; \
# 	argocd app diff reviewsentinel-production --server localhost:18080 --plaintext; \
# 	kill $$PF 2>/dev/null; true

# argocd-rollback:
# 	@kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & PF=$$!; sleep 2; \
# 	argocd app history reviewsentinel-production --server localhost:18080 --plaintext; \
# 	kill $$PF 2>/dev/null; true
# 	@read -p "Enter revision number to rollback to: " rev; \
# 	kubectl port-forward svc/argocd-server -n $(ARGOCD_NAMESPACE) 18080:80 &>/dev/null & PF=$$!; sleep 2; \
# 	argocd app rollback reviewsentinel-production $$rev --server localhost:18080 --plaintext; \
# 	kill $$PF 2>/dev/null; true

# argocd-password:
# 	@echo "ArgoCD admin password:"
# 	@kubectl -n $(ARGOCD_NAMESPACE) get secret argocd-initial-admin-secret \
# 	  -o jsonpath="{.data.password}" | base64 -d; echo

# argocd-apply-apps:
# 	kubectl apply -f gitops/argocd/projects/
# 	kubectl apply -f gitops/argocd/applications/app-of-apps.yaml

# .PHONY: ci-test ci-lint ci-quality-gate ci-full ci-pr-simulation

# ci-test:
# 	pytest tests/ \
# 	  --cov=src \
# 	  --cov-report=term-missing \
# 	  --cov-fail-under=70 \
# 	  -v

# ci-lint:
# 	ruff check src/ scripts/ --output-format text
# 	ruff format src/ scripts/ --check

# ci-quality-gate:
# 	python scripts/ci/check_model_quality.py \
# 	  --config configs/pipeline_params.yaml \
# 	  --f1-threshold 0.01 \
# 	  --output-format text

# ci-full: ci-lint ci-test ci-quality-gate
# 	@echo "All CI checks passed locally."

# # Simulate what GitHub Actions runs on a PR
# ci-pr-simulation: ci-full helm-lint
# 	@echo "PR simulation complete — safe to push."


