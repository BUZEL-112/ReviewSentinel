# ADR 006: Kind for Local Kubernetes

**Date:** June 2026  
**Status:** Accepted

## Context

We provide Kubernetes manifests (`k8s/`) so developers can test the deployment topology before pushing to staging or production. We needed a recommended tool for developers to spin up a local Kubernetes cluster on their laptops.

Options evaluated:
1. **Minikube:** The traditional local Kubernetes tool. Runs via virtual machines (VirtualBox, HyperKit).
2. **Docker Desktop Kubernetes:** Built into Docker Desktop for Mac/Windows.
3. **k3s:** A lightweight Kubernetes distribution by Rancher.
4. **Kind (Kubernetes in Docker):** Runs local Kubernetes clusters using Docker container "nodes".

## Decision

We chose **Kind** as the officially supported local Kubernetes environment.

## Rationale

1. **CI/CD Parity:** Kind was originally built to test Kubernetes itself in CI environments. It runs flawlessly inside GitHub Actions without complex nested virtualization. By standardizing on Kind, our local `make up` cluster behaves identically to the PR validation cluster.
2. **No VMs Required:** Unlike Minikube (which often requires managing a VM hypervisor), Kind just requires Docker. 
3. **HostPath Mounts:** Our Prefect workers require access to the local `data/` and `artifacts/` directories to run training without a remote cloud volume. Kind makes mounting host directories into the cluster nodes incredibly easy via the `kind-config.yaml` file `extraMounts` directive. Minikube makes this notably painful.
4. **Speed:** Spinning up and tearing down a Kind cluster takes seconds, making it ideal for clean-slate testing.

## Consequences

- **Positive:** We have a single `make up` command that works consistently across Mac, Linux, and CI runners.
- **Negative:** Kind requires Docker. Developers using Podman or other container runtimes may face compatibility issues (though Podman support for Kind is improving).
- **Negative:** Kind is explicitly **not** for production. Our Kubernetes manifests are optimized for Kind and may require modification (specifically the `hostPath` PersistentVolumes) before deployment to a managed cloud Kubernetes service like EKS or GKE.
