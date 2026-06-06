# ReviewSentinel

## Table of Contents
- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Training Pipeline](#training-pipeline)
- [Inference API](#inference-api)

## Overview

ReviewSentinel is a scalable machine learning pipeline for analysing sentiment of custumer review. It includes a complete MLOps pipeline for data ingestion (HuggingFace Hub streaming), BERT tokenization, model training, evaluation, and inference. The project is designed to be modular, configurable, and extensible.

The primary goal is to demonstrate a production-ready approach to fine-tuning and deploying Transformer-based models. It leverages best practices in software engineering and MLOps to create a robust and maintainable system.

**Key Features:**
- **Modular Pipeline:** The project is broken down into distinct stages: data loading, data cleaning, model training, and evaluation.
- **Configurable:** All aspects of the pipeline, from data sources to model hyperparameters, can be configured through YAML files.
- **Transformer Model:** Fine-tunes `distilbert-base-uncased` for three-class sentiment classification (negative / neutral / positive) using the HuggingFace `Trainer` API.
- **MLflow Integration:** The project is integrated with MLflow for experiment tracking and model management.
- **Inference API:** The project includes a simple API for making predictions on new data.

## Setup Instructions

To set up the project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/FAKER-112/End-to-End-Sentiment-Analysis-Pipeline-for-Customer-Reviews.git
cd End-to-End-Sentiment-Analysis-Pipeline-for-Customer-Reviews
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage the project's dependencies. You can create a virtual environment using `venv`:

```bash
python -m venv venv
source venv/bin/activate  
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Install NLTK Data

The project uses the NLTK library for text processing. You will need to download the required NLTK data. You can do this by running the following command in your terminal:

```bash
python -m nltk.downloader punkt stopwords wordnet
```

### 5. Download Word Embeddings

The project uses pre-trained word embeddings for the deep learning models. These will be downloaded automatically the first time you run the data cleaning pipeline.

## Training Pipeline

The training pipeline is responsible for orchestrating the entire process of training the sentiment analysis models. It is designed to be run from the command line and can be configured through YAML files.

### Running the Training Pipeline

To run the training pipeline, you can execute the following command from the root directory of the project:

```bash
python src/pipeline/train_pipeline.py
```

This will run the entire pipeline, from data ingestion to model evaluation, based on the configuration in `configs/pipeline_params.yaml`.

### Configuration

The training pipeline is configured through the `configs/pipeline_params.yaml` and `configs/config.yaml` files.

**`configs/config.yaml`:** This file contains the main configuration for the project, including data sources, model hyperparameters, and file paths.

- **`data_ingestion`:** You can specify the URL of the dataset to be downloaded in the `source_url` field.
- **`clean_data`:** This section contains parameters for the data cleaning process, such as the test set size and the name of the word embedding model.
- **`model_params`:** This section contains the hyperparameters for each of the models.

**`configs/pipeline_params.yaml`:** This file defines the stages of the training pipeline and the parameters for each stage.

- **`training_pipeline.pipeline.stages`:** This is a list of the stages to be executed, in order. The available stages are `load_data`, `clean_data`, `train_model`, and `evaluate_model`.
- **`training_pipeline.training`:** This section specifies the model to be trained and the path to the main configuration file.
- **`training_pipeline.evaluation`:** This section specifies the directories where the evaluation results and the best model should be saved.

### Pipeline Stages

The training pipeline consists of the following stages:

#### 1. `load_data`
This stage downloads the dataset from the URL specified in the configuration file, unzips it, and saves it as a CSV file.

#### 2. `clean_data`
This stage performs the following preprocessing steps on the raw data:
- **Text Cleaning:** Lowercasing, removing URLs, and removing non-alphabetic characters.
- **Sentiment Labeling:** Labeling reviews as "positive" or "negative" based on the rating.
- **Tokenization:** Splitting the text into individual words.
- **Stopword Removal:** Removing common English stopwords.
- **Lemmatization:** Reducing words to their base form.
- **Vectorization:** Converting the text into numerical vectors using pre-trained word embeddings.

#### 3. `train_model`
This stage trains the specified model on the preprocessed data. The project supports the following models:
- `logistic_regression`
- `lstm`
- `cnn`
- `cnn_lstm`

The model to be trained can be specified in the `configs/pipeline_params.yaml` file.

#### 4. `evaluate_model`
This stage evaluates the trained model on the test set and saves the evaluation metrics to a JSON file. The metrics calculated are:
- Accuracy
- Precision
- Recall
- F1-score

The best performing model is also saved to the `artifacts/best_model` directory.

### MLflow Integration

The training pipeline is integrated with MLflow for experiment tracking. When you run the training pipeline, the following information will be logged to MLflow:

- **Parameters:** The hyperparameters of the model.
- **Metrics:** The evaluation metrics of the model.
- **Artifacts:** The trained model itself.

To view the MLflow UI, you can run the following command from the root directory of the project:

```bash
mlflow ui
```

This will start the MLflow tracking server, which you can access in your web browser at `http://localhost:5000`.

## Inference API

The inference API allows you to use the trained sentiment analysis model to make predictions on new, unseen data. The API is implemented in the `src/pipeline/inference_pipeline.py` module.

### Running the Inference Pipeline

You can run the inference pipeline from the command line to see examples of single and batch predictions:

```bash
python src/pipeline/inference_pipeline.py
```

This will run the examples in the `if __name__ == "__main__":` block of the script, which demonstrate how to use the `InferencePipeline` class to make predictions.

### Using the `InferencePipeline`

To use the inference pipeline in your own code, you can import the `InferencePipeline` class and call its `run` method.

**Example: Single Prediction**

```python
from src.pipeline.inference_pipeline import InferencePipeline

# Initialize the pipeline
pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

# Make a single prediction
result_df = pipe.run(
    title="This is a great product!",
    text="I am very happy with my purchase.",
    batch_mode=False
)

print(result_df)
```

**Example: Batch Prediction**

```python
from src.pipeline.inference_pipeline import InferencePipeline

# Initialize the pipeline
pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

# Define the batch data
titles = (
    "This is a great product!|||"
    "This is a terrible product."
)

texts = (
    "I am very happy with my purchase.|||"
    "I am very unhappy with my purchase."
)

# Make a batch prediction
batch_df = pipe.run(
    title=titles,
    text=texts,
    batch_mode=True
)

print(batch_df)
```

### Configuration

The inference pipeline is configured through the `configs/inference_pipeline.yaml` file (or `configs/pipeline_params.yaml` if you are using the training pipeline's config).

- **`model_path`:** The path to the trained model to be used for inference. By default, this is set to the best model saved by the training pipeline.
- **`clean_config_path`:** The path to the main configuration file, which is used to configure the data cleaning process.
- **`batch_separator`:** The separator to be used when making batch predictions.

### Input and Output

**Input:**

The `run` method of the `InferencePipeline` class takes the following arguments:

- **`title` (str):** The title of the review(s). For batch predictions, the titles should be separated by the `batch_separator`.
- **`text` (str):** The text of the review(s). For batch predictions, the texts should be separated by the `batch_separator`.
- **`batch_mode` (bool):** Whether to make a single prediction or a batch prediction.

**Output:**

The `run` method returns a pandas DataFrame with the following columns:

- **`title`:** The original title of the review.
- **`text`:** The original text of the review.
- **`full_text`:** The concatenated title and text.
- **`clean_text`:** The preprocessed text.
- **`tokens`:** The tokenized text.
- **`vector` or `sequence`:** The numerical representation of the text.
- **`predicted_label`:** The predicted sentiment of the review ("positive" or "negative").

## Explainability

The core pitch is that a sentiment label alone ("negative") tells you a review is a problem. A SHAP explanation tells you what specifically in the language triggered that classification — whether it was the word "broken", "slow", "expensive", or "disappointed". Product managers can use this to triage feedback: not just how many negative reviews, but what vocabulary clusters are driving them. That's the difference between a metric and an insight.

The API exposes this via the `include_explanation` parameter in single predictions. When enabled, it returns the top words contributing toward or against the prediction.

### Example Request

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Terrible quality",
  "text": "The packaging was broken and the product looks cheap.",
  "include_explanation": true
}'
```

### Example Response

```json
{
  "text": "Terrible quality The packaging was broken and the product looks cheap.",
  "label": "negative",
  "confidence": 0.98,
  "scores": {
    "negative": 0.98,
    "neutral": 0.01,
    "positive": 0.01
  },
  "aspect": null,
  "flags": {
    "short_review": false,
    "low_confidence": false,
    "possible_sarcasm": false
  },
  "processing_time_ms": 45.2,
  "model_version": "unknown",
  "explanation": {
    "predicted_class": "negative",
    "target_class_explained": "negative",
    "baseline_probability": 0.3333,
    "tokens": [
      {
        "token": "broken",
        "shap_value": 0.45,
        "direction": "toward"
      },
      {
        "token": "Terrible",
        "shap_value": 0.38,
        "direction": "toward"
      },
      {
        "token": "cheap",
        "shap_value": 0.25,
        "direction": "toward"
      }
    ]
  }
}
```

## Orchestration

The project uses Prefect for orchestration and Great Expectations for data validation. This enables observable, retryable, schedulable, and auditable production training pipelines.

### Quality Gate

The pipeline includes a production quality gate: after a model is trained and evaluated, its F1 score is compared against the currently deployed model's F1 score (stored as a baseline in MLflow). The model is only deployed (tagged as `is_production="true"` in MLflow) if the F1 score improves by a configurable threshold (e.g., 1 percentage point). This prevents deploying degraded models to production.

### Data Validation

We run Great Expectations to catch data anomalies before training. Key expectations include:
- Required columns schema mismatch
- Null value checks on `rating` and `text` columns
- Target variable bounds checking (ratings must be 1-5)
- Class imbalance detection (prevents severe class skew leading to biased models)

### Usage

**Manual Trigger**
You can trigger the pipeline manually using the provided script:
```bash
python scripts/run_flow.py --config configs/pipeline_params.yaml
```

**Prefect UI**
Start the Prefect server locally to monitor flow runs, logs, and state history:
```bash
prefect server start
```

**Schedule Deployment**
Deploy the flow using the defined schedule in `prefect.yaml`:
```bash
prefect deploy
```

## Running Locally with Docker

This project uses Docker Compose to orchestrate the entire pipeline: FastAPI, MLflow, Prefect, MinIO (for artifact storage), and an Nginx reverse proxy.

### 1. Environment Setup
Copy the `.env.example` file to create your local `.env`:
```bash
cp .env.example .env
```
Update the values in `.env` with your secure local credentials.

### 2. Startup
Run the following command to build and start the entire stack:
```bash
docker-compose -f docker/docker-compose.yaml up --build -d
```
For development with hot-reloading and debug logs, add the dev override file:
```bash
docker-compose -f docker/docker-compose.yaml -f docker/docker-compose.dev.yaml up --build
```

### 3. Service URLs
Once the stack is up, you can access the services at the following URLs:

| Service | Description | URL |
|---------|-------------|-----|
| **Nginx** | Main entrypoint (reverse proxy) | `http://localhost/` |
| **API** | FastAPI endpoints via proxy | `http://localhost/api/` |
| **API (Direct)** | FastAPI without proxy | `http://localhost:8000/` |
| **MLflow UI** | Experiment tracking dashboard | `http://localhost:5000/` |
| **Prefect UI** | Pipeline orchestration dashboard | `http://localhost:4200/` |
| **MinIO Console**| S3 Bucket Management | `http://localhost:9001/` |

### 4. Triggering a Flow Run
To trigger the training flow from within the running Prefect worker container:
```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py
```

### 5. Teardown
To stop the services while preserving your databases and MLflow artifacts:
```bash
docker-compose -f docker/docker-compose.yaml down
```

To perform a clean reset (WARNING: this deletes all persisted named volumes including artifacts and runs):
```bash
docker-compose -f docker/docker-compose.yaml down -v
```

## Drift Monitoring

Models don't fail dramatically; they degrade slowly. Vocabulary shifts, user behavior changes, and product categories evolve. A model trained on last year's reviews may be silently wrong on this year's language. Monitoring exists to catch this before business metrics tank.

ReviewSentinel features a fully automated Drift Monitoring system using **Evidently AI** and **Prefect**.

### Technical Mechanism
The monitoring pipeline performs a weekly statistical comparison of incoming prediction inputs against the original training distribution. We monitor multiple dimensions simultaneously:
- **Text length**: Character and word count distributions.
- **Vocabulary richness**: Out-of-vocabulary rate shifts.
- **Sentiment drift**: Comparing simple sentiment scoring against reference.
- **Predicted label drift**: Has the output distribution of negative/positive predictions shifted?

### Automatic Retraining
When data drift exceeds configured thresholds, the system triggers retraining automatically. The data decides when to retrain, not a human checking a dashboard:
- **Alert**: If > 30% of features drift, a warning is sent.
- **Retrain**: If > 50% of features drift, the `training_flow` is automatically triggered.

### Endpoints
The Drift Monitoring dashboard is embedded directly into the API:
- `GET /api/monitoring/drift/latest`: Returns JSON metadata about the most recent drift evaluation.
- `GET /api/monitoring/drift/report`: Serves the full, interactive HTML Evidently report.

### Schedule
The `monitoring_flow` runs every Monday at 3:00 AM UTC (configured in `prefect.yaml`), analyzing prediction logs from the prior 7 days.

## Local LLM Judge & Active Learning

While DistilBERT is fast and efficient for the majority of reviews, it occasionally encounters complex edge cases (e.g., sarcasm, mixed sentiment, novel slang) where its confidence drops. To address this, ReviewSentinel incorporates an asynchronous **Local LLM Judge**.

### The "Second Opinion" Mechanism

When the primary classifier makes a prediction with a confidence score in the "uncertainty window" (by default, 0.40 to 0.60 on the winning class), the inference API queues the review for a second opinion instead of trusting it blindly. 

A Prefect orchestration flow (`llm-judge-processing`) periodically dequeues these uncertain reviews and runs them through a locally-hosted **Mistral 7B** model via Ollama. 

### Active Learning Feedback Loop

The LLM Judge is given the review and the original model's probability distribution, and asked to provide its own classification and reasoning. 
- If the LLM Judge **agrees** with the model, the prediction is verified.
- If the LLM Judge **disagrees** with the model, it logs a "conflict".

These conflicts represent the precise edge cases where the primary model is failing. Before any automated drift retraining runs, the pipeline exports these logged conflicts and ingests them into the training dataset. By automatically appending these high-value, LLM-labeled edge cases to the training data, the primary model continuously improves on the exact concepts it finds most difficult.

### Configuration

The LLM Judge is configured in `configs/pipeline_params.yaml`:
- **`confidence_window`**: Defines the probability range that triggers a second opinion.
- **`ollama`**: Configuration for connecting to the local Ollama instance.
- **`queue` & `conflicts`**: SQLite database paths and batch settings for the queueing system.

## Semantic Search

While DistilBERT handles sentiment classification, ReviewSentinel also provides a high-performance **Semantic Search** feature powered by `sentence-transformers` and **FAISS**.

### Architecture
- **Sentence Encoder**: `all-MiniLM-L6-v2` is used exclusively for generating dense semantic embeddings (float32 vectors). Unlike the fine-tuned DistilBERT which is optimized for sentiment separation, this model is specifically trained for general semantic similarity.
- **Vector Index**: A flat inner-product vector index (`faiss.IndexFlatIP` wrapped in `faiss.IndexIDMap`) enables exact nearest-neighbor retrieval with sub-100ms latency.
- **Graceful Degradation**: The API initializes the searcher lazily. If the index is missing, the API continues to function, returning `null` for similar reviews rather than failing the request.

### Sentiment Alignment Signals
Semantic search doesn't just find similar text; it analyzes the context. When a new review is classified, the search engine fetches historically similar reviews and compares their known sentiment against the current prediction.
- If ≥ 80% of similar past reviews match the current prediction, an alignment signal of `STRONG_SUPPORT` is provided.
- If < 40% match, a signal of `CONTRADICTS` is returned, alerting downstream consumers of a potential anomaly or nuanced edge case.

### Usage in API
To retrieve semantically similar historical reviews, simply set `include_similar_reviews: true` in your `/predict` request:

```json
{
  "title": "Battery dies too fast",
  "text": "I like the design but it doesn't last a full day.",
  "include_similar_reviews": true,
  "similar_reviews_count": 3
}
```

### Infrastructure & Orchestration
- **Offline Capabilities**: Model weights are cached in `artifacts/sentence_transformers/` making the system resilient in air-gapped or restricted network environments.
- **Automated Rebuilds**: The FAISS index is automatically rebuilt at the end of every successful training pipeline (`rebuild_search_index_task`), ensuring search results always reflect the latest production corpus.

## Infrastructure Provisioning

ReviewSentinel's infrastructure is managed entirely through code using **Terraform**. The core philosophy is reproducibility: infrastructure is not configured by clicking through cloud consoles, but defined declaratively. If a server dies, rebuilding it from code takes one command and produces an identical result.

### Setup and Requirements

1. **Install Terraform** (~> 1.7 required)
   - macOS: `brew install terraform`
   - Linux:
     ```bash
     wget https://releases.hashicorp.com/terraform/1.7.5/terraform_1.7.5_linux_amd64.zip
     unzip terraform_1.7.5_linux_amd64.zip
     sudo mv terraform /usr/local/bin/
     ```
   *Note: We pin to a specific version instead of `latest` to ensure reproducibility and prevent breaking changes from major version bumps.*
2. Generate a **Hetzner Cloud API Token** via the Hetzner Console.
3. You must have an SSH client and `kubectl` installed to interact with the cluster.

### Provisioning Workflow

1. Copy the variables template:
   ```bash
   cp terraform/environments/production/terraform.tfvars.example terraform/environments/production/terraform.tfvars
   ```
2. Edit `terraform.tfvars` and add your Hetzner Cloud API token and update `allowed_ssh_ips` to your local IP address (e.g., `["203.0.113.50/32"]`).
3. Initialize the Terraform modules:
   ```bash
   make tf-init
   ```
4. Preview the changes:
   ```bash
   make tf-plan
   ```
5. Apply the changes to provision the server and cluster:
   ```bash
   make tf-apply
   ```
6. Fetch the `kubeconfig` to connect to the new cluster:
   ```bash
   make tf-kubeconfig
   ```
7. Verify access:
   ```bash
   KUBECONFIG=./kubeconfig kubectl get nodes
   ```

To destroy the infrastructure, use `make tf-destroy`, which includes a safety confirmation step.

### Firewall Configuration

The firewall implements the principle of least privilege. 

| Port | Protocol | Service | Source IPs |
|------|----------|---------|------------|
| 22 | TCP | SSH | `allowed_ssh_ips` |
| 80 | TCP | HTTP / Reverse Proxy | `0.0.0.0/0`, `::/0` |
| 443 | TCP | HTTPS | `0.0.0.0/0`, `::/0` |
| 6443 | TCP | Kubernetes API Server | `allowed_ssh_ips` |
| 8000 | TCP | FastAPI Service | `allowed_api_ips` |
| 5000 | TCP | MLflow UI | `allowed_ssh_ips` |
| 4200 | TCP | Prefect UI | `allowed_ssh_ips` |
| Any | ICMP | Ping / Connectivity test | `0.0.0.0/0`, `::/0` |

### Terraform State

The `.tfstate` file is the source of truth for Terraform about what exists in the cloud. It is **gitignored** because it contains sensitive information, including generated private keys. 

- If the state gets out of sync with reality (e.g., someone manually modifies resources in the console), run `terraform refresh`.
- If the state file is lost, you must reconstruct it using `terraform import` or accept that the resources are orphaned and destroy/re-create them.

#### Production Upgrade Path
Currently, the state is stored locally (`backend "local"`). When transitioning to a multi-developer setup, migrate the state to a remote backend like AWS S3 with DynamoDB locking. To upgrade:

```hcl
# In backend.tf
terraform {
  backend "s3" {
    bucket         = "reviewsentinel-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "eu-central-1"
    encrypt        = true
    dynamodb_table = "reviewsentinel-terraform-locks"
  }
}
```
Run `terraform init` to automatically migrate your local state to the newly configured remote backend.

## Kubernetes Deployment

ReviewSentinel uses k3s (a lightweight Kubernetes distribution) as its container orchestration engine. It translates the local Docker Compose architecture into a production-ready, fault-tolerant cluster.

### Conceptual Model
- **Pod**: The smallest deployable unit (e.g., a single API container).
- **Deployment**: Manages stateless Pods (API, Prefect Worker). It handles rolling updates and ensures a specific number of replicas run.
- **StatefulSet**: Manages stateful Pods (MLflow, Prefect Server, MinIO, Ollama). It provides stable hostnames and persistent volumes.
- **Service**: A stable internal network endpoint that routes traffic to Pods.
- **Ingress**: External routing rules (managed by NGINX Ingress Controller) that map domains and paths to internal Services.
- **PersistentVolumeClaim (PVC)**: Requests for storage backed by the node's local disk via the `local-path` provisioner.

### Complete Deployment Sequence

Deploying the cluster from scratch follows a specific sequence. Ensure you have run Terraform first.

1. **Provision Infrastructure**: `make tf-apply` — Provisions VPS and installs k3s.
2. **Access Cluster**: `export KUBECONFIG=./kubeconfig` — Points your local `kubectl` to the remote cluster.
3. **Verify Node**: `kubectl get nodes` — Confirms the node is `Ready`.
4. **Bootstrap Cluster**: `make k8s-setup` — Installs Helm charts (NGINX ingress, cert-manager) and creates secrets.
5. **Create Namespaces**: `kubectl apply -f k8s/namespaces/`
6. **Deploy Application**: `make k8s-deploy` — Applies Kustomize overlays.
7. **Verify Deployment**: `make k8s-status`
8. **Verify TLS**: `kubectl get certificate -n reviewsentinel`

### Resource Requirements

To ensure stable inference and orchestration, the cluster requires adequate compute.

| Component | CPU Request | Memory Request | State |
|-----------|-------------|----------------|-------|
| Ollama (Mistral 7B) | 500m | 4Gi | Stateful (10Gi PVC) |
| API | 200m | 512Mi | Stateless |
| Prefect Worker | 200m | 512Mi | Stateless |
| Prefect Server | 100m | 256Mi | Stateful (5Gi PVC) |
| MLflow | 100m | 256Mi | Stateful (5Gi PVC) |
| MinIO | 100m | 256Mi | Stateful (20Gi PVC) |

> [!WARNING]
> **Ollama Memory Constraint**: Running Mistral 7B on CPU with 4-bit quantization strictly requires ~4GB RAM. A `cx32` (8GB RAM) Hetzner instance is the **recommended minimum** to house this entire stack securely without triggering OOM (Out Of Memory) kills.

### TLS Workflow
Cert-manager watches the Ingress object for the `cert-manager.io/cluster-issuer` annotation. It uses the HTTP-01 challenge solver (via the NGINX ingress class) to automatically request, provision, and renew Let's Encrypt certificates, storing them securely in the `reviewsentinel-tls` Secret.

### Storage Limitations
k3s defaults to the `local-path` provisioner. All PVCs are **ReadWriteOnce** and strictly node-local. This is optimal for our single-node architecture. If scaling to a multi-node cluster, you must replace `local-path` with a distributed block storage engine like **Longhorn** or an external NFS share to allow Pods to move between nodes without losing data.

### Operational Commands
- **View Logs**: `make k8s-logs` (Streams API logs)
- **Shell Access**: `make k8s-shell` (Executes `/bin/bash` in the first API Pod)
- **Update Secrets**: Edit `.env` and run `make k8s-secrets`

## Helm Chart Deployment

ReviewSentinel provides a comprehensive Helm Chart for managing the entire stack. Unlike Kustomize overlays, the Helm chart utilizes the `values.yaml` model to abstract environment configuration away from Kubernetes manifests.

### Values Architecture
- **`helm/reviewsentinel/values.yaml`**: The canonical defaults for the chart. Never modify this for a specific environment.
- **`helm/environments/<env>/values.yaml`**: Environment-specific overrides (e.g., pinning image tags, enabling autoscaling). Helm merges these over the defaults.

### Deployment Workflow

You must have `helm` (>= 3.12) installed locally, alongside the `helm-diff` plugin (`helm plugin install https://github.com/databus23/helm-diff`).

1. **Create Secrets**: The Helm chart *does not* manage sensitive credentials. You must create the secrets beforehand (see `helm/environments/production/secrets.yaml.example`).
   ```bash
   kubectl create secret generic reviewsentinel-secrets \
     --namespace reviewsentinel \
     --from-literal=minio-root-user=... \
     --from-literal=minio-root-password=... \
     --from-literal=aws-access-key-id=... \
     --from-literal=aws-secret-access-key=...
   ```
2. **Lint Chart**: Validate the syntax and logical integrity.
   ```bash
   make helm-lint
   ```
3. **Review Changes**: Preview the modifications against the live cluster.
   ```bash
   make helm-diff
   ```
4. **Deploy / Upgrade**: Apply the chart. The `--atomic` flag ensures that if the deployment fails health checks, it automatically rolls back.
   ```bash
   make helm-upgrade
   ```
5. **Verify**: Check `NOTES.txt` console output for active endpoints.

### Pod Lifecycle Probes

The API pod relies on three sequential probes:
1. **Startup Probe**: Tolerates up to 5 minutes of startup delay to account for SentenceTransformer model downloads. Prevents premature kills.
2. **Liveness Probe**: Restarts the pod if it becomes fundamentally deadlocked or unresponsive on the `/health` endpoint.
3. **Readiness Probe**: Temporarily removes the pod from the Service load balancer if it's too busy or temporarily failing, ensuring no user traffic hits an unready instance.

### State Retention Policy

All PersistentVolumeClaims (PVCs) generated by this chart (e.g., MinIO data, MLflow databases) are annotated with `helm.sh/resource-policy: keep`.

If you run `make helm-uninstall`, the deployments and services will be destroyed, but the **data will remain intact**. If you genuinely wish to perform a hard reset and delete all persisted data:
```bash
kubectl delete pvc -l app.kubernetes.io/instance=reviewsentinel -n reviewsentinel
```

## GitOps Deployment (ArgoCD)

ReviewSentinel uses GitOps as the definitive mechanism for deploying to the cluster. In this model, git is the single source of truth. If git says a component is deployed at version X, the cluster must match. If an operator manually alters the cluster, ArgoCD detects the configuration drift and immediately reverts it. This ensures absolute predictability and auditability.

### Day 0 — Infrastructure Bootstrap

To bring the initial infrastructure online and install ArgoCD:

1. Provision the infrastructure:
   ```bash
   make tf-apply
   make tf-kubeconfig
   ```
2. Install ArgoCD into the cluster:
   ```bash
   make argocd-bootstrap
   ```
3. Retrieve your initial admin password and login:
   ```bash
   make argocd-password
   make argocd-login
   ```
4. Create the repository credentials (so ArgoCD can pull updates) and notification secrets:
   ```bash
   kubectl create secret generic reviewsentinel-repo-creds \
     --from-literal=type=git \
     --from-literal=url=https://github.com/FAKER-112/End-to-End-Sentiment-Analysis-Pipeline-for-Customer-Reviews.git \
     --from-literal=username=x-token \
     --from-literal=password=<YOUR_GITHUB_PAT_WITH_READ_ACCESS> \
     -n argocd
     
   kubectl label secret reviewsentinel-repo-creds \
     argocd.argoproj.io/secret-type=repository \
     -n argocd
     
   kubectl create secret generic argocd-notifications-secret \
     --from-literal=slack-token=<YOUR_SLACK_BOT_TOKEN> \
     -n argocd
   ```
5. Apply the App-of-Apps (ArgoCD will now self-manage and deploy the rest of the stack):
   ```bash
   make argocd-apply-apps
   ```

### Continuous Delivery Pipeline

The deployment pipeline is fully automated and consists of three interconnected GitHub Actions workflows:

1. **Build and Push**: On a merge to `main`, the image is built, tagged with the short Git SHA, and pushed to GHCR along with a provenance attestation and SBOM.
2. **Update Image Tag**: Upon a successful build, a separate workflow updates the `environments/production/values.yaml` file with the new SHA tag, and commits it back to `main` with a `[skip ci]` flag to prevent an infinite build loop.
3. **ArgoCD Sync**: ArgoCD detects the commit, computes the diff, and initiates a Helm upgrade against the cluster. Kubernetes performs a rolling update. Slack and GitHub notifications are dispatched upon success or failure.

### HPA Interaction & Ignore Differences

Because we use a Horizontal Pod Autoscaler (HPA) to scale API replicas, we explicitly instruct ArgoCD to ignore the `/spec/replicas` field of the API Deployment. Without this `ignoreDifferences` configuration, ArgoCD would constantly fight the HPA, attempting to revert the replica count back to the static number defined in the Helm chart. 

### Rollback Procedures

There are two distinct paths to roll back a problematic release:

1. **Git Revert (Recommended)**: Use standard `git revert` to undo the commit that broke the cluster. This runs through the CI pipeline and leaves a clean audit trail.
2. **ArgoCD Rollback (Emergency)**: For critical outages where every second counts, bypass CI and instruct ArgoCD to instantly rollback to a previous known-good state. Run `make argocd-rollback` to view history and select a revision interactively. 

### Accessing ArgoCD

The ArgoCD dashboard is exposed via the cluster ingress. You can access it securely at:
**[https://argocd.127.0.0.1.nip.io](https://argocd.127.0.0.1.nip.io)**

## CI/CD Pipeline

ReviewSentinel employs a robust CI/CD pipeline using **GitHub Actions** for push-based operations, seamlessly handing off to **ArgoCD** for pull-based deployment.

### Push vs Pull Architecture
GitHub Actions is the only push component in the system. It builds, tests, and updates the repository state. Everything else is pull-based. This limits the failure domain—ArgoCD and Kubernetes are self-healing and will continuously reconcile against the repository, whereas push-based systems break permanently on connection errors.

### Two Pipeline Modes
The pipeline operates in two distinct modes:
1. **PR Gates (`pr-checks.yaml`)**: An expected failure mechanism. If tests, linting, or the model quality gate fail, the PR is blocked. This protects the `main` branch.
2. **Main Branch Delivery (`build-push.yaml`, `update-image-tag.yaml`)**: Takes approved code, builds it, pushes it, and triggers a deployment. Failures here are incidents that require immediate attention.

### Model Quality Gate
This is not standard software CI. The Model Quality Gate is an automated policy enforcement mechanism: **no model that performs worse than production can be deployed**.
If a developer alters `src/models/`, the CI pipeline compares the new F1 score against the live MLflow baseline. If it doesn't meet the required threshold (e.g., 1% improvement), the PR is failed automatically. This prevents silent model degradation and drift.

### The Trigger Chain
1. Developer merges PR to `main`.
2. `build-push.yaml` builds the Docker image, tags it with the short git SHA, signs it via Cosign, and pushes it to GHCR.
3. `update-image-tag.yaml` detects the successful build, safely modifies `environments/production/values.yaml` with the new tag, and pushes a commit with `[skip ci]`.
4. ArgoCD detects the commit, initiates a diff, and triggers a Kubernetes rolling update.
5. Slack receives the success/failure notification.

### Secrets and Operational Tokens
To provision this project from scratch, refer to `docs/secrets.md` for a comprehensive list of all required GitHub Actions and Kubernetes secrets.

### Local CI Simulation
Before pushing a PR, it's highly recommended to run the CI checks locally to avoid wasting Actions minutes.
```bash
make ci-pr-simulation
```
#   R e v i e w S e n t i n e l  
 #   R e v i e w S e n t i n e l  
 