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
!mlflow ui --host 0.0.0.0 --port 5000
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

## Kubernetes Deployment (Kind)

ReviewSentinel uses [Kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker) for a lightweight, local Kubernetes deployment. This replaces the need for complex cloud provisioning while still providing a full Kubernetes environment.

### Setup and Requirements

1. **Install Docker**: Ensure Docker is installed and running on your machine.
2. **Install Kind**: Follow the [official instructions](https://kind.sigs.k8s.io/docs/user/quick-start/#installation).
3. **Install kubectl**: Required to interact with the cluster.
4. **Install envsubst**: Standard on most Linux distributions (part of `gettext`). Used for templating the Kind config.

### Provisioning Workflow

We manage the local cluster lifecycle via the provided `Makefile`.

1. **Start the Cluster and Deploy**:
   ```bash
   make up
   ```
   This command will:
   - Generate `kind-config.yaml` from `kind-config.yaml.template` using your project root directory.
   - Destroy any existing Kind cluster named `kind`.
   - Create a new Kind cluster and mount necessary local directories (e.g., `data/`, `artifacts/`, `configs/`).
   - Apply all Kubernetes manifests in the `k8s/` directory.

2. **Watch the Deployment**:
   ```bash
   make logs
   ```
   This streams the status of all pods in the cluster as they start up.

3. **Apply Manifest Updates**:
   If you modify any files in the `k8s/` directory, you can apply them without recreating the cluster:
   ```bash
   make apply
   ```

4. **Tear Down the Cluster**:
   ```bash
   make down
   ```
   This removes the Kind cluster. It does not delete data stored in your local mounted directories (`data/`, `artifacts/`, etc.).

### Architecture and Volumes

The Kind configuration (`kind-config.yaml.template`) automatically mounts local host paths into the Kubernetes nodes. This ensures that persistent volumes and configurations are shared seamlessly between your host machine and the containers:

- `hostPath: ${PROJECT_ROOT}/data` -> `/data`
- `hostPath: ${PROJECT_ROOT}/artifacts` -> `/artifacts`
- `hostPath: ${PROJECT_ROOT}/configs` -> `/configs`

These mounts allow local data persistence for MLflow, MinIO, and Prefect without needing complex Kubernetes StorageClasses.
