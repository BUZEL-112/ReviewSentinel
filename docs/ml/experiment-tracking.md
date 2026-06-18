# Experiment Tracking (MLflow)

ReviewSentinel uses MLflow for comprehensive experiment tracking. Every time the `ModelTrainer` runs, it logs parameters, metrics, and the resulting model weights to an MLflow tracking server.

---

## 1. Accessing the UI

In local development (Docker Compose or Kind), the MLflow UI is exposed at port 5000.
- **URL:** [http://localhost:5000](http://localhost:5000)

Navigate to the `reviewsentinel-training` experiment in the left sidebar to view the run history.

---

## 2. What is Logged?

The HuggingFace `Trainer` is configured with `report_to=["mlflow"]`. This automatically logs:

**Parameters:**
- `learning_rate`
- `train_batch_size`
- `eval_batch_size`
- `seed`
- `optimizer`
- `lr_scheduler_type`
- `num_epochs`

**Metrics:**
- `eval_loss`
- `eval_accuracy`
- `eval_f1` (Weighted F1 score, used by the Quality Gate)
- `train_runtime`

**Artifacts:**
In addition to the automatic logging, `ModelTrainer` explicitly logs the final model weights and the tokenizer via `mlflow.transformers.log_model()`. These files are uploaded to the MinIO artifact store.

---

## 3. The `is_production` Tag

MLflow runs are just historical records until they are explicitly tagged for deployment.

If a new model passes the Prefect Quality Gate, the `deploy_model_task` programmatically tags the winning MLflow run with:
`is_production: true`

The Quality Gate *always* looks for the most recent run with this tag to establish the baseline for the next training cycle.

---

## 4. MinIO Artifact Storage

While MLflow manages the metadata (metrics, parameters, tags) in a PostgreSQL or SQLite database, the actual multi-gigabyte model artifacts are stored in an S3-compatible backend. We use MinIO.

- **MinIO Console:** [http://localhost:9001](http://localhost:9001)
- **Bucket:** `reviewsentinel-artifacts`

If you delete the contents of this bucket, the MLflow UI will still show the runs, but attempting to load the model (e.g., via the API) will result in a 503 error because the actual `.safetensors` or `.bin` weights are gone.

---

## 5. Comparing Runs

To evaluate hyperparameter tuning:
1. Open the `reviewsentinel-training` experiment in MLflow.
2. Check the boxes next to multiple runs.
3. Click **Compare**.
4. Use the scatter plot view to graph `learning_rate` against `eval_f1` to visualize optimal training parameters.
