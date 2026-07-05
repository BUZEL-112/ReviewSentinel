# Training Guide

This document covers how to train a new model in ReviewSentinel, how the quality gate evaluates it, and how to safely navigate the codebase for ML experimentation.

---

## 1. Running a Training Job

ReviewSentinel models are trained via a Prefect flow defined in `src/orchestration/flows.py`. This flow manages data ingestion, validation, preprocessing, training, evaluation, and conditional deployment.

### Development Mode (The `frac=0.001` Footgun)

> [!WARNING]
> By default, `ModelTrainer.__init__()` in [`src/models/train_model.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/models/train_model.py) contains the following line:
> ```python
> self.df = dataframe.sample(frac=0.001).copy()
> ```
> This samples **0.1% of the dataset** (roughly 50–100 reviews). This is extremely useful for verifying that the pipeline runs end-to-end without crashing, but **it will produce a garbage model.** 
> 
> **You MUST comment out or modify this line before running a real training job.**

### Triggering the Flow
To trigger a training run manually in your local Docker environment:
```bash
docker-compose -f docker/docker-compose.yaml exec prefect-worker python scripts/run_flow.py
```

---

## 2. The Quality Gate

ReviewSentinel implements a strict "do no harm" deployment policy. A model can finish training successfully, but it will not be deployed unless it passes the Quality Gate.

### How it works
1. The `ModelEvaluator` computes metrics (Accuracy, F1, Precision, Recall) on the held-out test split.
2. The `quality_gate_task` queries MLflow for the most recent run tagged with `is_production=true`.
3. It extracts the `eval_f1` metric from that baseline run.
4. If the new model's F1 score is $\ge$ baseline + `min_f1_improvement` (default 0.01), the gate passes.

### First-Run Auto-Deploy
If the gate cannot find a production baseline in MLflow (i.e., this is the first time you are running the system in a fresh environment), the gate defaults to `first_run_auto_deploy=True` and passes automatically.

### Overriding the Gate
If you need to force a model into production regardless of its metrics (e.g., to fix a critical bug in the inference code that required a model rebuild), you must bypass the gate manually:
1. Locate your run in the MLflow UI.
2. Manually add the tag `is_production` with value `true`.
3. Copy the model artifacts from `artifacts/models/distilbert` to `artifacts/best_model`.

---

## 3. Modifying Hyperparameters

All ML hyperparameters are centralized in [`configs/config.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/configs/config.yaml).

**Key parameters to tune:**
- `distilbert_model.training.epochs`: Default is 1 for speed. Increase to 3–5 for production runs.
- `distilbert_model.training.learning_rate`: Default is `2.0e-5`. Standard fine-tuning range is `1e-5` to `5e-5`.
- `clean_data_bert.max_len`: Default is `128`. Increase to `512` to eliminate the training/inference discrepancy (at the cost of significantly longer training times and VRAM usage if using GPUs).

---

## 4. Active Learning: Ingesting Conflicts

ReviewSentinel uses an active learning loop via the LLM Judge. When the LLM Judge disagrees with the DistilBERT model, the disagreement is logged to `artifacts/llm_judge/conflicts.db`.

Currently, these conflicts must be explicitly exported and merged into the `dataset.csv` before running a retraining job. If you rely solely on the automated `training_flow`, the model will retrain on the original dataset, ignoring the conflicts.

To incorporate conflicts:
1. Extract rows from `conflicts.db` where the LLM sentiment is deemed correct.
2. Append them to `data/raw/dataset.csv`.
3. Run the training flow. The increased volume of hard examples will nudge the model boundary in future inferences.
