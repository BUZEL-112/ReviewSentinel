# Data Flow — Life of a Customer Review

This document traces a single customer review through every transformation it undergoes in ReviewSentinel, from its raw source on Amazon's servers to its logged prediction in the monitoring system. Knowing this flow end-to-end is essential for debugging unexpected model behaviour or drift alerts.

---

## Overview

```
Amazon JSONL.gz (raw)
    │
    ▼ LoadData
data/raw/dataset.csv (pandas DataFrame)
    │
    ▼ DataValidator (Great Expectations)
Validated DataFrame (or pipeline aborted)
    │
    ▼ CleanDataBERT
label column added (0/1/2) → text cleaned (URL strip + whitespace)
→ tokenised (distilbert-base-uncased, max_len=128)
→ split (70% train / 10% val / 20% test)
→ HuggingFace Dataset objects
    │
    ▼ ModelTrainer (HuggingFace Trainer)
Fine-tuned DistilBERT weights → artifacts/models/distilbert/
MLflow run logged (params, metrics, model artifact → MinIO)
    │
    ▼ ModelEvaluator
metrics.json → artifacts/evaluation/
    │
    ▼ Quality Gate
Compare new F1 vs. MLflow production baseline
    │
    ├─ PASS ─► deploy_model_task
    │           → artifacts/best_model/
    │           → MLflow run tagged is_production=true
    │           → FAISS index rebuilt from new corpus
    │
    └─ FAIL ─► pipeline halts; current production model unchanged
```

---

## Stage 1 — Raw Ingestion (`LoadData`)

**Source:** `configs/config.yaml` → `data_ingestion.source_url`  
**Output:** `data/raw/dataset.csv`  
**Code:** [`src/data/load_data.py`](../../src/data/load_data.py)

`LoadData.load_data()` streams the JSONL.gz file from the UCSD McAuley Lab dataset server using `requests` with streaming enabled. The raw JSONL is parsed line-by-line. Each review is a JSON object with fields including `rating`, `title`, `text`, `verified_purchase`, and timestamp metadata.

Only `rating`, `title`, and `text` are retained for downstream use. The raw CSV is saved to `data/raw/dataset.csv` and returned as a pandas DataFrame.

**Prefect task config:** 3 retries, delays of 30s / 60s / 120s (handles transient network issues).

---

## Stage 2 — Data Validation (`DataValidator`)

**Input:** Raw DataFrame  
**Code:** [`src/orchestration/validation.py`](../../src/orchestration/validation.py)  
**Config:** `configs/pipeline_params.yaml` → `orchestration.validation`

Before any preprocessing, the pipeline runs a validation suite. If validation fails, a Prefect `Abort` is raised and the pipeline halts — no model is trained on corrupted data.

| Check | Default threshold | Failure action |
|-------|-----------------|----------------|
| Required columns present (`rating`, `title`, `text`) | — | Abort |
| Null ratio on `rating` | ≤ 10% | Abort |
| Null ratio on `text` | ≤ 10% | Abort |
| Rating bounds | Must be 1–5 | Abort |
| Class imbalance | No class > 80% of total | Abort |
| Minimum row count | ≥ 100 rows | Abort |

---

## Stage 3 — Preprocessing (`CleanDataBERT`)

**Input:** Validated raw DataFrame  
**Output:** HuggingFace `Dataset` objects (train, val, test) + test labels array  
**Code:** [`src/data/clean_data.py`](../../src/data/clean_data.py)  
**Config:** `configs/config.yaml` → `clean_data_bert`

### 3a — Label Mapping
```python
rating 1, 2  →  label 0  (negative)
rating 3     →  label 1  (neutral)
rating 4, 5  →  label 2  (positive)
```

### 3b — Minimal Text Cleaning
The title and body text fields are each independently cleaned with:
1. URL removal: `re.sub(r"http\S+|www\S+", "", text)`
2. Whitespace collapse: `re.sub(r"\s+", " ", text).strip()`

No lowercasing, no punctuation removal, no stemming. See [Data Card](../ml/data-card.md) for the rationale.

### 3c — Concatenation
`"{title} {body_text}"` — a single space joins the two fields. Empty body text becomes an empty string.

### 3d — Tokenisation
```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Applied with: max_length=128, padding=True, truncation=True
```

Reviews longer than 128 tokens are **truncated** at 128. This is the `max_len` used during training.

> [!WARNING]
> **max_len discrepancy:** Inference uses `max_len=512` (set in `pipeline_params.yaml` → `inference_pipeline.max_len`). Training used `max_len=128`. Reviews between 128 and 512 tokens will be processed at inference time with tokens 129–512 that the model never saw during training. The model may not utilise this extra context effectively. See [Data Card](../ml/data-card.md) for the documented trade-off.

### 3e — Train / Val / Test Split
- Stratified by label to maintain class proportions
- Proportions: 70% train, 10% val, 20% test (derived from `test_size=0.2`, `val_size=0.1`)
- `random_state=42`

The split DataFrames are converted to HuggingFace `Dataset` objects and returned to `ModelTrainer`.

---

## Stage 4 — Training (`ModelTrainer`)

**Input:** HuggingFace Dataset objects  
**Output:** Model weights at `artifacts/models/distilbert/`, MLflow run  
**Code:** [`src/models/train_model.py`](../../src/models/train_model.py)

> [!CAUTION]
> `ModelTrainer.__init__()` contains `self.df = dataframe.sample(frac=0.001).copy()` on line 30. This samples **0.1% of the data** for fast development iteration. **Remove or change `frac` before running a real training job.** See [Training Guide](training-guide.md).

The HuggingFace `Trainer` API handles the training loop, gradient updates, and evaluation. Key hyperparameters from `configs/config.yaml` → `distilbert_model.training`:

| Hyperparameter | Default |
|---|---|
| Epochs | 1 |
| Train batch size | 8 |
| Eval batch size | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |

MLflow logging is enabled via `report_to=["mlflow"]` in `TrainingArguments`. The model weights and tokeniser are saved to `artifacts/models/distilbert/` and also logged as an MLflow artifact (backed by MinIO).

---

## Stage 5 — Evaluation (`ModelEvaluator`)

**Input:** Trained `Trainer` object  
**Output:** `artifacts/evaluation/metrics.json`, metrics dict  
**Code:** [`src/models/evaluate_model.py`](../../src/models/evaluate_model.py)

The evaluator runs `trainer.evaluate()` on the held-out test split and computes:
- Weighted F1 score (used by the quality gate)
- Accuracy
- Per-class precision, recall, F1 (negative / neutral / positive individually)

---

## Stage 6 — Quality Gate

**Input:** `metrics` dict from evaluator  
**Code:** [`src/orchestration/flows.py`](../../src/orchestration/flows.py) — `quality_gate_task()`  
**Config:** `pipeline_params.yaml` → `orchestration.quality_gate`

The gate queries MLflow for the most recent run tagged `is_production=true` and retrieves its F1 score as the baseline. The new model's F1 must exceed the baseline by at least `min_f1_improvement` (default: 0.01 = 1 percentage point) to pass.

**First run behaviour:** If no production baseline exists in MLflow, and `first_run_auto_deploy=true` (default), the gate passes automatically.

---

## Stage 7 — Inference (Runtime)

**Trigger:** `POST /predict` or `POST /predict/batch`  
**Code:** [`src/api/api.py`](../../src/api/api.py), [`src/pipeline/inference_pipeline.py`](../../src/pipeline/inference_pipeline.py)

At runtime, every incoming review goes through exactly the same text cleaning as training (URL removal + whitespace collapse + concatenation), but tokenised with `max_len=512`.

The model returns softmax logits → probabilities. The winning class probability is the `confidence` score.

### Enrichment layers (optional, per-request)

| Parameter | Effect |
|-----------|--------|
| `include_explanation: true` | Runs SHAP `KernelExplainer` on the input; returns top-k token attributions |
| `include_similar_reviews: true` | Queries FAISS index; returns top-k similar historical reviews + alignment signal |

### Post-prediction actions (always)

1. **Prediction logged** — appended to `artifacts/monitoring/prediction_log.jsonl`
2. **Prometheus metrics updated** — `sentiment_predictions_total`, confidence histogram
3. **LLM Judge queue** — if `0.40 ≤ confidence ≤ 0.60`, review is written to SQLite queue

---

## Stage 8 — Drift Monitoring

**Trigger:** Prefect schedule — every Monday at 03:00 UTC  
**Code:** [`src/monitoring/drift_monitor.py`](../../src/monitoring/drift_monitor.py)  
**Config:** `pipeline_params.yaml` → `monitoring`

The monitoring flow reads prediction logs from the last 7 days and compares them against the training reference distribution (saved to `artifacts/monitoring/` at deploy time by `ReferenceStore`). Evidently AI runs statistical tests across:
- Text length distributions
- Vocabulary richness (OOV rate)
- Simple sentiment score distributions
- Predicted label distributions

**Thresholds:**
- **> 30% features drifted** → alert logged
- **> 50% features drifted** → `training_flow` automatically triggered

The Evidently HTML report is saved to `artifacts/monitoring/reports/` and served at `GET /api/monitoring/drift/report`.

---

## Stage 9 — LLM Judge Processing

**Trigger:** Prefect schedule — every 4 hours  
**Code:** [`src/llm_judge/judge.py`](../../src/llm_judge/judge.py), [`src/orchestration/judge_tasks.py`](../../src/orchestration/judge_tasks.py)

The `judge_processing_flow` dequeues up to `batch_size` (default: 50) pending reviews from the SQLite queue. Each review is sent to Ollama with the original prediction and probability distribution in the prompt (see [`src/llm_judge/prompt_builder.py`](../../src/llm_judge/prompt_builder.py)).

The LLM returns a classification and reasoning. If it disagrees with the primary model, the entry is written to `conflicts.db`. Before the next automated retraining, these conflicts are exported and prepended to the training DataFrame, closing the active-learning loop.
