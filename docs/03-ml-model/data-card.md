# Data Card — ReviewSentinel Training Dataset

**Last Updated:** June 2026  
**Dataset Version:** Amazon Review 2023 — `All_Beauty`  
**Managed by:** [`src/data/load_data.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/data/load_data.py), [`src/data/clean_data.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/data/clean_data.py)

---

## Dataset Source

| Property | Value |
|----------|-------|
| **Name** | Amazon Review 2023 |
| **Category used** | `All_Beauty` |
| **Format** | JSONL.gz (streamed) |
| **Download URL** | `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz` |
| **Reference** | Hou et al., "Bridging Language and Items for Retrieval and Recommendation", 2024 |
| **License** | Non-commercial research use (per UCSD dataset terms) |

The URL is configured in [`configs/config.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/configs/config.yaml) under `data_ingestion.source_url`. `LoadData` streams and saves the raw file to `data/raw/dataset.csv`.

---

## Label Construction

Raw Amazon reviews carry a `rating` field (integer 1–5). ReviewSentinel maps these to three sentiment classes:

| Rating | Sentiment Label | Integer ID |
|--------|----------------|-----------|
| 1, 2 | `negative` | 0 |
| 3 | `neutral` | 1 |
| 4, 5 | `positive` | 2 |

This mapping is applied in [`src/data/clean_data.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/data/clean_data.py) during the `CleanDataBERT` preprocessing step.

> [!NOTE]
> Rating 3 ("neutral") is the smallest class in this dataset. The model's performance on neutral reviews is likely lower than on positive or negative reviews. Check per-class metrics in MLflow before drawing conclusions from the weighted F1 score alone.

---

## Data Splits

| Split | Fraction | Config Key |
|-------|---------|------------|
| **Train** | 70% | `clean_data_bert.val_size` (derived) |
| **Validation** | 10% | `clean_data_bert.val_size = 0.1` |
| **Test** | 20% | `clean_data_bert.test_size = 0.2` |

- `random_state = 42` for reproducibility
- `shuffle = true` before splitting
- Splits are stratified by label to maintain class proportions

Configuration: [`configs/config.yaml`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/configs/config.yaml) under `clean_data_bert`.

---

## Text Fields

The model is trained on the **concatenation of two Amazon fields**:

| Amazon field | Role | Notes |
|---|---|---|
| `title` | Review headline | Short; strong sentiment signal |
| `text` | Review body | Long-form; provides context |

Concatenation format: `"{title} {body}"` — implemented in [`_build_texts()`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/pipeline/inference_pipeline.py) and applied consistently in both training and inference.

---

## Preprocessing Decisions

ReviewSentinel uses **minimal cleaning** rather than aggressive NLP preprocessing. This is a deliberate design choice: BERT-family models are pre-trained on raw text with punctuation and mixed case, so aggressive lowercasing, stemming, and stopword removal destroy signal the model already understands.

### What IS applied
| Step | Reason |
|------|--------|
| URL removal (`http://...`, `www...`) | URLs add no sentiment signal |
| Whitespace collapse | Normalises multi-space and newline formatting |

### What is NOT applied (and why)
| Step | Why excluded |
|------|-------------|
| Lowercasing | DistilBERT's uncased tokenizer handles this internally |
| Stopword removal | Stopwords like "not", "never" carry critical sentiment negation |
| Stemming / lemmatisation | Destroys morphological information the tokenizer uses |
| Punctuation removal | `!`, `?`, `...` carry affective signal |

Implementation: [`_minimal_clean()`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/pipeline/inference_pipeline.py#L135) and [`CleanDataBERT`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/data/clean_data.py).

---

## Tokenisation

| Property | Value | Source |
|----------|-------|--------|
| Tokenizer | `distilbert-base-uncased` (WordPiece) | HuggingFace Hub |
| `max_len` during training | **128** tokens | `configs/config.yaml` → `clean_data_bert.max_len` |
| `max_len` during inference | **512** tokens | `configs/pipeline_params.yaml` → `inference_pipeline.max_len` |

> [!WARNING]
> **Known discrepancy:** Training truncates at 128 tokens; inference allows up to 512 tokens. Reviews longer than 128 tokens were truncated during training, meaning the model has limited exposure to long-review structures. Inference allows longer inputs but the model may not utilise the extra context effectively. This is a known limitation, not a bug — it was accepted as a tradeoff between training speed and inference flexibility. A future training run with `max_len=256` or `max_len=512` would address this.

---

## Class Imbalance

Amazon review datasets are typically skewed toward positive ratings (products with very few reviews are less likely to be reviewed at all; satisfied customers are slightly more likely to leave reviews). The `All_Beauty` category follows this pattern.

The Great Expectations data validation in [`src/orchestration/validation.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/orchestration/validation.py) enforces a `max_class_imbalance` threshold (default: 80%) to prevent severe skew from reaching the trainer. If any class exceeds 80% of total samples, the pipeline aborts.

---

## LLM Judge Conflict Ingestion

Over time, the training dataset grows beyond the original Amazon split. When the LLM Judge disagrees with the primary classifier on a prediction, the disagreement is logged as a **conflict** in `artifacts/llm_judge/conflicts.db`.

Before automated retraining triggered by drift monitoring, the pipeline exports these conflicts and appends them to the training DataFrame. This means:

1. **The dataset is not static** — it grows with production traffic
2. **Conflict records are human-validated** only indirectly (via the LLM Judge, which itself has error rates)
3. **The proportion of conflict records in the training set grows over time** — this is intentional active learning but should be monitored to ensure the conflict data quality is acceptable

See [`judge_tasks.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/orchestration/judge_tasks.py) for the export logic.

---

## Data Lineage

```
Amazon Review 2023 JSONL.gz
  └─► LoadData.load_data()          → data/raw/dataset.csv
        └─► CleanDataBERT.prepare_datasets()
              ├─► label mapping (rating → 0/1/2)
              ├─► URL removal + whitespace collapse
              ├─► tokenisation (distilbert-base-uncased, max_len=128)
              └─► train/val/test split (70/10/20, random_state=42)
                    └─► HuggingFace Dataset objects → ModelTrainer
```

---

## Known Data Quality Issues

| Issue | Impact | Status |
|-------|--------|--------|
| Duplicate reviews | Amazon datasets contain some exact or near-duplicate reviews | Not deduplicated; low estimated impact at this dataset size |
| Verified vs. unverified purchases | Dataset contains both; unverified purchase reviews may be noisier | Not filtered; no evidence this materially affects accuracy |
| Time period coverage | Dataset reflects reviews up to 2023; model may not reflect post-2023 vocabulary | Drift monitoring catches this over time |
