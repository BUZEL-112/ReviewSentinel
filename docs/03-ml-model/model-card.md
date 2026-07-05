# Model Card — ReviewSentinel Sentiment Classifier

**Last Updated:** June 2026  
**Status:** Production  
**Model ID:** `distilbert-base-uncased` (fine-tuned)  
**Task:** Three-class sequence classification — negative / neutral / positive  
**Artifact path:** `artifacts/models/distilbert/`

---

## Model Description

ReviewSentinel's primary classifier is a fine-tuned instance of `distilbert-base-uncased`, a 66M-parameter distilled version of BERT. It was fine-tuned on labelled Amazon customer reviews (`All_Beauty` category) to predict sentiment as one of three classes. See [ADR 001](../06-decisions/001-distilbert-over-larger-models.md) for the rationale behind this model choice.

The model is loaded by `InferencePipeline` ([src/pipeline/inference_pipeline.py](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/pipeline/inference_pipeline.py)) and served via the FastAPI application ([src/api/api.py](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/api/api.py)).

---

## Intended Use

**This model is intended for:**
- Classifying English-language product reviews into negative / neutral / positive sentiment
- Prioritising customer feedback triage in product management workflows
- Powering SHAP-explained sentiment breakdowns for vocabulary analysis

**This model is NOT intended for:**
- Non-English text (it will produce unreliable results; no multilingual fine-tuning was applied)
- Medical, legal, financial, or safety-critical sentiment analysis
- Detecting hate speech, abuse, or toxicity
- Any use case requiring demographic fairness guarantees (the training data was not audited for bias)

---

## Training Data

See [Data Card](data-card.md) for full details.

| Property | Value |
|----------|-------|
| Source | Amazon Review 2023 dataset (`All_Beauty` category) |
| Label mapping | rating 1–2 → negative, rating 3 → neutral, rating 4–5 → positive |
| Train / Val / Test split | 70% / 10% / 20% |
| Random state | 42 |
| Text preprocessing | URL removal + whitespace collapse only (no stemming, no stopword removal) |

---

## Performance

> [!NOTE]
> The metrics below represent the model's performance on the held-out 20% test split of the `All_Beauty` dataset. Performance on other product categories, other languages, or reviews with very different vocabulary distributions may be substantially different.

The quality gate in `flows.py` uses **weighted F1** as the deployment gate metric, requiring the new model to exceed the current production F1 by at least **0.01** (1 percentage point) before promotion.

Metrics are logged per training run to MLflow under experiment `reviewsentinel-training`. To view live metrics:
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Navigate to http://localhost:5000
```

---

## Confidence Score Interpretation

The model outputs a softmax probability distribution across all three classes. The winning class probability is exposed as `confidence` in the API response.

| Confidence Range | Meaning | System Behaviour |
|-----------------|---------|-----------------|
| ≥ 0.80 | High confidence — reliable prediction | Served directly |
| 0.60–0.79 | Moderate confidence | Served; `low_confidence` flag = `false` |
| 0.40–0.60 | **Uncertainty window** | Served + queued for LLM Judge second opinion |
| < 0.40 | Low confidence | Served; `low_confidence` flag = `true` |

The `low_confidence` API flag fires when `confidence < 0.60` on the winning class.

---

## Edge-Case Flags

The API applies three lightweight heuristics to every prediction. These are best-effort signals, not guarantees.

### `short_review` (confidence.length < 10 chars)
Reviews shorter than 10 characters provide minimal signal to the model. Treat predictions on very short inputs with higher scepticism.

### `low_confidence` (winning class probability < 0.60)
The model is not strongly committed to its prediction. Consider this a prompt to seek additional signal (e.g., check `similar_reviews` alignment).

### `possible_sarcasm`
A regex heuristic fires on patterns like `"just wonderful"`, `"oh great"`, `"thanks a lot"`. The pattern set is intentionally conservative. **It is not a reliable sarcasm detector** — it is a flag that a human reviewer may want to double-check.

Pattern source: [`src/api/api.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/api/api.py) — `_SARCASM_PATTERNS`.

---

## Known Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| Sarcasm | `"Absolutely fantastic — if you enjoy broken products"` will likely predict **positive** | LLM Judge second opinion on uncertain predictions |
| Non-English text | French, Spanish, German reviews will be misclassified without warning | No current mitigation — filter at ingest if applicable |
| Highly technical jargon | Niche B2B reviews with domain-specific vocabulary may be poorly tokenised | No current mitigation |
| Short reviews | Single words or emoji-only reviews have low textual signal | `short_review` flag alerts downstream consumers |
| Novel slang | Post-training vocabulary shifts (new product categories, internet slang) will degrade accuracy silently | Drift monitoring catches this over time |

---

## Semantic Search Alignment Signals

When `include_similar_reviews: true` is passed to the `/predict` endpoint, the API retrieves semantically similar historical reviews and computes an **alignment signal** comparing their known labels to the current prediction.

| Signal | Condition | Interpretation |
|--------|-----------|----------------|
| `STRONG_SUPPORT` | ≥ 80% of similar reviews match the prediction | High confidence the prediction is directionally correct |
| `MIXED` | 40–80% match | Ambiguous — the similar reviews are split |
| `CONTRADICTS` | < 40% match | Historical evidence suggests the prediction may be wrong |

Source: [`src/search/searcher.py`](https://github.com/BUZEL-112/ReviewSentinel/blob/main/src/search/searcher.py) lines 108–113.

---

## LLM Judge Integration

When a prediction falls in the uncertainty window (0.40–0.60), the review is written to a SQLite queue (`artifacts/llm_judge/review_queue.db`). A Prefect flow (`judge_processing_flow`) runs every 4 hours, dequeuing these reviews and running them through **Mistral 7B via Ollama** (`nemotron-3-nano:4b` by default — configurable in `pipeline_params.yaml`).

If the LLM Judge disagrees with the primary classifier, the disagreement is logged as a **conflict** in `artifacts/llm_judge/conflicts.db`. These conflicts are automatically ingested into the next training run, providing a continuous active-learning signal.

See [Training Guide](../02-guides/training-pipeline.md) for how conflicts enter the training set.

---

## Limitations Summary

- Trained on a single product category (`All_Beauty`) — generalisation to other categories is expected but not validated
- English-only
- No fairness or demographic bias evaluation has been performed on this model
- Sarcasm detection is heuristic-only; the model itself has no sarcasm awareness
- Performance degrades as product vocabulary evolves — the drift monitoring system is the primary mitigation
