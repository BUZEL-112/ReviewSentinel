# ADR 001: DistilBERT over Larger Language Models

**Date:** June 2026  
**Status:** Accepted

## Context

ReviewSentinel requires a core sequence classification model to predict sentiment (negative, neutral, positive) from customer reviews. The model must be served synchronously behind a FastAPI endpoint.

We evaluated several architectures:
1. **Large Language Models (LLMs)** (e.g., Llama 3, Mistral 7B) running zero-shot or few-shot inference.
2. **BERT-base / RoBERTa-base** (110M–125M parameters).
3. **DistilBERT** (66M parameters).
4. **SetFit / Sentence Transformers** (few-shot contrastive learning).
5. **Traditional ML** (Logistic Regression over TF-IDF).

Our operational constraints:
- **Cost:** We want to run inference on CPU instances in Kubernetes to keep infrastructure costs low. GPU nodes are too expensive for 24/7 API availability at our current scale.
- **Latency:** The prediction endpoint should return results in under 100ms.
- **Accuracy:** Must significantly outperform a TF-IDF baseline, especially on nuanced language (e.g., "Not exactly what I was hoping for").

## Decision

We chose **DistilBERT** (`distilbert-base-uncased`) fine-tuned via the standard HuggingFace `Trainer` API.

## Rationale

1. **CPU Inference Latency:** DistilBERT is 40% smaller and 60% faster than BERT-base. On a standard cloud vCPU, a DistilBERT forward pass (sequence length 128) takes ~30–50ms. LLMs cannot meet the 100ms latency budget on CPU.
2. **Accuracy Trade-off:** DistilBERT retains 97% of BERT's language understanding capabilities. The 3% drop in accuracy is an acceptable trade-off for the massive gain in inference speed and reduction in memory footprint.
3. **Training Overhead:** Fine-tuning DistilBERT on our dataset takes hours on a single GPU, compared to the complexity of fine-tuning an LLM.
4. **Why not SetFit?** While SetFit excels in few-shot scenarios, we have access to a large labelled dataset (Amazon Reviews). Standard fine-tuning of a classification head outperforms few-shot SetFit when $N > 10,000$.

## Consequences

- **Positive:** Inference is fast and cheap. The entire API container requires < 1GB of RAM.
- **Negative:** DistilBERT struggles with highly contextual sarcasm and implicit sentiment compared to modern LLMs.
- **Mitigation:** To address the negative consequence, we introduced the LLM Judge pattern. DistilBERT handles the "easy" 90% of traffic, and we queue the uncertain 10% (confidence 0.40–0.60) for asynchronous review by an LLM.
