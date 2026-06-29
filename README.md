# ReviewSentinel

ReviewSentinel is a fine-tuned DistilBERT classifier for three-class review sentiment (negative / neutral / positive), built around an MLOps pipeline rather than a one-shot script. The system handles quality-gated model promotion, scheduled drift monitoring with auto-retraining, and an async LLM second-opinion loop for low-confidence predictions.

[![CI](https://github.com/BUZEL-112/ReviewSentinel/actions/workflows/pr-checks.yaml/badge.svg)](https://github.com/BUZEL-112/ReviewSentinel/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Start Here

* **Want to run it right now?** Head to the [Quickstart](docs/00-quickstart/overview.md).
* **Want to see everything else?** Check the [Docs Index](docs/INDEX.md).

The documentation has been restructured into a persona-driven hierarchy. See the `docs/INDEX.md` for architecture details, API references, guides, runbooks, and design decisions (ADRs).

---

## Quick Test

If you already have the system running, here is a single curl command to test the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Broke after one week", "text": "Complete waste of money."}'
```
