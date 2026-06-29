# Training Details

The sentiment classification model is a fine-tuned transformer. We use a DistilBERT variant optimized for speed in our MLOps pipeline.

## Model Configuration
- **Base Model:** `prajjwal1/bert-tiny` (a compact version of BERT/DistilBERT architecture)
- **Objective:** 3-class sequence classification (Negative, Neutral, Positive)
- **Loss Function:** Cross-Entropy Loss

## Hyperparameters
These are defined in `configs/model_params.yaml` and tracked via MLflow:
- **Epochs:** 3
- **Batch Size (Train):** 14
- **Batch Size (Eval):** 4
- **Learning Rate:** 2.0e-5
- **Warmup Steps:** 100
- **Weight Decay:** 0.01

## Ablations
- **Larger Models:** Moving to `bert-base-uncased` improved F1 by ~0.04 but increased inference latency beyond our 100ms budget for single-core CPU serving. We stick to the tiny model for this pipeline.
