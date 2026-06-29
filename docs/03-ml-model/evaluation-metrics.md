# Evaluation Metrics

The primary metric used to evaluate and gate model promotions is the **Macro F1 Score**.

## Interpreting F1 in 3-Class Sentiment
In a 3-class setup (Negative, Neutral, Positive), class imbalance is very common. Typically, "Neutral" reviews are underrepresented compared to strongly polarized reviews.
- We use **Macro F1** instead of Accuracy or Micro F1 to ensure the model isn't just blindly guessing the majority class. Macro F1 averages the F1 score of each class equally, heavily penalizing the model if it fails to detect the minority class.
- A high Precision on "Negative" means when the model flags a bad review, it's usually correct.
- A high Recall on "Negative" means the model finds most of the actual bad reviews.

## The Quality Gate (F1 ≥ baseline + 0.01)
During automated retraining, the `ModelEvaluator` compares the newly trained model against the currently deployed model (the baseline). 
- A new model is only promoted if `new_f1 >= baseline_f1 + 0.01`.
- **Why this threshold?** Deep learning models exhibit non-determinism during training. A new model might achieve a `0.002` higher F1 simply due to random seed variations in weight initialization or data shuffling. 
- Demanding a strict `0.01` (1 absolute point) improvement ensures we only absorb the operational cost of a deployment (and the O(n) FAISS index rebuild) when there is a meaningful, robust improvement in predictive power.
