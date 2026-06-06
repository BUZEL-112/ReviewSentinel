"""
Model Evaluation Module for Customer Review Sentiment Analysis Pipeline

Handles evaluation for both the transformer sentiment model (DistilBERT/RoBERTa etc.)
and the SetFit aspect classification model. Each model has its own evaluate method
since they have fundamentally different inference interfaces.

Transformer model:  evaluated via HuggingFace Trainer.evaluate()
Aspect model:       evaluated via AspectModel.predict() on sample inputs
"""

import os
import json
from src.utils.logger import logger
from src.utils.exception import CustomException


class ModelEvaluator:

    def __init__(self):
        self.metrics = {}

    def evaluate(self, trainer, test_dataset, model_name: str = "distilbert"):
        """
        Evaluate the transformer model on the held-out test set.

        Args:
            trainer:      HuggingFace Trainer instance from ModelTrainer.trainer
            test_dataset: SentimentDataset test split from ModelTrainer.test_dataset
            model_name:   Name key used to store and log metrics.

        Returns:
            dict: Evaluation metrics including loss, accuracy, f1, and runtime.
        """
        try:
            logger.info(f"Evaluating '{model_name}' on test dataset...")
            results = trainer.evaluate(test_dataset)

            metrics = {
                "loss":                results.get("eval_loss"),
                "accuracy":            results.get("eval_accuracy"),
                "f1":                  results.get("eval_f1"),
                "runtime":             results.get("eval_runtime"),
                "samples_per_second":  results.get("eval_samples_per_second"),
            }

            self.metrics[model_name] = metrics
            logger.info(
                f"{model_name} — "
                f"Loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed for '{model_name}': {e}")
            raise CustomException(e)

    def evaluate_aspect(self, aspect_model, sample_texts: list[str]):
        """
        Run inference on sample texts and log aspect predictions.
        SetFit evaluation metrics are already logged during AspectModel.train(),
        so this method serves as a sanity check on the saved model.

        Args:
            aspect_model: Loaded AspectModel instance (call .load() before passing).
            sample_texts: List of raw review strings to run inference on.

        Returns:
            list[dict]: Predictions with 'aspect' and 'aspect_id' per input.
        """
        try:
            logger.info(f"Running aspect model sanity check on {len(sample_texts)} samples...")
            predictions = aspect_model.predict(sample_texts)

            for text, pred in zip(sample_texts, predictions):
                logger.info(f"  Input : '{text}'")
                logger.info(f"  Aspect: {pred['aspect']} (id={pred['aspect_id']})")

            return predictions

        except Exception as e:
            logger.error(f"Aspect evaluation failed: {e}")
            raise CustomException(e)

    def save_results(self, results: dict, output_dir: str = "artifacts/evaluation"):
        """
        Persist evaluation metrics to a JSON file.

        Args:
            results:    Dict of {model_name: metrics_dict}.
            output_dir: Directory to write metrics.json into.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "metrics.json")
            with open(path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Metrics saved to {path}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
            raise CustomException(e)

    def save_best_model(self, model, output_dir: str = "artifacts/best_model"):
        """
        Save the transformer model using save_pretrained.
        SetFit model saving is handled internally by AspectModel.save().

        Args:
            model:      The fine-tuned transformer model (torch.nn.Module).
            output_dir: Directory to save model weights and config into.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            logger.info(f"Best model saved to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            raise CustomException(e)