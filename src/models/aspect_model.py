"""
Aspect Model Module - Few-Shot Aspect Classification using SetFit

This module handles training, saving, loading, and inference for the
aspect-based sentiment model. It runs entirely independently from the
DistilBERT sentiment pipeline and is only merged at inference time.

SetFit works by fine-tuning a sentence transformer using contrastive
learning on a small number of manually curated examples per aspect,
making it ideal for cases where large labeled datasets don't exist.
"""

import os
import torch
from setfit import SetFitModel, SetFitTrainer, TrainingArguments
from src.data.aspect_data import (
    build_setfit_dataset,
    get_id2label,
    get_label2id,
    validate_examples,
)
from src.utils.logger import logger
from src.utils.exception import CustomException


class AspectModel:
    """
    Trains, saves, loads and runs inference for aspect classification
    using SetFit few-shot learning.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        save_dir: str = "artifacts/models/aspect_model",
    ):
        """
        Args:
            model_name: Sentence transformer backbone for SetFit.
            save_dir:   Directory to save/load the trained aspect model.
        """
        self.model_name = model_name
        self.save_dir = save_dir
        self.model = None
        self.id2label = get_id2label()
        self.label2id = get_label2id()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, num_epochs: int = 1, batch_size: int = 16):
        """
        Fine-tune SetFit on the manually curated aspect examples.

        Args:
            num_epochs: Number of training epochs (1 is usually enough for SetFit).
            batch_size: Batch size for contrastive training.
        """
        try:
            logger.info("Validating aspect examples before training...")
            validate_examples()

            logger.info("Building few-shot dataset...")
            dataset = build_setfit_dataset()

            # SetFit works best with a small train split
            # since examples are already few-shot by design
            split = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]

            logger.info(f"Loading SetFit backbone: {self.model_name}")
            self.model = SetFitModel.from_pretrained(
                self.model_name,
            )

            training_args = TrainingArguments(
                num_epochs=num_epochs,
                batch_size=batch_size,
                logging_steps=5,
            )

            trainer = SetFitTrainer(
                model=self.model,
                batch_size=batch_size, num_epochs=num_epochs,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                metric="accuracy",
            )

            logger.info("Starting SetFit training...")
            trainer.train()

            metrics = trainer.evaluate()
            logger.info(f"Aspect model evaluation: {metrics}")

            self.save()
            return metrics

        except Exception as e:
            logger.error(f"Aspect model training failed: {e}")
            raise CustomException(e)

    def save(self):
        """Save the trained SetFit model to disk."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save_pretrained(self.save_dir)
            logger.info(f"Aspect model saved to {self.save_dir}")
        except Exception as e:
            raise CustomException(e)

    def load(self):
        """Load a previously trained SetFit model from disk."""
        try:
            tokenizer = self.model.model_body[0].tokenizer
            if "token_type_ids" in tokenizer.model_input_names:
                tokenizer.model_input_names.remove("token_type_ids")
        except Exception as e:
            pass
        try:
            # Check for the model head, not just the directory
            head_path = os.path.join(self.save_dir, "model_head.pkl")
            if not os.path.exists(head_path):
                raise FileNotFoundError(
                    f"No trained aspect model head found at {head_path}. "
                    f"Run train() first."
                )
            logger.info(f"Loading aspect model from {self.save_dir}")
            self.model = SetFitModel.from_pretrained(self.save_dir)
            logger.info("Aspect model loaded successfully.")
        except Exception as e:
            raise CustomException(e)

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Predict the aspect category for a list of review texts.

        Args:
            texts: List of raw review strings.

        Returns:
            List of dicts with 'aspect' and 'aspect_id' per input.

        Example:
            >>> model.predict(["arrived two weeks late"])
            [{"aspect": "shipping", "aspect_id": 1}]
        """
        try:
            tokenizer = self.model.model_body[0].tokenizer
            if "token_type_ids" in tokenizer.model_input_names:
                tokenizer.model_input_names.remove("token_type_ids")
        except Exception as e:
            pass
        try:
            if self.model is None:
                raise RuntimeError(
                    "Model not loaded. Call train() or load() first."
                )

            if isinstance(texts, str):
                texts = [texts]

            logger.info(f"Running aspect inference on {len(texts)} input(s)...")
            predictions = self.model(texts)

            results = []
            for pred in predictions:
                aspect_id = int(pred)
                results.append({
                    "aspect":    self.id2label[aspect_id],
                    "aspect_id": aspect_id,
                })

            return results

        except Exception as e:
            logger.error(f"Aspect prediction failed: {e}")
            raise CustomException(e)