"""
Model Training Module - DistilBERT Fine-tuning

This module provides the training pipeline for fine-tuning a DistilBERT 
model on customer reviews. It integrates with HuggingFace Trainer and 
MLflow for experiment tracking.
"""

import os
import torch
import numpy as np
import yaml
import mlflow
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

from src.models.build_model import ModelBuilder
from src.data.clean_data import CleanDataBERT
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import resolve_tracking_uri


class ModelTrainer:
    """
    Orchestrates the fine-tuning of transformer models (e.g., DistilBERT)
    for sequence classification, including data preparation and MLflow tracking.
    """

    def __init__(self, dataframe, yaml_config_path, target_column="label"):
        # self.df = dataframe.copy()
        self.df = dataframe.sample(frac=0.001).copy()
        self.target_column = target_column

        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"Config not found: {yaml_config_path}")
        with open(yaml_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_builder = ModelBuilder(config_dict=self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = os.path.join("artifacts", "models", "distilbert")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup MLflow
        mlflow_cfg = self.config.get("mlflow", {})
        tracking_uri = mlflow_cfg.get("tracking_uri", "http://mlflow:5000")
        experiment_name = mlflow_cfg.get("experiment_name", "distilbert_training")
        # Convert Docker service URIs to localhost when needed to avoid MLflow tracking URI conflicts between containerized and local runs
        tracking_uri = resolve_tracking_uri(tracking_uri)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info("ModelTrainer initialized successfully with MLflow tracking.")

    def _compute_metrics(self, pred):
        """
        Calculates accuracy and F1 score during evaluation.
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    def _prepare_data(self):
        """Clean data, tokenize, split into train/val/test datasets."""
        try:
            logger.info("Preparing data for training...")
            cleaner = CleanDataBERT()
            # The prepare_datasets method handles cleaning, labeling, and splitting
            self.train_dataset, self.val_dataset, self.test_dataset, self.test_labels = cleaner.prepare_datasets(self.df)
            cleaner.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Datasets ready — Train: {len(self.train_dataset)}, "
                       f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

        except Exception as e:
            logger.error("Failed to prepare data.")
            raise CustomException(e)

    def train_model(self, model_type="distilbert", custom_params=None):
        """Fine-tune DistilBERT using HuggingFace Trainer and log to MLflow."""
        try:
            self._prepare_data()
            model = self.model_builder.build_model(model_type, custom_params)

            train_cfg = self.config.get("distilbert_model", {}).get("training", {})

            
            training_args = TrainingArguments(
                output_dir=train_cfg.get("output_dir", "./results"),
                num_train_epochs=train_cfg.get("epochs", 3),
                per_device_train_batch_size=train_cfg.get("train_batch_size", 16),
                per_device_eval_batch_size=train_cfg.get("eval_batch_size", 16),
                warmup_steps=train_cfg.get("warmup_steps", 100),
                weight_decay=train_cfg.get("weight_decay", 0.01),
                logging_dir=train_cfg.get("logging_dir", "./logs"),
                logging_steps=train_cfg.get("logging_steps", 10),
                save_steps=train_cfg.get("save_steps", 50),
                eval_strategy="epoch",  # evaluate at end of each epoch
                report_to=["mlflow"],         # explicitly enable MLflow tracking
            )

            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=self._compute_metrics
            )

            # Start an MLflow run to log custom details alongside Trainer's automatic logging
            with mlflow.start_run(run_name=f"{model_type}_finetuning"):
                logger.info(f"Starting {model_type.upper()} fine-tuning with MLflow tracking...")
                self.trainer.train()

                # Evaluate on test set
                logger.info("Evaluating on test dataset...")
                test_results = self.trainer.evaluate(eval_dataset=self.test_dataset)
                logger.info(f"Test Results: {test_results}")

                # Save model locally
                
                model.save_pretrained(self.output_dir)       # saves config + weights
                
                # Log model to MLflow explicitly for artifact tracking
                components = {
                    "model": model,
                    "tokenizer": CleanDataBERT().tokenizer
                }
                mlflow.transformers.log_model(
                    transformers_model=components,
                    artifact_path="model",
                    task="text-classification"
                )
                
                logger.info(f"Model saved to {self.output_dir} and logged to MLflow.")

            self.models = {model_type: model}
            return model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CustomException(e)