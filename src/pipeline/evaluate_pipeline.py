"""
Evaluation Pipeline Module - DistilBERT Sentiment + SetFit Aspect

This module orchestrates the end-to-end evaluation of trained models.
It replaces the old Keras/sklearn approach with the new transformer pipeline.

Responsibilities:
    - Load raw data and pass it through ModelTrainer to get labelled datasets.
    - Run the HuggingFace Trainer's evaluate() on the held-out test set.
    - Optionally evaluate the SetFit aspect model on a set of sample texts.
    - Persist evaluation metrics to JSON and save the best transformer model.

Evaluation flow:
    1. Load config and data.
    2. Initialise ModelTrainer -> prepares train/val/test splits.
    3. Fine-tune (or load) the transformer model.
    4. Evaluate with ModelEvaluator.evaluate(trainer, test_dataset).
    5. Optionally evaluate AspectModel.
    6. Save metrics + best model via ModelEvaluator helpers.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


class EvaluationPipeline:
    """
    End-to-end evaluation pipeline for the transformer sentiment model
    and the optional SetFit aspect classifier.
    """

    def __init__(self, pipeline_config_path: str = "configs/pipeline_params.yaml"):
        """
        Loads and parses the evaluation pipeline configuration.

        Args:
            pipeline_config_path (str): Path to the pipeline YAML config.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        try:
            logger.info("Initializing EvaluationPipeline...")

            if not os.path.exists(pipeline_config_path):
                raise FileNotFoundError(f"Config not found: {pipeline_config_path}")

            with open(pipeline_config_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f)

            # Pull relevant sections from the config
            eval_pipeline_cfg  = full_cfg.get("evaluation_pipeline", {})
            self.data_cfg      = eval_pipeline_cfg.get("data", {})
            self.train_cfg     = eval_pipeline_cfg.get("training", {})
            self.eval_cfg      = eval_pipeline_cfg.get("evaluation", {})
            self.aspect_cfg    = eval_pipeline_cfg.get("aspect", {})

            logger.info(f"EvaluationPipeline config loaded from: {pipeline_config_path}")

        except Exception as e:
            logger.error("Failed to initialize EvaluationPipeline.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        """
        Loads the dataset for evaluation. Uses a pre-processed CSV if
        available, otherwise expects the caller to provide it.

        Returns:
            pd.DataFrame: Raw dataframe with at least 'title', 'text', 'rating' columns.

        Raises:
            FileNotFoundError: If no data source is configured or found.
        """
        try:
            data_path = self.data_cfg.get("data_path", "data/processed/cleaned_data.csv")

            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Data file not found at: '{data_path}'. "
                    "Ensure the data pipeline has been run first."
                )

            logger.info(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df

        except Exception as e:
            logger.error("Failed to load evaluation data.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _load_aspect_model(self):
        """
        Lazily imports and loads the trained SetFit aspect model.

        Returns:
            AspectModel: Loaded and ready for inference.
        """
        try:
            from src.models.aspect_model import AspectModel
            aspect_dir = self.aspect_cfg.get("model_dir", "artifacts/models/aspect_model")
            logger.info(f"Loading aspect model from: {aspect_dir}")
            aspect = AspectModel(save_dir=aspect_dir)
            aspect.load()
            return aspect
        except Exception as e:
            logger.error("Failed to load aspect model.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def run(self):
        """
        Executes the full evaluation pipeline:
            1. Load data.
            2. Run ModelTrainer to prepare datasets and fine-tune the model.
            3. Evaluate sentiment model on the held-out test set.
            4. Optionally evaluate the SetFit aspect model.
            5. Save metrics and best model to disk.

        Returns:
            dict: Evaluation metrics keyed by model name.
        """
        try:
            logger.info("Starting EvaluationPipeline run...")

            # 1. Load raw data
            df = self._load_data()

            # 2. Initialise trainer (splits data and builds datasets internally)
            model_config_path = self.train_cfg.get("config_path", "configs/config.yaml")
            trainer_obj = ModelTrainer(
                dataframe=df,
                yaml_config_path=model_config_path,
                target_column=self.train_cfg.get("target_column", "label"),
            )

            # 3. Fine-tune the transformer model
            # train_model() returns the model and populates trainer_obj.trainer
            # and trainer_obj.test_dataset for downstream evaluation.
            model_type = self.train_cfg.get("model_type", "distilbert")
            logger.info(f"Running ModelTrainer.train_model('{model_type}')...")
            trained_model = trainer_obj.train_model(model_type=model_type)

            # 4. Evaluate sentiment model on the test split
            evaluator = ModelEvaluator()
            sentiment_metrics = evaluator.evaluate(
                trainer=trainer_obj.trainer,
                test_dataset=trainer_obj.test_dataset,
                model_name=model_type,
            )

            all_results = {model_type: sentiment_metrics}

            # 5. Optionally evaluate the SetFit aspect model
            if self.aspect_cfg.get("run_evaluation", False):
                sample_texts = self.aspect_cfg.get("sample_texts", [
                    "It arrived two weeks late and the packaging was damaged.",
                    "The product is completely useless and doesn't work at all.",
                    "Feels like a counterfeit, not what was shown in the listing.",
                ])
                aspect_model = self._load_aspect_model()
                aspect_results = evaluator.evaluate_aspect(aspect_model, sample_texts)
                all_results["aspect_model"] = aspect_results
                logger.info("Aspect model evaluation complete.")

            # 6. Persist metrics to JSON
            save_dir = self.eval_cfg.get("save_dir", "artifacts/evaluation")
            evaluator.save_results(all_results, output_dir=save_dir)

            # 7. Save the best (and only) transformer model
            best_model_dir = self.eval_cfg.get("best_model_dir", "artifacts/best_model")
            evaluator.save_best_model(trained_model, output_dir=best_model_dir)

            logger.info("EvaluationPipeline completed successfully.")
            return all_results

        except Exception as e:
            logger.error(f"EvaluationPipeline run failed: {e}")
            raise CustomException(e)


# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        pipeline = EvaluationPipeline(
            pipeline_config_path="configs/pipeline_params.yaml"
        )
        results = pipeline.run()
        logger.info(f"Final evaluation results: {results}")
    except Exception as e:
        logger.error(f"Fatal error in EvaluationPipeline: {e}")
