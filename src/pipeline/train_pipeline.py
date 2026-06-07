"""
Training Pipeline Module for Customer Review Sentiment Analysis

This module provides end-to-end orchestration of the complete machine learning training workflow.
It provides functionality to:
    - Orchestrate multiple pipeline stages in a configurable sequence:
        * Data ingestion - downloading and loading raw review data
        * Data cleaning - preprocessing, tokenization, and vectorization
        * Model training - training one or more ML/DL models
        * Model evaluation - computing metrics and selecting the best model
    - Load pipeline configuration from YAML files specifying:
        * Pipeline stages to execute in order
        * Training parameters (config paths, target columns, model types)
        * Evaluation parameters (output directories for metrics and models)
    - Coordinate data flow between pipeline stages:
        * Pass cleaned data from preprocessing to training
        * Pass trained models and test data to evaluation
    - Execute selective or complete workflows based on configuration
    - Handle errors gracefully with comprehensive logging at each stage
    - Support flexible pipeline customization without code changes

The TrainingPipeline class orchestrates the entire workflow by reading a YAML configuration
that defines which stages to run and their parameters. This allows easy modification of the
pipeline flow (e.g., skipping data loading if data is already available) and experimentation
with different model types and evaluation strategies. The module integrates all components
of the ML pipeline into a cohesive, production-ready training workflow.
"""

import os, sys
import yaml

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src.data.load_data import LoadData
from src.data.clean_data import CleanDataBERT
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.utils.logger import logger
from src.utils.exception import CustomException


class TrainingPipeline:
    """End-to-end orchestrator for the ML training workflow."""

    def __init__(self, pipeline_config_path="config/pipeline_params.yaml"):
        try:
            logger.info("Initializing Training Pipeline...")
            if not os.path.exists(pipeline_config_path):
                raise FileNotFoundError(
                    f"Pipeline config not found at: {pipeline_config_path}"
                )

            with open(pipeline_config_path, "r") as f:
                self.pipeline_config = yaml.safe_load(f).get("training_pipeline", {})

            # Parse configuration sections
            self.pipeline_stages = self.pipeline_config.get("pipeline", {}).get(
                "stages", []
            )
            self.training_cfg = self.pipeline_config.get("training", {})
            self.eval_cfg = self.pipeline_config.get("evaluation", {})

            logger.info(f"Loaded pipeline configuration from {pipeline_config_path}")

        except Exception as e:
            logger.error("Failed to initialize TrainingPipeline.")
            raise CustomException(e)

    def run(self):
        """Run the entire pipeline as per defined stages."""
        try:
            logger.info("Starting Training Pipeline Execution...")

            df = None
            trainer = None

            # Iterate over the configured stages
            for stage in self.pipeline_stages:
                logger.info(f"Running stage: {stage}")

                if stage == "load_data":
                    df = LoadData().load_data()

                elif stage == "clean_data":
                    continue
                #     cleaner = CleanDataBERT()
                #     # The prepare_datasets method handles cleaning, labeling, and splitting
                #     self.train_dataset, self.val_dataset, self.test_dataset, self.test_labels = cleaner.prepare_datasets(df)
                #     cleaner.tokenizer.save_pretrained(self.output_dir)
                #     logger.info(f"Datasets ready — Train: {len(self.train_dataset)}, "
                #             f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

                ## the cleandata has been previously integrated the model trainer 

                elif stage == "train_model":
                    if df is None:
                        raise CustomException("No data available for training.")
                    trainer = ModelTrainer(
                        dataframe=df.sample(frac=0.1).reset_index(drop=True),
                        yaml_config_path=self.training_cfg.get("config_path"),
                        target_column=self.training_cfg.get("target_column"),
                    )
                    model_type = self.training_cfg.get("model_type")
                    model = trainer.train_model(model_type)
                    logger.info(f"Model '{model_type}' trained successfully.")

                elif stage == "evaluate_model":
                    if trainer is None:
                        raise CustomException(
                            "Trainer not initialized before evaluation."
                        )
                    evaluator = ModelEvaluator()
                    
                    # 1. Fetch the HuggingFace dataset and trainer instance
                    test_dataset = trainer.test_dataset
                    hf_trainer = trainer.trainer
                    model_type = self.training_cfg.get("model_type")
                    
                    # 2. Pass the correct arguments matching evaluate_model.py
                    metrics = evaluator.evaluate(
                        hf_trainer, test_dataset, model_type
                    )
                    
                    evaluator.save_results(
                        {model_type: metrics}, output_dir=self.eval_cfg.get("save_dir")
                    )
                    
                    # 3. Remove the extra 'model_type' string argument here
                    evaluator.save_best_model(
                        trainer.models[model_type],
                        output_dir=self.eval_cfg.get("best_model_dir"),
                    )
                    logger.info(f"Evaluation completed for model: {model_type}")

                else:
                    logger.warning(f"Unknown stage '{stage}' — skipping.")

            logger.info("Training pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline(pipeline_config_path="configs/pipeline_params.yaml")
        pipeline.run()
    except Exception as e:
        logger.error(f"Fatal error in training pipeline: {e}")
