"""
Model Building Module - Transformer Models for Sentiment Analysis

This module is responsible for loading and configuring transformer-based 
models (e.g., DistilBERT) for sequence classification. 
It replaces the older Keras-based architectures.
"""

import os
import yaml
import torch
from transformers import AutoModelForSequenceClassification
from src.utils.logger import logger
from src.utils.exception import CustomException

class ModelBuilder:
    """
    A builder class for instantiating transformer sequence classification models.
    """

    def __init__(self, config_dict: dict = None, config_path: str = None):
        """
        Initializes the ModelBuilder with a configuration.
        
        Args:
            config_dict (dict, optional): A dictionary containing model configurations.
            config_path (str, optional): A path to a YAML configuration file.
            
        Raises:
            ValueError: If neither config_dict nor config_path is provided.
            FileNotFoundError: If the specified config_path does not exist.
        """
        try:
            if config_path:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at: {config_path}")
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            elif config_dict is not None:
                self.config = config_dict
            else:
                raise ValueError("Either config_path or config_dict must be provided")
        except Exception as e:
            logger.error("Failed to initialize ModelBuilder.")
            raise CustomException(e)

    def build_model(self, model_type: str = "distilbert", custom_params: dict = None, show_summary: bool = False):
        """
        Builds and loads the transformer model for sequence classification.
        
        Args:
            model_type (str): Type of the model (currently 'distilbert' is assumed).
            custom_params (dict, optional): Custom parameters to override the default config.
            show_summary (bool): If True, logs the model architecture.
            
        Returns:
            torch.nn.Module: The loaded PyTorch transformer model on the appropriate device.
        """
        try:
            if model_type != "distilbert":
                logger.warning(f"Requested model type '{model_type}' is unconventional. Proceeding with transformer logic.")

            # Resolve configuration parameters
            cfg = custom_params or self.config.get("distilbert_model", {})
            model_name = cfg.get("model_name", "distilbert-base-uncased")
            num_labels = cfg.get("num_labels", 3)

            logger.info(f"Loading '{model_name}' with {num_labels} labels...")
            
            # Determine optimal available device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use AutoModelForSequenceClassification for maximum flexibility
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            ).to(device)

            logger.info(f"Model loaded successfully on {device}")
            
            # Display model structure if requested
            if show_summary:
                logger.info(f"Model Summary:\n{model}")

            return model

        except Exception as e:
            logger.error(f"Error occurred while building the model.")
            raise CustomException(e)