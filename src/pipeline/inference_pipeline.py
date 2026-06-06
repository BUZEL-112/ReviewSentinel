"""
Inference Pipeline Module - DistilBERT Sentiment + SetFit Aspect Classification

This module provides production-ready inference for the new transformer pipeline.
It supports single and batch prediction and optionally enriches each prediction
with an aspect category via the SetFit model.

Key Features:
    - Loads a fine-tuned DistilBERT (or any AutoModel) from disk for sentiment prediction.
    - Applies the same minimal text cleaning used during training.
    - Supports both single-sample and batch inference.
    - Optionally runs the SetFit aspect classifier on the same inputs.
    - Saves results to CSV when configured to do so.

Sentiment labels:  0 = negative | 1 = neutral | 2 = positive
Aspect labels:     See src/data/aspect_data.py
"""

import os, sys
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


# Integer -> sentiment string mapping (must match training label order)
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class InferencePipeline:
    """
    Runs sentiment inference using a fine-tuned transformer model.
    Optionally enriches results with aspect classification via SetFit.
    """

    def __init__(self, config_path: str = str(Path("configs/inference_pipeline.yaml"))):
        """
        Initializes the pipeline by loading config, tokenizer, and model.

        Args:
            config_path (str): Path to the inference YAML configuration file.
        """
        try:
            logger.info("Initializing InferencePipeline...")
            self.config = load_config(config_path)
            infer_cfg = self.config.get("inference_pipeline", {})

            # Paths and runtime options loaded from config
            self.model_dir        = infer_cfg.get("model_dir",        "artifacts/models/distilbert-finetuned/")
            self.max_len          = infer_cfg.get("max_len",           512)
            self.batch_separator  = infer_cfg.get("batch_separator",   "|||")
            self.save_results     = infer_cfg.get("save_results",      False)
            self.output_path      = infer_cfg.get("output_path",       "artifacts/inference/inference_results.csv")
            self.run_aspect       = infer_cfg.get("run_aspect_model",  False)
            self.aspect_model_dir = infer_cfg.get("aspect_model_dir",  "artifacts/models/aspect_model")

            # Determine compute device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load transformer tokenizer + model
            self.tokenizer, self.model = self._load_model()

            # Optionally load the SetFit aspect model
            self.aspect_model = None
            if self.run_aspect:
                self.aspect_model = self._load_aspect_model()

            logger.info(f"InferencePipeline ready — device: {self.device}")

        except Exception as e:
            logger.error("Failed to initialize InferencePipeline.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Loads the fine-tuned tokenizer and sentiment model from disk.

        Returns:
            tuple: (AutoTokenizer, AutoModelForSequenceClassification)

        Raises:
            FileNotFoundError: If model_dir does not exist on disk.
        """
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(
                    f"No model directory found at: '{self.model_dir}'. "
                    "Run the training pipeline first."
                )
            logger.info(f"Loading tokenizer and model from: {self.model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self.device)
            # Disable dropout layers for stable inference
            model.eval()
            logger.info("Transformer model loaded and set to eval mode.")
            return tokenizer, model

        except Exception as e:
            logger.error("Error loading transformer model.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _load_aspect_model(self):
        """
        Loads the saved SetFit aspect model from disk.
        Imported lazily so SetFit is only required when aspect inference is enabled.

        Returns:
            AspectModel: Ready for inference.
        """
        try:
            from src.models.aspect_model import AspectModel
            logger.info(f"Loading aspect model from: {self.aspect_model_dir}")
            aspect = AspectModel(save_dir=self.aspect_model_dir)
            aspect.load()
            return aspect
        except Exception as e:
            logger.error("Error loading aspect model.")
            raise CustomException(e)

    # ------------------------------------------------------------------
    def _minimal_clean(self, text: str) -> str:
        """
        Applies the same minimal cleaning used during training:
        removes URLs and collapses whitespace without touching case or punctuation.

        Args:
            text (str): Raw input string.

        Returns:
            str: Cleaned text.
        """
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    def _build_texts(self, title: str, text: str, batch_mode: bool) -> list:
        """
        Combines title and body text into a cleaned list of strings.

        Args:
            title (str):      Single or separator-joined batch of titles.
            text (str | None): Single or separator-joined batch of bodies.
            batch_mode (bool): Whether to split on batch_separator.

        Returns:
            list[str]: Cleaned, combined strings ready for tokenization.
        """
        if batch_mode:
            titles = [t.strip() for t in title.split(self.batch_separator)]
            bodies = (
                [b.strip() for b in text.split(self.batch_separator)]
                if text
                else [""] * len(titles)
            )
        else:
            titles = [title]
            bodies = [text if text else ""]

        # Concatenate title + body the same way as training
        combined = [
            self._minimal_clean(f"{t} {b}".strip())
            for t, b in zip(titles, bodies)
        ]
        return combined

    # ------------------------------------------------------------------
    def _predict_sentiment(self, texts: list) -> list:
        """
        Tokenizes cleaned text and runs a forward pass through the model.

        Args:
            texts (list[str]): Pre-cleaned input strings.

        Returns:
            list[dict]: Each entry contains:
                - 'text'       : the input string
                - 'label'      : winning sentiment class
                - 'confidence' : softmax probability for the winning class
                - 'scores'     : dict of all class probabilities {negative, neutral, positive}
        """
        try:
            encodings = self.tokenizer(
                texts,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # Send tensors to the active device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                logits = self.model(**encodings).logits

            # Convert logits to softmax probabilities
            probs    = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_ids = probs.argmax(axis=1)

            results = []
            for i, (pid, prob_row) in enumerate(zip(pred_ids, probs)):
                results.append({
                    "text":       texts[i],
                    "label":      SENTIMENT_MAP[int(pid)],
                    "confidence": round(float(prob_row[pid]), 4),
                    # Full class distribution — used by the API for richer responses
                    "scores": {
                        "negative": round(float(prob_row[0]), 4),
                        "neutral":  round(float(prob_row[1]), 4),
                        "positive": round(float(prob_row[2]), 4),
                    },
                })
            return results

        except Exception as e:
            logger.error("Error during sentiment prediction.")
            raise CustomException(e)


    # ------------------------------------------------------------------
    def run(self, title: str, text: str = None, batch_mode: bool = False) -> pd.DataFrame:
        """
        Main entry point — runs sentiment (and optionally aspect) inference.

        Args:
            title (str):       Review title, or separator-joined batch of titles.
            text (str):        Review body, or separator-joined batch of bodies.
            batch_mode (bool): Set True when title/text contain multiple samples.

        Returns:
            pd.DataFrame: One row per input with columns:
                          'text', 'label', 'confidence'
                          and 'aspect' if aspect model is enabled.
        """
        try:
            logger.info("Starting inference...")

            # 1. Build and clean input texts
            texts = self._build_texts(title, text, batch_mode)
            logger.info(f"Running inference on {len(texts)} sample(s).")

            # 2. Sentiment classification
            sentiment_results = self._predict_sentiment(texts)
            df = pd.DataFrame(sentiment_results)

            # 3. Aspect classification (optional)
            if self.aspect_model is not None:
                logger.info("Running aspect classification...")
                aspect_preds = self.aspect_model.predict(texts)
                df["aspect"] = [p["aspect"] for p in aspect_preds]

            # 4. Optionally persist results to CSV
            if self.save_results:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                df.to_csv(self.output_path, index=False)
                logger.info(f"Results saved to: {self.output_path}")

            logger.info("Inference completed successfully.")
            return df

        except Exception as e:
            logger.error("Inference pipeline failed.")
            raise CustomException(e)


# ------------------------------------------------------------------
# Quick smoke test
if __name__ == "__main__":
    pipe = InferencePipeline(config_path="configs/inference_pipeline.yaml")

    # --- Single sample ---
    result = pipe.run(
        title="Foundation oxidized terribly",
        text="This foundation turned orange after a few minutes of application. "
             "It looked fine at first, but quickly darkened and made my skin patchy.",
        batch_mode=False,
    )
    print("\nSingle Sample Result:")
    print(result.to_string(index=False))

    # --- Batch mode ---
    titles = (
        "Mascara clumps everywhere|||"
        "Caused irritation and redness|||"
        "Too expensive for the quality|||"
        "Pump stopped working|||"
        "Overhyped product"
    )
    bodies = (
        "The formula is too thick and clumps on my lashes no matter how carefully I apply it.|||"
        "After two days of use, my skin became itchy and red. Had to stop immediately.|||"
        "The packaging looks fancy, but the product performs like a cheap drugstore brand.|||"
        "The pump broke after two uses, making it impossible to get any product out.|||"
        "Everyone raved about it, but I saw no difference in my skin after a month."
    )
    batch_result = pipe.run(title=titles, text=bodies, batch_mode=True)
    print("\nBatch Sample Results:")
    print(batch_result.to_string(index=False))
