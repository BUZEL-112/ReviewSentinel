"""
Data Preprocessing & Tokenization Module for BERT

This module provides a specialized pipeline for cleaning and preparing customer review data
specifically for Transformer-based models (e.g., DistilBERT). It handles sentiment
classification, text normalization, and WordPiece tokenization.

Key Features:
    - Sentiment Labeling: Maps numeric ratings (1-5) to sentiment categories.
    - Minimal Cleaning: Removes URLs and excessive whitespace while preserving linguistic structure.
    - BERT Tokenization: Handles batch tokenization using the HuggingFace Transformers library.
    - Configuration Driven: Parameters like model name and max length are loaded from YAML.
"""

import re
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.config_parser import load_config


class CleanDataBERT:
    """
    Handles data preprocessing and tokenization for BERT-based sentiment analysis.

    This class simplifies the transition from raw dataframes to model-ready encodings.
    """

    def __init__(self, config_path: str = str(Path("configs/config.yaml"))):
        """
        Initializes the preprocessing pipeline with configuration settings.

        Args:
            config_path (str): Path to the project configuration file.
        """
        try:
            self.config_path = config_path
            self.config = load_config(config_path)

            # Extract BERT-specific configurations
            bert_cfg = self.config.get("clean_data_bert", {})
            self.model_name = bert_cfg.get("model_name", "distilbert-base-uncased")
            self.max_len = bert_cfg.get("max_len", 512)

            self.logger = logger
            self.logger.info(
                f"Initializing CleanDataBERT with model: {self.model_name}"
            )

            # Load the pre-trained tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        except Exception as e:
            # Wrap standard exceptions in our custom handler
            raise CustomException(e)

    def _label_sentiment(self, rating):
        """
        Maps numeric ratings to categorical labels:
        1-2 -> negative, 3 -> neutral, 4-5 -> positive.

        Args:
            rating (int): The numeric star rating.

        Returns:
            str: The sentiment label.
        """
        if rating in [1, 2]:
            return "negative"
        elif rating == 3:
            return "neutral"
        elif rating in [4, 5]:
            return "positive"
        else:
            raise ValueError("Rating must be an integer from 1 to 5")

    def _minimal_clean(self, text):
        """
        Performs minimal text cleaning suitable for BERT.
        Preserves case and punctuation but removes noise like URLs.

        Args:
            text (str): The raw input text.

        Returns:
            str: Cleaned text.
        """
        text = str(text)
        # Remove URLs (HTTP/WWW)
        text = re.sub(r"http\S+|www\S+", "", text)
        # Normalize whitespace (replace tabs/newlines with single space)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text):
        """
        Tokenizes a single string for DistilBERT.

        Args:
            text (str): Input text to tokenize.

        Returns:
            dict: Tokenized output including input_ids and attention_mask.
        """
        try:
            return self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        except Exception as e:
            self.logger.error("Error occurred during tokenization of single text.")
            raise CustomException(e)

    def _clean_data(self, df):
        """
        Executes the full cleaning pipeline on a DataFrame.
        Tokenization is deferred to SentimentDataset.__getitem__ to avoid
        materialising the entire padded tensor dataset in memory at once.

        Args:
            df (pd.DataFrame): Input dataframe containing 'title', 'text', and 'rating'.

        Returns:
            pd.DataFrame: Processed DataFrame with 'clean_text' and 'sentiment' columns.
        """
        try:
            self.logger.info(
                "Starting minimal data cleaning for BERT..."
            )

            # 1. Apply sentiment labeling
            df["sentiment"] = df["rating"].apply(self._label_sentiment)

            # 2. Combine title and text fields
            df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

            # 3. Apply minimal cleaning
            df["clean_text"] = df["full_text"].apply(self._minimal_clean)

            # 3.5 Ingest LLM Judge conflicts if present
            export_path = "artifacts/llm_judge/exported_conflicts_for_training.csv"
            if Path(export_path).exists():
                try:
                    conflicts_df = pd.read_csv(export_path)
                    conflicts_df["clean_text"] = conflicts_df["text"].apply(
                        self._minimal_clean
                    )
                    # Align columns
                    conflicts_df = conflicts_df[["clean_text", "sentiment"]]
                    df = pd.concat([df, conflicts_df], ignore_index=True)
                    self.logger.info(
                        f"Ingested {len(conflicts_df)} conflict records from LLM Judge into training data."
                    )
                except Exception as e:
                    self.logger.warning(f"Could not ingest conflicts: {e}")

            # Tokenization is now lazy — done per sample inside SentimentDataset.
            self.logger.info(
                f"Cleaning complete. {len(df)} samples ready for lazy tokenization."
            )
            return df

        except Exception as e:
            self.logger.error(
                "Error occurred during CleanDataBERT clean_data execution."
            )
            raise CustomException(e)

    def prepare_datasets(self, df):
        try:
            df = self._clean_data(df)
            label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
            df["label"] = df["sentiment"].map(label_mapping)

            # Split on plain Python lists — no tensors in memory yet.
            texts = df["clean_text"].tolist()
            labels = df["label"].tolist()

            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                texts, labels, test_size=0.3, random_state=42
            )
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, random_state=42
            )

            # SentimentDataset tokenizes lazily inside __getitem__,
            # so no large tensor block is ever allocated up front.
            train_dataset = SentimentDataset(
                train_texts, train_labels, self.tokenizer, self.max_len
            )
            val_dataset = SentimentDataset(
                val_texts, val_labels, self.tokenizer, self.max_len
            )
            test_dataset = SentimentDataset(
                test_texts, test_labels, self.tokenizer, self.max_len
            )
            logger.info(
                f"Datasets ready — Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )

            return train_dataset, val_dataset, test_dataset, test_labels

        except Exception as e:
            self.logger.error(
                "Error occurred during CleanDataBERT prepare_datasets execution."
            )
            raise CustomException(e)


class SentimentDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that tokenizes text lazily inside __getitem__.

    Instead of pre-tokenizing the entire corpus (which allocates a huge
    padded tensor block up front), each sample is tokenized on the fly when
    the DataLoader requests it.  Only one batch worth of tensors lives in
    RAM / VRAM at any given moment.
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        """
        Args:
            texts (list[str]): Raw cleaned text strings.
            labels (list[int]): Integer class labels aligned with texts.
            tokenizer: HuggingFace tokenizer instance (shared, not copied).
            max_len (int): Maximum token sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize a single sample — only this sample's tensors exist in memory.
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item
