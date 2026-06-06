"""
Data Ingestion Module for Customer Review Analysis Pipeline

This module handles the complete data ingestion process for customer review datasets.
It provides functionality to:
    - Download compressed review data from a remote source URL
    - Check for existing files to avoid redundant downloads
    - Unzip gzipped JSON Line (JSONL) files containing review data
    - Parse JSON review records and extract relevant fields (rating, title, text)
    - Convert the data into a pandas DataFrame
    - Save a subset of the data (first 1000 reviews) to a CSV file for downstream processing

The LoadData class orchestrates the entire ingestion workflow and uses configuration
from a YAML file to determine source URLs, save directories, and output file locations.
All operations include comprehensive logging and custom exception handling.
"""

import os
import sys
import gzip
import json
import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.utils import download_file
from src.utils.config_parser import load_config


class LoadData:
    """Handles data ingestion: download, unzip, and save dataset."""

    def __init__(self, raw_data_yaml: str = str(Path("configs/config.yaml"))):
        """
        Initialize the LOAD_DATA instance and load configuration for data ingestion

        :param self: Instance of the LOAD_DATA class
        :param raw_data_yaml: Path to the YAML configuration file
        :type raw_data_yaml: str
        """
        try:
            self.config = load_config(raw_data_yaml)
            self.logger = logger

            paths = self.config.get("paths", {})
            load_cfg = self.config.get("data_ingestion", {})

            self.source_url = load_cfg.get("source_url")
            self.save_dir = load_cfg.get("save_dir", "data/raw")
            self.local_data_file = load_cfg.get("local_data_file")

            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.local_data_file), exist_ok=True)

        except Exception as e:
            raise CustomException(e)

    def _check_file_exists(self):
        """
        Check if the file from the source URL already exists in the save directory.

        :return: Tuple of (bool: exists, str: file_path)
        """
        # Compute the expected file name and path (mirroring logic from download_file)
        filename = self.source_url.split("/")[-1]
        file_path = os.path.join(self.save_dir, filename)

        if os.path.exists(file_path):
            self.logger.info(f"File already exists: {file_path}")
            return True, file_path
        else:
            self.logger.info(
                f"File does not exist: {file_path}. Proceeding to download."
            )
            return False, file_path

    def _download_file(self):
        """
        Download a file from the configured source URL and update the download path,
        but only if it doesn't already exist.

        :param self: Instance of the LOAD_DATA class
        """
        try:
            # First, check if the file exists
            exists, file_path = self._check_file_exists()

            if exists:
                self.download_path = file_path
                self.logger.info(
                    f"Skipping download as file already exists: {self.download_path}"
                )
                return  # Exit early if file exists

            # If not, proceed with download
            self.logger.info(f"Downloading data from: {self.source_url}")
            file_path = download_file(self.source_url, self.save_dir)
            self.download_path = file_path
            self.logger.info(f"File downloaded successfully: {self.download_path}")

        except Exception as e:
            self.logger.error("Error during file download or check.")
            raise CustomException(e)

    def _unzip(self):

        try:

            self.logger.info(f"Unzipping file: {self.download_path}")
            reviews = []
            with gzip.open(str(self.download_path).replace("\\", "/"), "rb") as f:
                for line in f:
                    reviews.append(json.loads(line))
            df = pd.DataFrame(reviews)[["rating", "title", "text"]]
            df.to_csv(self.local_data_file, index=False)
            self.logger.info(f"Unzipped and saved dataset to {self.local_data_file}")
            return df

        except Exception as e:

            self.logger.error("Error during unzip or data parsing.")
            raise CustomException(e)

    def load_data(self):
        """End-to-end method to download and prepare dataset."""

        try:

            self._download_file()
            df = self._unzip()
            self.logger.info("Data ingestion completed successfully.")
            return df

        except Exception as e:

            self.logger.error("Data ingestion failed.")
            raise CustomException(e)
