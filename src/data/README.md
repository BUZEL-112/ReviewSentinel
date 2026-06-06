# Data Module

This module handles the complete data ingestion, cleaning, and preprocessing pipeline for customer review analysis.

## Overview

The data module consists of two main components that work together to prepare customer review data for sentiment analysis:

1. **Data Loading** - Download and extract raw review data
2. **Data Cleaning** - Preprocess and vectorize text data

## Files

### `load_data.py`

Handles the data ingestion process for customer review datasets.

**Key Features:**
- Downloads compressed review data from remote sources
- Checks for existing files to avoid redundant downloads
- Unzips gzipped JSON Line (JSONL) files
- Parses JSON review records and extracts relevant fields (rating, title, text)
- Converts data into pandas DataFrame
- Saves a subset of reviews to CSV for downstream processing

**Main Class:** `LoadData`

**Usage:**
```python
from src.data.load_data import LoadData

# Initialize with default config
loader = LoadData()

# Download and prepare dataset
df = loader.load_data()
```

**Configuration:**
The module uses configuration from `configs/config.yaml`:
- `source_url`: URL to download the dataset from
- `save_dir`: Directory to save downloaded files
- `local_data_file`: Path to save the processed CSV file

---

### `clean_data.py`

Handles comprehensive text preprocessing and vectorization for customer review datasets.

**Key Features:**
- Text cleaning (lowercasing, noise removal)
- Sentiment labeling based on ratings
- Text tokenization using NLTK
- Stopword removal and lemmatization
- Word embedding vectorization using pre-trained models (GloVe or Word2Vec)
- Batch and single-instance processing support

**Main Class:** `CleanData`

**Usage:**
```python
from src.data.clean_data import CleanData

# Initialize with default config
cleaner = CleanData()

# Clean and vectorize training data
df_cleaned = cleaner.clean_data(save=True)

# Process new input for inference
result = cleaner.process_input(
    title="Great product!",
    text="I love this item. Highly recommended.",
    use_tokenizer=True
)
```

**Methods:**
- `clean_data(save=True)`: Main method to clean, tokenize, and vectorize the entire dataset
- `process_input(title, text, use_tokenizer, batch_mode, separator)`: Process new input(s) for inference

**Configuration:**
The module uses configuration from `configs/config.yaml`:
- `local_data_file`: Input CSV file path
- `clean_data_file`: Output path for cleaned data
- `embeddings_model_name`: Pre-trained embedding model to use
- `embeddings_model_dir`: Directory to store embedding models

## Workflow

```mermaid
graph LR
    A[Remote Data Source] -->|load_data.py| B[Raw CSV]
    B -->|clean_data.py| C[Cleaned & Vectorized Data]
    C --> D[Model Training]
```

1. **Data Loading**: Use `load_data.py` to download and extract raw customer reviews
2. **Data Cleaning**: Use `clean_data.py` to preprocess text and generate embeddings
3. **Ready for Training**: Cleaned data is ready for sentiment analysis model training

## Dependencies

- pandas
- numpy
- nltk
- gensim
- PyYAML

## Logging & Error Handling

Both modules use:
- Custom logger from `src.utils.logger`
- Custom exception handling from `src.utils.exception`
- Configuration parsing from `src.utils.config_parser`

All operations include comprehensive logging and error handling for production-ready data processing.
