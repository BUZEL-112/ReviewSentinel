"""
tests/integration/test_training_flow.py

Integration tests for the data-preprocessing + tokenisation pipeline.

Scope:
- Uses bert-tiny (prajjwal1/bert-tiny) instead of distilbert-base-uncased so
  the test runs in CI without heavy downloads.
- Validates the CleanDataBERT.prepare_datasets() contract end-to-end with real
  tokenisation on a tiny synthetic DataFrame.
- Does NOT invoke the Prefect flow or MLflow — those are tested via unit mocks.

Markers:
- @pytest.mark.slow  — excluded from the fast `ci-unit` target (> 30 s on cold start).
"""

import pytest
import logging
from unittest.mock import patch
from transformers import AutoTokenizer
from src.data.clean_data import CleanDataBERT


# ---------------------------------------------------------------------------
# Fixture: CleanDataBERT with bert-tiny
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaner():
    """
    CleanDataBERT with bert-tiny injected to avoid downloading DistilBERT.
    Bypasses __init__ config loading.
    """
    instance = CleanDataBERT.__new__(CleanDataBERT)
    instance.model_name = "prajjwal1/bert-tiny"
    instance.max_len = 64
    instance.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    instance.logger = logging.getLogger("test.cleaner")
    return instance


# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """
    50-row balanced DataFrame — large enough for an 80/10/10 split to work
    without producing empty datasets.
    """
    import pandas as pd
    import random, string

    random.seed(42)
    ratings = ([1, 2] * 10 + [3] * 10 + [4, 5] * 10)[:50]
    rows = []
    for r in ratings:
        word = "".join(random.choices(string.ascii_lowercase, k=6))
        rows.append({
            "rating": r,
            "title": f"Title {word}",
            "text": f"This product is {'great' if r >= 4 else 'okay' if r == 3 else 'bad'} because {word}.",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_prepare_datasets_returns_four_values(cleaner, small_df):
    """prepare_datasets must return (train, val, test, test_labels)."""
    result = cleaner.prepare_datasets(small_df.copy())
    assert len(result) == 4


@pytest.mark.slow
def test_prepare_datasets_train_is_non_empty(cleaner, small_df):
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert len(train) > 0


@pytest.mark.slow
def test_prepare_datasets_val_is_non_empty(cleaner, small_df):
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert len(val) > 0


@pytest.mark.slow
def test_prepare_datasets_test_is_non_empty(cleaner, small_df):
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert len(test) > 0


@pytest.mark.slow
def test_prepare_datasets_total_samples_matches_input(cleaner, small_df):
    """Total across all splits must equal the number of input rows."""
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert len(train) + len(val) + len(test) == len(small_df)


@pytest.mark.slow
def test_prepare_datasets_test_labels_align_with_test_dataset(cleaner, small_df):
    """test_labels list must have the same length as test_dataset."""
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert len(test_labels) == len(test)


@pytest.mark.slow
def test_prepare_datasets_labels_are_valid_integers(cleaner, small_df):
    """All labels must be in {0, 1, 2} — the integer encoding of {negative, neutral, positive}."""
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    assert all(label in (0, 1, 2) for label in test_labels)


@pytest.mark.slow
def test_sentimentdataset_item_has_required_keys(cleaner, small_df):
    """A single __getitem__ call must return the keys expected by HuggingFace Trainer."""
    import torch
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    item = train[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item


@pytest.mark.slow
def test_sentimentdataset_item_shapes(cleaner, small_df):
    """Tensors must have shape (max_len,) for input_ids and attention_mask."""
    import torch
    train, val, test, test_labels = cleaner.prepare_datasets(small_df.copy())
    item = train[0]
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["labels"].shape == torch.Size([])   # scalar
