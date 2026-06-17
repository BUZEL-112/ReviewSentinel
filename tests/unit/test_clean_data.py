"""
tests/unit/test_clean_data.py

Unit tests for CleanDataBERT (src/data/clean_data.py).

Strategy:
- Instantiate CleanDataBERT manually (bypassing __init__ disk I/O) and inject
  a bert-tiny tokenizer so tests run without downloading DistilBERT.
- Focus exclusively on the two pure methods: _label_sentiment and _minimal_clean.
- Use pytest.mark.parametrize to avoid copy-paste test functions.
"""

import logging
import pytest
from transformers import AutoTokenizer
from src.data.clean_data import CleanDataBERT


# ---------------------------------------------------------------------------
# Module-scoped fixture — tokenizer download happens once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaner():
    """
    CleanDataBERT with a bert-tiny tokenizer injected directly.
    Bypasses __init__ to avoid reading YAML from disk.
    """
    instance = CleanDataBERT.__new__(CleanDataBERT)
    instance.model_name = "prajjwal1/bert-tiny"
    instance.max_len = 64
    instance.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    instance.logger = logging.getLogger("test.clean_data")
    return instance


# ---------------------------------------------------------------------------
# _label_sentiment — parametrized happy paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rating, expected", [
    (1, "negative"),
    (2, "negative"),
    (3, "neutral"),
    (4, "positive"),
    (5, "positive"),
])
def test_label_sentiment_valid_ratings(cleaner, rating, expected):
    assert cleaner._label_sentiment(rating) == expected


# ---------------------------------------------------------------------------
# _label_sentiment — error paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_rating", [0, 6, -1, 100])
def test_label_sentiment_invalid_rating_raises(cleaner, bad_rating):
    with pytest.raises(ValueError):
        cleaner._label_sentiment(bad_rating)


# ---------------------------------------------------------------------------
# _minimal_clean — URL removal
# ---------------------------------------------------------------------------

def test_minimal_clean_removes_http_url(cleaner):
    raw = "Check http://example.com for details"
    assert "http://" not in cleaner._minimal_clean(raw)


def test_minimal_clean_removes_https_url(cleaner):
    raw = "Visit https://example.com/path?q=1 now"
    result = cleaner._minimal_clean(raw)
    assert "https://" not in result
    assert "example.com" not in result


def test_minimal_clean_removes_www_url(cleaner):
    raw = "See www.example.com for more"
    result = cleaner._minimal_clean(raw)
    assert "www." not in result


def test_minimal_clean_preserves_surrounding_words(cleaner):
    """Words around the URL must survive cleaning."""
    raw = "Check https://example.com for details"
    result = cleaner._minimal_clean(raw)
    assert "Check" in result
    assert "details" in result


# ---------------------------------------------------------------------------
# _minimal_clean — whitespace normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected_absent", [
    ("Too   many    spaces here",    "  "),   # multiple consecutive spaces
    ("Tab\there",                    "\t"),   # tab character
    ("Newline\nhere",                "\n"),   # newline character
])
def test_minimal_clean_collapses_whitespace(cleaner, raw, expected_absent):
    assert expected_absent not in cleaner._minimal_clean(raw)


def test_minimal_clean_strips_leading_trailing_whitespace(cleaner):
    raw = "  hello world  "
    result = cleaner._minimal_clean(raw)
    assert result == result.strip()


# ---------------------------------------------------------------------------
# _minimal_clean — preservation guarantees
# ---------------------------------------------------------------------------

def test_minimal_clean_preserves_case(cleaner):
    raw = "Amazing! I Love it."
    result = cleaner._minimal_clean(raw)
    assert "Amazing" in result
    assert "Love" in result


def test_minimal_clean_preserves_punctuation(cleaner):
    raw = "Works great! Highly recommended."
    result = cleaner._minimal_clean(raw)
    assert "!" in result
    assert "." in result


def test_minimal_clean_empty_string_is_safe(cleaner):
    result = cleaner._minimal_clean("")
    assert isinstance(result, str)
    assert result == ""


def test_minimal_clean_none_coerces_to_string(cleaner):
    """str(None) -> 'None' — should not raise."""
    result = cleaner._minimal_clean(None)   # type: ignore[arg-type]
    assert isinstance(result, str)
