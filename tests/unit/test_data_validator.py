"""
tests/unit/test_data_validator.py

Unit tests for DataValidator (src/orchestration/validation.py).

No mocks needed — DataValidator is pure input/output logic over a DataFrame.
The validator config uses min_row_count=3 to keep test DataFrames tiny.
"""

import pytest
import pandas as pd
from src.orchestration.validation import DataValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator():
    """DataValidator configured for small test datasets."""
    return DataValidator(validation_config={
        "max_null_ratio": 0.10,
        "max_class_imbalance": 0.80,
        "min_row_count": 3,
    })


@pytest.fixture
def valid_df():
    """Balanced 5-row DataFrame that passes every validation rule."""
    return pd.DataFrame({
        "rating": [5, 1, 3, 2, 4],
        "title":  ["Great!", "Terrible", "Okay", "Bad packaging", "Works well"],
        "text": [
            "Loved it.",
            "Fell apart.",
            "Does the job.",
            "Arrived crushed.",
            "Solid product.",
        ],
    })


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

def test_valid_dataframe_passes(validator, valid_df):
    result = validator.validate(valid_df)
    assert result.success is True
    assert result.failed_expectations == []


def test_valid_df_row_count_returned(validator, valid_df):
    result = validator.validate(valid_df)
    assert result.row_count == 5


def test_valid_df_class_distribution_populated(validator, valid_df):
    result = validator.validate(valid_df)
    assert isinstance(result.class_distribution, dict)
    assert len(result.class_distribution) == 5   # one entry per rating


# ---------------------------------------------------------------------------
# Null-ratio failures
# ---------------------------------------------------------------------------

def test_high_text_null_ratio_fails(validator):
    """33 % null text exceeds the 10 % threshold."""
    df = pd.DataFrame({
        "rating": [5, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["Good", None, "Ok"],
    })
    result = validator.validate(df)
    assert result.success is False
    assert any("null" in e.lower() for e in result.failed_expectations)


def test_text_null_at_threshold_passes(validator):
    """Exactly 0 % null — right at the clean edge."""
    df = pd.DataFrame({
        "rating": [5, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["Good", "Bad", "Ok"],
    })
    result = validator.validate(df)
    assert result.success is True


# ---------------------------------------------------------------------------
# Class imbalance failures
# ---------------------------------------------------------------------------

def test_class_imbalance_fails(validator):
    """90 % rating-5 rows exceed the 80 % imbalance threshold."""
    df = pd.DataFrame({
        "rating": [5] * 9 + [1],
        "title":  [str(i) for i in range(10)],
        "text":   ["text"] * 10,
    })
    result = validator.validate(df)
    assert result.success is False
    assert any("imbalance" in e.lower() for e in result.failed_expectations)


def test_balanced_distribution_passes(validator, valid_df):
    """One row per rating — no class dominates."""
    result = validator.validate(valid_df)
    assert result.success is True


# ---------------------------------------------------------------------------
# Row count failures
# ---------------------------------------------------------------------------

def test_below_min_row_count_fails(validator):
    """2 rows < min_row_count=3."""
    df = pd.DataFrame({
        "rating": [5, 1],
        "title":  ["A", "B"],
        "text":   ["Good", "Bad"],
    })
    result = validator.validate(df)
    assert result.success is False
    assert any("minimum" in e.lower() for e in result.failed_expectations)


def test_exactly_min_row_count_passes(validator):
    """Exactly 3 rows == min_row_count=3 — edge case, must pass."""
    df = pd.DataFrame({
        "rating": [5, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["Good", "Bad", "Ok"],
    })
    result = validator.validate(df)
    # May fail for other reasons (imbalance) but NOT for row count
    row_count_failures = [e for e in result.failed_expectations if "minimum" in e.lower()]
    assert row_count_failures == []


# ---------------------------------------------------------------------------
# Schema / column failures
# ---------------------------------------------------------------------------

def test_wrong_column_names_fails(validator):
    """Wrong column names should trigger the schema check."""
    df = pd.DataFrame({
        "stars":    [5, 1, 3],
        "headline": ["A", "B", "C"],
        "body":     ["x", "y", "z"],
    })
    result = validator.validate(df)
    assert result.success is False
    assert any("mismatch" in e.lower() or "column" in e.lower()
               for e in result.failed_expectations)


def test_extra_columns_triggers_schema_fail(validator):
    """An extra column means the column list differs from the expected schema."""
    df = pd.DataFrame({
        "rating": [5, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["x", "y", "z"],
        "extra":  [1, 2, 3],
    })
    result = validator.validate(df)
    assert result.success is False


# ---------------------------------------------------------------------------
# Pydantic row-level validation failures
# ---------------------------------------------------------------------------

def test_out_of_range_rating_fails(validator):
    """Rating of 6 violates ReviewRecord(ge=1, le=5)."""
    df = pd.DataFrame({
        "rating": [6, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["Good", "Bad", "Ok"],
    })
    result = validator.validate(df)
    assert result.success is False
    assert any("pydantic" in e.lower() or "schema" in e.lower() or "rows" in e.lower()
               for e in result.failed_expectations)


def test_rating_zero_fails(validator):
    """Rating of 0 also violates the ge=1 constraint."""
    df = pd.DataFrame({
        "rating": [0, 1, 3],
        "title":  ["A", "B", "C"],
        "text":   ["Good", "Bad", "Ok"],
    })
    result = validator.validate(df)
    assert result.success is False


# ---------------------------------------------------------------------------
# Multiple simultaneous failures
# ---------------------------------------------------------------------------

def test_multiple_failures_are_all_collected(validator):
    """
    A DataFrame that fails three separate rules should report all three
    in failed_expectations, not short-circuit on the first failure.
    Rules triggered: wrong columns + below min rows + (implicit) column mismatch.
    """
    df = pd.DataFrame({
        "stars": [5],
        "body":  ["x"],
    })
    result = validator.validate(df)
    assert result.success is False
    # At minimum the column mismatch and row count should be reported
    assert len(result.failed_expectations) >= 2
