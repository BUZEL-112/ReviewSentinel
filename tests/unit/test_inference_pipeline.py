"""
tests/unit/test_inference_pipeline.py

Unit tests for InferencePipeline (src/pipeline/inference_pipeline.py).

Strategy:
- The model is mocked at AutoModelForSequenceClassification.from_pretrained so
  no weights are loaded from disk and no GPU is required.
- bert-tiny tokenizer (from conftest.mock_tokenizer) is injected so tokenization
  is real but cheap.
- Tests focus exclusively on the pipeline's own logic: _build_texts,
  _minimal_clean, and the DataFrame contract of run().
  Model accuracy is NOT tested here.
"""

import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from src.pipeline.inference_pipeline import InferencePipeline, SENTIMENT_MAP


# ---------------------------------------------------------------------------
# Fixture: fully initialised pipeline with mocked model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, mock_tokenizer):
    """
    InferencePipeline with a fake model_dir and mocked model weights.
    No disk I/O beyond creating a temporary directory, no GPU needed.
    """
    tmp_path = tmp_path_factory.mktemp("model")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    config = {
        "inference_pipeline": {
            "model_dir": str(model_dir),
            "max_len": 64,
            "batch_separator": "|||",
            "save_results": False,
            "output_path": str(tmp_path / "out.csv"),
            "run_aspect_model": False,
            "aspect_model_dir": str(tmp_path / "aspect"),
        }
    }

    fake_model = MagicMock()
    # Logits biased toward class 2 (positive) — deterministic output
    fake_model.return_value.logits = torch.tensor([[0.1, 0.2, 2.5]])
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model

    with patch("src.pipeline.inference_pipeline.load_config", return_value=config), \
         patch(
             "src.pipeline.inference_pipeline.AutoTokenizer.from_pretrained",
             return_value=mock_tokenizer,
         ), \
         patch(
             "src.pipeline.inference_pipeline.AutoModelForSequenceClassification.from_pretrained",
             return_value=fake_model,
         ):
        pipe = InferencePipeline(config_path="configs/pipeline_params.yaml")

    return pipe


# ---------------------------------------------------------------------------
# _build_texts — single mode
# ---------------------------------------------------------------------------

def test_build_texts_single_returns_one_item(pipeline):
    texts = pipeline._build_texts("Great product", "Really loved it", batch_mode=False)
    assert len(texts) == 1


def test_build_texts_single_contains_title(pipeline):
    texts = pipeline._build_texts("My Title", "Some body.", batch_mode=False)
    assert "My Title" in texts[0]


def test_build_texts_single_contains_body(pipeline):
    texts = pipeline._build_texts("My Title", "Review body text.", batch_mode=False)
    assert "body text" in texts[0]


def test_build_texts_single_no_body_does_not_crash(pipeline):
    texts = pipeline._build_texts("Title only", None, batch_mode=False)
    assert len(texts) == 1
    assert "Title only" in texts[0]


# ---------------------------------------------------------------------------
# _build_texts — batch mode
# ---------------------------------------------------------------------------

def test_build_texts_batch_returns_correct_count(pipeline):
    titles = "Title One|||Title Two|||Title Three"
    bodies = "Body One|||Body Two|||Body Three"
    texts = pipeline._build_texts(titles, bodies, batch_mode=True)
    assert len(texts) == 3


def test_build_texts_batch_correct_first_item(pipeline):
    titles = "Title One|||Title Two"
    bodies = "Body One|||Body Two"
    texts = pipeline._build_texts(titles, bodies, batch_mode=True)
    assert "Title One" in texts[0]
    assert "Body One" in texts[0]


def test_build_texts_batch_correct_last_item(pipeline):
    titles = "Title One|||Title Two|||Title Three"
    bodies = "Body One|||Body Two|||Body Three"
    texts = pipeline._build_texts(titles, bodies, batch_mode=True)
    assert "Title Three" in texts[2]


def test_build_texts_batch_no_body_fills_empty_strings(pipeline):
    """Missing body in batch mode — empty strings substituted, no crash."""
    titles = "A|||B"
    texts = pipeline._build_texts(titles, None, batch_mode=True)
    assert len(texts) == 2


# ---------------------------------------------------------------------------
# _minimal_clean
# ---------------------------------------------------------------------------

def test_minimal_clean_removes_http_url(pipeline):
    raw = "Visit http://example.com for details"
    assert "http://" not in pipeline._minimal_clean(raw)


def test_minimal_clean_removes_https_url(pipeline):
    raw = "See https://example.com/path?q=1"
    assert "https://" not in pipeline._minimal_clean(raw)


def test_minimal_clean_collapses_multiple_spaces(pipeline):
    raw = "too  many   spaces  here"
    result = pipeline._minimal_clean(raw)
    assert "  " not in result


def test_minimal_clean_strips_surrounding_whitespace(pipeline):
    raw = "  leading and trailing  "
    result = pipeline._minimal_clean(raw)
    assert result == result.strip()


def test_minimal_clean_preserves_text_content(pipeline):
    raw = "Works perfectly for the price"
    result = pipeline._minimal_clean(raw)
    assert "perfectly" in result


# ---------------------------------------------------------------------------
# run() — DataFrame contract
# ---------------------------------------------------------------------------

def test_run_single_returns_dataframe(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    assert isinstance(df, pd.DataFrame)


def test_run_single_has_one_row(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    assert len(df) == 1


def test_run_single_has_required_columns(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    for col in ("text", "label", "confidence", "scores"):
        assert col in df.columns, f"Missing expected column: {col}"


def test_run_single_label_is_valid_sentiment(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    assert df.iloc[0]["label"] in SENTIMENT_MAP.values()


def test_run_single_confidence_is_in_range(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    confidence = df.iloc[0]["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_run_single_scores_has_all_classes(pipeline):
    df = pipeline.run(title="Great purchase", text="Works perfectly.", batch_mode=False)
    scores = df.iloc[0]["scores"]
    assert set(scores.keys()) == {"negative", "neutral", "positive"}


# ---------------------------------------------------------------------------
# SENTIMENT_MAP regression
# ---------------------------------------------------------------------------

def test_sentiment_map_has_all_three_classes():
    """Regression: all three classes must always be present in SENTIMENT_MAP."""
    assert set(SENTIMENT_MAP.values()) == {"negative", "neutral", "positive"}


def test_sentiment_map_keys_are_ints():
    assert all(isinstance(k, int) for k in SENTIMENT_MAP)
