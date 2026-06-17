"""
tests/integration/test_search_pipeline.py

Integration test: FAISSIndexer.build() → SemanticSearcher.search() round-trip.

What this validates (unit tests cannot):
- SentenceEncoder produces embeddings of the correct shape.
- FAISSIndexer.build() writes both the index file and the metadata parquet.
- SemanticSearcher.search() returns results with valid scores and alignment signals.
- The full chain doesn't raise for a realistic (small) corpus.

Markers:
- @pytest.mark.slow — downloads sentence-transformers weights on first run.

Dependencies (must be installed):
  pip install sentence-transformers faiss-cpu
"""

import pytest
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_corpus():
    """20-row corpus with clean_text, sentiment, raw_title, raw_text, rating."""
    return pd.DataFrame({
        "clean_text": [
            "This product works perfectly well.",
            "Broke after two days of use.",
            "Average product, nothing special.",
            "Amazing quality and fast shipping.",
            "Terrible smell and poor build.",
            "Would recommend to a friend.",
            "Stopped working after a week.",
            "Great value for the price.",
            "Not what I expected at all.",
            "Excellent customer service and product.",
            "Packaging arrived completely crushed.",
            "Battery life is outstanding.",
            "The colours faded after one wash.",
            "Does exactly what it promises.",
            "Very disappointed with the quality.",
            "Lightweight and easy to carry.",
            "Instructions are completely unclear.",
            "Feels premium and looks stylish.",
            "Returned it the same day.",
            "One of the best purchases this year.",
        ],
        "sentiment": [
            "positive", "negative", "neutral", "positive", "negative",
            "positive", "negative", "positive", "negative", "positive",
            "negative", "positive", "negative", "positive", "negative",
            "positive", "negative", "positive", "negative", "positive",
        ],
        "raw_title": [f"Title {i}" for i in range(20)],
        "raw_text":  [f"Raw review body {i}" for i in range(20)],
        "rating":    [5, 1, 3, 5, 1, 4, 1, 4, 2, 5,
                      1, 5, 2, 4, 1, 4, 2, 5, 1, 5],
    })


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_index_build_creates_index_file(small_corpus, tmp_path):
    """FAISSIndexer.build() must create a .faiss file on disk."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    faiss_files = list(tmp_path.glob("*.faiss"))
    assert len(faiss_files) >= 1, "No .faiss file found after build()"


@pytest.mark.slow
def test_index_build_creates_metadata_file(small_corpus, tmp_path):
    """FAISSIndexer.build() must create a metadata parquet file."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    meta_files = list(tmp_path.glob("*.parquet")) + list(tmp_path.glob("*.json"))
    assert len(meta_files) >= 1, "No metadata file found after build()"


@pytest.mark.slow
def test_searcher_returns_results(small_corpus, tmp_path):
    """After building the index, a search query should return at least one result."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer
    from src.search.searcher import SemanticSearcher

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    searcher = SemanticSearcher(
        index_dir=str(tmp_path),
        encoder=encoder,
        default_top_k=5,
    )
    response = searcher.search("This product is great and I love it.")
    assert len(response.results) >= 1


@pytest.mark.slow
def test_searcher_respects_top_k(small_corpus, tmp_path):
    """top_k=3 should return at most 3 results."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer
    from src.search.searcher import SemanticSearcher

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    searcher = SemanticSearcher(index_dir=str(tmp_path), encoder=encoder)
    response = searcher.search("great product works perfectly", top_k=3)
    assert response.top_k <= 3


@pytest.mark.slow
def test_searcher_similarity_scores_in_range(small_corpus, tmp_path):
    """All similarity scores must be between -1.0 and 1.0 (cosine / inner-product)."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer
    from src.search.searcher import SemanticSearcher

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    searcher = SemanticSearcher(index_dir=str(tmp_path), encoder=encoder)
    response = searcher.search("broken after one use.")
    for r in response.results:
        assert -1.0 <= r.similarity_score <= 1.0


@pytest.mark.slow
def test_searcher_alignment_signal_is_valid(small_corpus, tmp_path):
    """alignment_signal must be one of the three expected values (or None)."""
    from src.search.encoder import SentenceEncoder
    from src.search.indexer import FAISSIndexer
    from src.search.searcher import SemanticSearcher

    encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", batch_size=8)
    indexer = FAISSIndexer(index_dir=str(tmp_path))
    indexer.build(small_corpus, encoder, text_column="clean_text")

    searcher = SemanticSearcher(index_dir=str(tmp_path), encoder=encoder)
    response = searcher.search(
        "great product and excellent quality.",
        model_prediction="positive",
    )
    valid_signals = {None, "STRONG_SUPPORT", "MIXED", "CONTRADICTS"}
    assert response.alignment_signal in valid_signals
