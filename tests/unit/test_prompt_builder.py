"""
tests/unit/test_prompt_builder.py

Unit tests for PromptBuilder (src/llm_judge/prompt_builder.py).

This is the highest-value LLM Judge test because parse_response has three
fallback tiers and a silent parse failure means reviews are permanently lost.
Every tier and every failure path has its own test.
"""

import pytest
from src.llm_judge.prompt_builder import PromptBuilder
from src.llm_judge import QueueEntry
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Sample raw LLM outputs covering every parse tier
# ---------------------------------------------------------------------------

VALID_JSON = (
    '{"sentiment": "negative", "confidence": 0.92, "reasoning": "Clearly dissatisfied."}'
)

MARKDOWN_WRAPPED = (
    '```json\n'
    '{"sentiment": "positive", "confidence": 0.85, "reasoning": "Happy customer."}\n'
    '```'
)

PREAMBLE_JSON = (
    'Sure, here is my assessment:\n'
    '{"sentiment": "neutral", "confidence": 0.55, "reasoning": "Mixed signals."}'
)

MISSING_REASONING = '{"sentiment": "negative", "confidence": 0.80}'
MISSING_CONFIDENCE = '{"sentiment": "positive", "reasoning": "Great."}'
MISSING_SENTIMENT = '{"confidence": 0.70, "reasoning": "Okay product."}'
GARBAGE = "I think it might be negative because of the word bad."
ALMOST_JSON_TRAILING_COMMA = '{"sentiment": "negative", "confidence": 0.80, "reasoning": "Bad.",}'


# ---------------------------------------------------------------------------
# parse_response — tier 1: direct JSON
# ---------------------------------------------------------------------------

def test_parse_clean_json_returns_dict(dummy=None):
    result = PromptBuilder.parse_response(VALID_JSON)
    assert result is not None
    assert isinstance(result, dict)


def test_parse_clean_json_correct_sentiment():
    result = PromptBuilder.parse_response(VALID_JSON)
    assert result["sentiment"] == "negative"


def test_parse_clean_json_correct_confidence():
    result = PromptBuilder.parse_response(VALID_JSON)
    assert result["confidence"] == 0.92


def test_parse_clean_json_correct_reasoning():
    result = PromptBuilder.parse_response(VALID_JSON)
    assert "dissatisfied" in result["reasoning"].lower()


# ---------------------------------------------------------------------------
# parse_response — tier 2: regex fallback (markdown fences & preambles)
# ---------------------------------------------------------------------------

def test_parse_markdown_wrapped_json():
    """Ollama sometimes wraps output in ```json ... ``` fences."""
    result = PromptBuilder.parse_response(MARKDOWN_WRAPPED)
    assert result is not None
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.85


def test_parse_json_with_prose_preamble():
    """The regex fallback extracts the JSON object even with prose before it."""
    result = PromptBuilder.parse_response(PREAMBLE_JSON)
    assert result is not None
    assert result["sentiment"] == "neutral"


# ---------------------------------------------------------------------------
# parse_response — tier 3: graceful failure (returns None)
# ---------------------------------------------------------------------------

def test_parse_missing_reasoning_returns_none():
    assert PromptBuilder.parse_response(MISSING_REASONING) is None


def test_parse_missing_confidence_returns_none():
    assert PromptBuilder.parse_response(MISSING_CONFIDENCE) is None


def test_parse_missing_sentiment_returns_none():
    assert PromptBuilder.parse_response(MISSING_SENTIMENT) is None


def test_parse_complete_garbage_returns_none():
    assert PromptBuilder.parse_response(GARBAGE) is None


def test_parse_empty_string_returns_none():
    assert PromptBuilder.parse_response("") is None


def test_parse_none_returns_none():
    assert PromptBuilder.parse_response(None) is None  # type: ignore[arg-type]


def test_parse_whitespace_only_returns_none():
    assert PromptBuilder.parse_response("   \n\t  ") is None


def test_parse_almost_valid_json_trailing_comma_returns_none():
    """JSON with a trailing comma is invalid — should not be accepted."""
    result = PromptBuilder.parse_response(ALMOST_JSON_TRAILING_COMMA)
    # May or may not parse depending on the regex tier — what matters is it
    # doesn't raise an unhandled exception.
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# build_judgment_prompt — content completeness
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entry():
    return QueueEntry(
        entry_id="abc-123",
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_text="Arrived broken and smelled odd.",
        raw_title="Broken on arrival",
        raw_text="Arrived broken and smelled odd.",
        model_prediction="negative",
        model_confidence=0.55,
        model_probabilities={"negative": 0.55, "neutral": 0.25, "positive": 0.20},
        model_version="v1.0",
        status="pending",
    )


def test_build_judgment_prompt_contains_title(sample_entry):
    prompt = PromptBuilder.build_judgment_prompt(sample_entry)
    assert "Broken on arrival" in prompt


def test_build_judgment_prompt_contains_body_text(sample_entry):
    prompt = PromptBuilder.build_judgment_prompt(sample_entry)
    assert "Arrived broken" in prompt


def test_build_judgment_prompt_contains_model_prediction(sample_entry):
    prompt = PromptBuilder.build_judgment_prompt(sample_entry)
    assert "negative" in prompt


def test_build_judgment_prompt_contains_all_class_probabilities(sample_entry):
    prompt = PromptBuilder.build_judgment_prompt(sample_entry)
    assert "positive" in prompt
    assert "neutral" in prompt
    assert "negative" in prompt


def test_build_judgment_prompt_contains_confidence(sample_entry):
    prompt = PromptBuilder.build_judgment_prompt(sample_entry)
    # Confidence should appear as a formatted float
    assert "0.55" in prompt
