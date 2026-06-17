"""
tests/unit/test_conflict_logger.py

Unit tests for ConflictLogger (src/llm_judge/conflict_logger.py).

Uses the `conflict_logger` fixture (SQLite :memory:) from conftest.py.
Tests cover: log_conflict, get_unlabeled_conflicts, get_training_candidates,
mark_exported, and get_conflict_stats.
"""

import pytest
from datetime import datetime, timezone
from src.llm_judge import QueueEntry, JudgmentResult
from tests.conftest import make_queue_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_judgment(
    entry_id: str,
    judge_prediction: str = "negative",
    judge_confidence: float = 0.88,
    is_conflict: bool = True,
    parse_success: bool = True,
) -> JudgmentResult:
    return JudgmentResult(
        entry_id=entry_id,
        judge_prediction=judge_prediction,
        judge_confidence=judge_confidence,
        judge_reasoning="Test reasoning.",
        is_conflict=is_conflict,
        raw_response='{"sentiment": "negative", "confidence": 0.88, "reasoning": "Bad."}',
        parse_success=parse_success,
        latency_ms=123.4,
    )


# ---------------------------------------------------------------------------
# log_conflict
# ---------------------------------------------------------------------------

def test_log_conflict_stores_entry(conflict_logger):
    entry = make_queue_entry(entry_id="e1", prediction="positive")
    result = make_judgment(entry_id="e1", judge_prediction="negative", is_conflict=True)
    conflict_logger.log_conflict(entry, result)

    stats = conflict_logger.get_conflict_stats()
    assert stats["total_conflicts"] == 1


def test_log_conflict_ignores_agreements(conflict_logger):
    """Entries where is_conflict=False must NOT be stored."""
    entry = make_queue_entry(entry_id="e2", prediction="positive")
    result = make_judgment(entry_id="e2", judge_prediction="positive", is_conflict=False)
    conflict_logger.log_conflict(entry, result)

    stats = conflict_logger.get_conflict_stats()
    assert stats["total_conflicts"] == 0


def test_log_conflict_ignores_failed_parses(conflict_logger):
    """Entries where parse_success=False must NOT be stored."""
    entry = make_queue_entry(entry_id="e3")
    result = make_judgment(entry_id="e3", is_conflict=True, parse_success=False)
    conflict_logger.log_conflict(entry, result)

    stats = conflict_logger.get_conflict_stats()
    assert stats["total_conflicts"] == 0


def test_log_conflict_is_idempotent(conflict_logger):
    """Same entry_id twice — INSERT OR IGNORE keeps only one row."""
    entry = make_queue_entry(entry_id="e4")
    result = make_judgment(entry_id="e4")
    conflict_logger.log_conflict(entry, result)
    conflict_logger.log_conflict(entry, result)

    stats = conflict_logger.get_conflict_stats()
    assert stats["total_conflicts"] == 1


# ---------------------------------------------------------------------------
# get_unlabeled_conflicts
# ---------------------------------------------------------------------------

def test_get_unlabeled_conflicts_returns_all_fresh_entries(conflict_logger):
    for i in range(3):
        e = make_queue_entry(entry_id=f"ul-{i}")
        r = make_judgment(entry_id=f"ul-{i}")
        conflict_logger.log_conflict(e, r)

    df = conflict_logger.get_unlabeled_conflicts(limit=10)
    assert len(df) == 3


def test_get_unlabeled_conflicts_respects_limit(conflict_logger):
    for i in range(5):
        e = make_queue_entry(entry_id=f"lim-{i}")
        r = make_judgment(entry_id=f"lim-{i}")
        conflict_logger.log_conflict(e, r)

    df = conflict_logger.get_unlabeled_conflicts(limit=3)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# get_training_candidates
# ---------------------------------------------------------------------------

def test_get_training_candidates_filters_by_confidence(conflict_logger):
    """Only entries with judge_confidence >= threshold are returned."""
    high_conf = make_queue_entry(entry_id="tc-high")
    low_conf = make_queue_entry(entry_id="tc-low")

    conflict_logger.log_conflict(high_conf, make_judgment(entry_id="tc-high", judge_confidence=0.90))
    conflict_logger.log_conflict(low_conf, make_judgment(entry_id="tc-low", judge_confidence=0.30))

    candidates = conflict_logger.get_training_candidates(min_confidence_gap=0.60)
    assert len(candidates) == 1
    assert candidates.iloc[0]["entry_id"] == "tc-high"


# ---------------------------------------------------------------------------
# mark_exported
# ---------------------------------------------------------------------------

def test_mark_exported_updates_flag(conflict_logger):
    entry = make_queue_entry(entry_id="exp-1")
    result = make_judgment(entry_id="exp-1")
    conflict_logger.log_conflict(entry, result)

    conflict_logger.mark_exported(["exp-1"])

    # After marking, this entry should NOT appear in training candidates
    candidates = conflict_logger.get_training_candidates(min_confidence_gap=0.0)
    assert candidates.empty


def test_mark_exported_empty_list_is_safe(conflict_logger):
    """Passing an empty list must not raise."""
    conflict_logger.mark_exported([])   # should not throw


# ---------------------------------------------------------------------------
# get_conflict_stats
# ---------------------------------------------------------------------------

def test_get_conflict_stats_total_count(conflict_logger):
    for i in range(4):
        e = make_queue_entry(entry_id=f"stat-{i}")
        r = make_judgment(entry_id=f"stat-{i}")
        conflict_logger.log_conflict(e, r)

    stats = conflict_logger.get_conflict_stats()
    assert stats["total_conflicts"] == 4


def test_get_conflict_stats_patterns_structure(conflict_logger):
    entry = make_queue_entry(entry_id="pattern-1", prediction="positive")
    result = make_judgment(entry_id="pattern-1", judge_prediction="negative")
    conflict_logger.log_conflict(entry, result)

    stats = conflict_logger.get_conflict_stats()
    assert "patterns" in stats
    assert len(stats["patterns"]) >= 1
    pattern = stats["patterns"][0]
    assert "model" in pattern
    assert "judge" in pattern
    assert "count" in pattern
