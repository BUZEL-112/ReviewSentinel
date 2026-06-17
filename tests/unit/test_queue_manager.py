"""
tests/unit/test_queue_manager.py

Unit tests for QueueManager (src/llm_judge/queue_manager.py).

Key design choices:
- Every test uses QueueManager(db_path=":memory:") via the `qm` fixture defined
  in conftest.py, giving each test a fresh, empty SQLite database with zero
  teardown required.
- Tests follow the happy-path workflow first, then edge cases.
- The `make_queue_entry` helper from conftest is imported to avoid copy-paste.
"""

import pytest
import uuid
from datetime import datetime, timezone

from tests.conftest import make_queue_entry


# ---------------------------------------------------------------------------
# Enqueue / basic persistence
# ---------------------------------------------------------------------------

def test_enqueue_returns_entry_id(qm):
    entry = make_queue_entry()
    returned_id = qm.enqueue(entry)
    assert returned_id == entry.entry_id


def test_enqueue_increments_pending_count(qm):
    assert qm.get_pending_count() == 0
    qm.enqueue(make_queue_entry())
    assert qm.get_pending_count() == 1
    qm.enqueue(make_queue_entry())
    assert qm.get_pending_count() == 2


def test_duplicate_enqueue_is_idempotent(qm):
    """Same entry_id twice — INSERT OR IGNORE guarantees only one row."""
    entry = make_queue_entry()
    qm.enqueue(entry)
    qm.enqueue(entry)
    assert qm.get_pending_count() == 1


# ---------------------------------------------------------------------------
# Dequeue
# ---------------------------------------------------------------------------

def test_dequeue_batch_returns_entry(qm):
    entry = make_queue_entry()
    qm.enqueue(entry)
    batch = qm.dequeue_batch(batch_size=10)
    assert len(batch) == 1
    assert batch[0].entry_id == entry.entry_id


def test_dequeue_batch_marks_as_processing(qm):
    """After dequeue the entry is no longer visible as pending."""
    qm.enqueue(make_queue_entry())
    qm.dequeue_batch(batch_size=10)
    assert qm.get_pending_count() == 0


def test_dequeue_batch_respects_batch_size(qm):
    for _ in range(5):
        qm.enqueue(make_queue_entry())

    batch = qm.dequeue_batch(batch_size=3)
    assert len(batch) == 3
    # 2 entries were not dequeued — they remain pending
    assert qm.get_pending_count() == 2


def test_dequeue_empty_queue_returns_empty_list(qm):
    batch = qm.dequeue_batch(batch_size=10)
    assert batch == []


def test_dequeue_sets_status_to_processing(qm):
    qm.enqueue(make_queue_entry())
    batch = qm.dequeue_batch(batch_size=1)
    assert batch[0].status == "processing"


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------

def test_update_status_to_conflict(qm):
    entry = make_queue_entry()
    qm.enqueue(entry)
    qm.dequeue_batch(batch_size=1)
    qm.update_status(
        entry.entry_id,
        status="conflict",
        judge_prediction="negative",
        judge_reasoning="Clearly negative.",
        is_conflict=True,
    )
    rate = qm.get_conflict_rate(window_hours=9999)
    assert rate == 1.0


def test_update_status_to_agreement(qm):
    entry = make_queue_entry(prediction="positive")
    qm.enqueue(entry)
    qm.dequeue_batch(batch_size=1)
    qm.update_status(
        entry.entry_id,
        status="agreement",
        judge_prediction="positive",
        judge_reasoning="Confirmed positive.",
        is_conflict=False,
    )
    rate = qm.get_conflict_rate(window_hours=9999)
    assert rate == 0.0


# ---------------------------------------------------------------------------
# get_conflict_rate
# ---------------------------------------------------------------------------

def test_conflict_rate_zero_when_no_judged_entries(qm):
    """No division-by-zero when the table is empty."""
    assert qm.get_conflict_rate(window_hours=168) == 0.0


def test_conflict_rate_mixed_outcomes(qm):
    """2 conflicts out of 3 judged -> 0.666..."""
    for i in range(3):
        e = make_queue_entry(entry_id=str(i))
        qm.enqueue(e)
        qm.dequeue_batch(batch_size=1)
        status = "conflict" if i < 2 else "agreement"
        qm.update_status(e.entry_id, status=status, is_conflict=(i < 2))
    rate = qm.get_conflict_rate(window_hours=9999)
    assert abs(rate - (2 / 3)) < 0.001


# ---------------------------------------------------------------------------
# export_conflicts
# ---------------------------------------------------------------------------

def test_export_conflicts_returns_only_conflicts(qm):
    # One conflict, one agreement
    conflict_entry = make_queue_entry(entry_id="conflict-1")
    agree_entry = make_queue_entry(entry_id="agree-1")

    for e in [conflict_entry, agree_entry]:
        qm.enqueue(e)
        qm.dequeue_batch(batch_size=1)

    qm.update_status("conflict-1", status="conflict", is_conflict=True)
    qm.update_status("agree-1", status="agreement", is_conflict=False)

    df = qm.export_conflicts()
    assert len(df) == 1
    assert df.iloc[0]["entry_id"] == "conflict-1"


def test_export_conflicts_empty_when_no_conflicts(qm):
    import pandas as pd
    df = qm.export_conflicts()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ---------------------------------------------------------------------------
# cleanup_agreements
# ---------------------------------------------------------------------------

def test_cleanup_agreements_removes_old_agreements(qm):
    entry = make_queue_entry()
    qm.enqueue(entry)
    qm.dequeue_batch(batch_size=1)
    qm.update_status(entry.entry_id, status="agreement")

    # retention_days=0 treats everything as old enough to delete
    qm.cleanup_agreements(retention_days=0)
    assert qm.get_pending_count() == 0
    # The export should also be empty after cleanup
    df = qm.export_conflicts()
    assert df.empty


def test_cleanup_agreements_does_not_remove_conflicts(qm):
    entry = make_queue_entry()
    qm.enqueue(entry)
    qm.dequeue_batch(batch_size=1)
    qm.update_status(entry.entry_id, status="conflict", is_conflict=True)

    qm.cleanup_agreements(retention_days=0)

    # The conflict row should still be there
    df = qm.export_conflicts()
    assert len(df) == 1
