import sqlite3
import json
import os
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
from src.llm_judge import QueueEntry
from src.utils.logger import logger

class QueueManager:
    def __init__(self, db_path: str = "artifacts/llm_judge/review_queue.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    input_text TEXT,
                    raw_title TEXT,
                    raw_text TEXT,
                    model_prediction TEXT,
                    model_confidence REAL,
                    model_probabilities TEXT,
                    model_version TEXT,
                    status TEXT,
                    judge_prediction TEXT,
                    judge_reasoning TEXT,
                    is_conflict BOOLEAN
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status_timestamp ON review_queue (status, timestamp)")
            conn.commit()

    def enqueue(self, entry: QueueEntry) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO review_queue (
                    entry_id, timestamp, input_text, raw_title, raw_text,
                    model_prediction, model_confidence, model_probabilities,
                    model_version, status, judge_prediction, judge_reasoning, is_conflict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id, entry.timestamp, entry.input_text, entry.raw_title, entry.raw_text,
                entry.model_prediction, entry.model_confidence, json.dumps(entry.model_probabilities),
                entry.model_version, entry.status, entry.judge_prediction, entry.judge_reasoning,
                entry.is_conflict
            ))
            conn.commit()
        return entry.entry_id

    def dequeue_batch(self, batch_size: int, max_age_minutes: Optional[int] = None) -> List[QueueEntry]:
        entries = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM review_queue WHERE status = 'pending'"
            params = []
            
            if max_age_minutes is not None:
                cutoff_time = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
                query += " AND timestamp < ?"
                params.append(cutoff_time)
                
            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(batch_size)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return entries
                
            entry_ids = [row[0] for row in rows]
            placeholders = ",".join(["?"] * len(entry_ids))
            
            # Atomically update to processing
            cursor.execute(f"UPDATE review_queue SET status = 'processing' WHERE entry_id IN ({placeholders})", entry_ids)
            conn.commit()
            
            for row in rows:
                entries.append(QueueEntry(
                    entry_id=row[0], timestamp=row[1], input_text=row[2], raw_title=row[3], raw_text=row[4],
                    model_prediction=row[5], model_confidence=row[6], model_probabilities=json.loads(row[7]),
                    model_version=row[8], status="processing", judge_prediction=row[10], judge_reasoning=row[11],
                    is_conflict=bool(row[12]) if row[12] is not None else None
                ))
                
        return entries

    def update_status(self, entry_id: str, status: str, judge_prediction: Optional[str] = None, 
                      judge_reasoning: Optional[str] = None, is_conflict: Optional[bool] = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE review_queue 
                SET status = ?, judge_prediction = ?, judge_reasoning = ?, is_conflict = ?
                WHERE entry_id = ?
            """, (status, judge_prediction, judge_reasoning, is_conflict, entry_id))
            conn.commit()

    def get_pending_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'")
            return cursor.fetchone()[0]

    def get_conflict_rate(self, window_hours: int = 168) -> float:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
            
            cursor.execute("""
                SELECT COUNT(*) FROM review_queue 
                WHERE status IN ('conflict', 'agreement') AND timestamp >= ?
            """, (cutoff,))
            total_judged = cursor.fetchone()[0]
            
            if total_judged == 0:
                return 0.0
                
            cursor.execute("""
                SELECT COUNT(*) FROM review_queue 
                WHERE status = 'conflict' AND timestamp >= ?
            """, (cutoff,))
            total_conflicts = cursor.fetchone()[0]
            
            return total_conflicts / total_judged

    def export_conflicts(self, since: Optional[str] = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM review_queue WHERE status = 'conflict'"
            params = []
            if since:
                query += " AND timestamp >= ?"
                params.append(since)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['model_probabilities'] = df['model_probabilities'].apply(json.loads)
                df['is_conflict'] = df['is_conflict'].astype(bool)
            return df
            
    def cleanup_agreements(self, retention_days: int = 7):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
            cursor.execute("DELETE FROM review_queue WHERE status = 'agreement' AND timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old agreement entries from queue")
