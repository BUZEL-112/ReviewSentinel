import sqlite3
import json
import os
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timezone
from src.llm_judge import QueueEntry, JudgmentResult
from src.utils.logger import logger

class ConflictLogger:
    def __init__(self, db_path: str = "artifacts/llm_judge/conflicts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    input_text TEXT,
                    raw_title TEXT,
                    raw_text TEXT,
                    model_prediction TEXT,
                    model_confidence REAL,
                    model_probabilities TEXT,
                    model_version TEXT,
                    judge_prediction TEXT,
                    judge_confidence REAL,
                    judge_reasoning TEXT,
                    raw_response TEXT,
                    latency_ms REAL,
                    human_label TEXT,
                    human_reviewed_at TEXT,
                    exported_for_training BOOLEAN
                )
            """)
            conn.commit()

    def log_conflict(self, entry: QueueEntry, result: JudgmentResult):
        if not result.is_conflict or not result.parse_success:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO conflicts (
                    entry_id, timestamp, input_text, raw_title, raw_text,
                    model_prediction, model_confidence, model_probabilities, model_version,
                    judge_prediction, judge_confidence, judge_reasoning,
                    raw_response, latency_ms, human_label, human_reviewed_at, exported_for_training
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id, entry.timestamp, entry.input_text, entry.raw_title, entry.raw_text,
                entry.model_prediction, entry.model_confidence, json.dumps(entry.model_probabilities), entry.model_version,
                result.judge_prediction, result.judge_confidence, result.judge_reasoning,
                result.raw_response, result.latency_ms, None, None, False
            ))
            conn.commit()

    def get_unlabeled_conflicts(self, limit: int = 100) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM conflicts WHERE human_label IS NULL ORDER BY timestamp DESC LIMIT ?", 
                conn, params=(limit,)
            )
            if not df.empty:
                df['model_probabilities'] = df['model_probabilities'].apply(json.loads)
                df['exported_for_training'] = df['exported_for_training'].astype(bool)
            return df

    def get_training_candidates(self, min_confidence_gap: float = 0.1) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            # We want cases where judge was reasonably confident, and it was exported_for_training = False
            df = pd.read_sql_query(
                "SELECT * FROM conflicts WHERE exported_for_training = 0 AND judge_confidence >= ?", 
                conn, params=(min_confidence_gap,)
            )
            if not df.empty:
                df['model_probabilities'] = df['model_probabilities'].apply(json.loads)
                df['exported_for_training'] = df['exported_for_training'].astype(bool)
            return df

    def mark_exported(self, entry_ids: List[str]):
        if not entry_ids:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(entry_ids))
            cursor.execute(f"""
                UPDATE conflicts 
                SET exported_for_training = 1 
                WHERE entry_id IN ({placeholders})
            """, entry_ids)
            conn.commit()

    def get_conflict_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM conflicts")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT model_prediction, judge_prediction, COUNT(*) FROM conflicts GROUP BY model_prediction, judge_prediction")
            patterns = [{"model": row[0], "judge": row[1], "count": row[2]} for row in cursor.fetchall()]
            
            return {
                "total_conflicts": total,
                "patterns": patterns
            }
