import sys
import uuid
import json
import sqlite3
from datetime import datetime, timezone
from src.llm_judge.queue_manager import QueueManager
from src.llm_judge import QueueEntry
from src.orchestration.flows import judge_processing_flow
from src.utils.logger import logger

def run_comprehensive_test():
    db_path = "artifacts/llm_judge/review_queue.db"
    qm = QueueManager(db_path=db_path)
    
    logger.info("--- STEP 1: Injecting Dummy Data ---")
    # We deliberately create an entry with 0.51 confidence
    dummy_entry = QueueEntry(
        entry_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_text="The battery life is amazing but the screen broke in two days.",
        raw_title="Mixed feelings",
        raw_text="The battery life is amazing but the screen broke in two days.",
        model_prediction="positive", # Original model's guess
        model_confidence=0.51,       # Forces it to be "uncertain"
        model_probabilities={"negative": 0.49, "neutral": 0.0, "positive": 0.51},
        model_version="test-script-v1",
        status="pending",
        judge_prediction=None,
        judge_reasoning=None,
        is_conflict=None
    )
    
    # Enqueue the dummy data
    qm.enqueue(dummy_entry)
    pending_count = qm.get_pending_count()
    logger.info(f"Pending entries in queue: {pending_count}")
    
    if pending_count == 0:
        logger.error("Failed to enqueue item!")
        return

    logger.info("--- STEP 2: Triggering LLM Judge Flow ---")
    # This will pull the pending item and send it to Ollama
    try:
        judge_processing_flow()
    except Exception as e:
        logger.error(f"Flow failed: {e}")
        return

    logger.info("--- STEP 3: Verifying Results ---")
    # Check the database to see what the Judge decided
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_id, status, model_prediction, judge_prediction, judge_reasoning 
            FROM review_queue 
            WHERE entry_id = ?
        """, (dummy_entry.entry_id,))
        
        result = cursor.fetchone()
        
        if result:
            print("\n" + "="*50)
            print("📝 LLM JUDGE RESULT:")
            print("="*50)
            print(f"Status:           {result[1]}")
            print(f"Original Model:   {result[2]}")
            print(f"LLM Judge:        {result[3]}")
            print(f"Judge Reasoning:\n{result[4]}")
            print("="*50 + "\n")
        else:
            logger.error("Could not find the processed entry in the database.")

if __name__ == "__main__":
    run_comprehensive_test()