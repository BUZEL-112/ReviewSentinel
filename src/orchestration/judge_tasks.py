import yaml
from typing import List, Dict, Any
import mlflow
from prefect import task
from prefect.exceptions import Abort
from src.llm_judge import QueueEntry, JudgmentResult
from src.llm_judge.judge import LLMJudge
from src.llm_judge.queue_manager import QueueManager
from src.llm_judge.conflict_logger import ConflictLogger
from src.utils.logger import logger

def _get_config(config_path: str = "configs/pipeline_params.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@task(name="Check Ollama Health", tags=["judge", "health"])
def check_ollama_health_task(config_path: str = "configs/pipeline_params.yaml"):
    config = _get_config(config_path).get("llm_judge", {})
    judge = LLMJudge(config=config)
    if not judge._check_ollama_health():
        logger.error("Ollama health check failed. Aborting flow.")
        raise Abort("Ollama is not healthy or model is not loaded.")

@task(name="Dequeue Pending Reviews", retries=1, tags=["judge", "queue"])
def dequeue_pending_task(batch_size: int = 50, config_path: str = "configs/pipeline_params.yaml") -> List[QueueEntry]:
    config = _get_config(config_path).get("llm_judge", {})
    q_db_path = config.get("queue", {}).get("db_path", "artifacts/llm_judge/review_queue.db")
    max_age = config.get("queue", {}).get("max_age_minutes", 10)
    
    qm = QueueManager(db_path=q_db_path)
    if qm.get_pending_count() == 0:
        logger.info("Queue empty, nothing to process.")
        return []
        
    entries = qm.dequeue_batch(batch_size=batch_size, max_age_minutes=max_age)
    logger.info(f"Dequeued {len(entries)} pending entries for judgment.")
    return entries

@task(name="Run LLM Judge Batch", retries=1, retry_delay_seconds=[60], tags=["judge", "inference"])
def run_judge_batch_task(entries: List[QueueEntry], config_path: str = "configs/pipeline_params.yaml") -> List[JudgmentResult]:
    if not entries:
        return []
        
    config = _get_config(config_path).get("llm_judge", {})
    judge = LLMJudge(config=config)
    
    logger.info(f"Running LLM Judge on {len(entries)} entries...")
    results = judge.judge_batch(entries)
    return results

@task(name="Process Judge Results", tags=["judge", "db"])
def process_results_task(entries: List[QueueEntry], results: List[JudgmentResult], config_path: str = "configs/pipeline_params.yaml") -> Dict[str, Any]:
    if not entries or not results:
        return {"processed": 0, "conflicts": 0, "parse_failures": 0, "avg_latency_ms": 0.0}
        
    config = _get_config(config_path).get("llm_judge", {})
    q_db_path = config.get("queue", {}).get("db_path", "artifacts/llm_judge/review_queue.db")
    c_db_path = config.get("conflicts", {}).get("db_path", "artifacts/llm_judge/conflicts.db")
    
    qm = QueueManager(db_path=q_db_path)
    cl = ConflictLogger(db_path=c_db_path)
    
    conflicts = 0
    parse_failures = 0
    total_latency = 0.0
    
    for entry, result in zip(entries, results):
        total_latency += result.latency_ms
        
        if not result.parse_success:
            qm.update_status(entry.entry_id, "judge_failed")
            parse_failures += 1
            continue
            
        status = "conflict" if result.is_conflict else "agreement"
        qm.update_status(
            entry.entry_id, 
            status, 
            judge_prediction=result.judge_prediction, 
            judge_reasoning=result.judge_reasoning,
            is_conflict=result.is_conflict
        )
        
        if result.is_conflict:
            cl.log_conflict(entry, result)
            conflicts += 1
            
    summary = {
        "processed": len(entries),
        "conflicts": conflicts,
        "parse_failures": parse_failures,
        "avg_latency_ms": total_latency / len(entries)
    }
    
    logger.info(f"Processed results: {summary}")
    return summary

@task(name="Log Judge Metrics", tags=["judge", "metrics"])
def log_judge_metrics_task(summary: Dict[str, Any], config_path: str = "configs/pipeline_params.yaml"):
    if summary["processed"] == 0:
        return
        
    full_config = _get_config(config_path)
    mlflow_uri = full_config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("llm_judge")
    
    with mlflow.start_run(run_name="batch_processing"):
        mlflow.log_metrics({
            "entries_processed": summary["processed"],
            "conflicts_found": summary["conflicts"],
            "conflict_rate": summary["conflicts"] / summary["processed"] if summary["processed"] > 0 else 0,
            "parse_failures": summary["parse_failures"],
            "parse_failure_rate": summary["parse_failures"] / summary["processed"] if summary["processed"] > 0 else 0,
            "avg_latency_ms": summary["avg_latency_ms"]
        })
        
    # Also cleanup old agreement queue entries
    judge_cfg = full_config.get("llm_judge", {})
    q_db_path = judge_cfg.get("queue", {}).get("db_path", "artifacts/llm_judge/review_queue.db")
    retention_days = judge_cfg.get("queue", {}).get("retention_days", 7)
    
    qm = QueueManager(db_path=q_db_path)
    qm.cleanup_agreements(retention_days=retention_days)
