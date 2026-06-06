import os
import yaml
from enum import Enum
import pandas as pd
from prefect import task
from prefect.deployments import run_deployment
from prefect.blocks.notifications import SlackWebhook  # Or other preferred block

from src.monitoring.reference_store import ReferenceStore
from src.monitoring.drift_monitor import DriftMonitor, DriftResult
from src.utils.logger import logger

class DriftAction(Enum):
    NONE = "NONE"
    ALERT = "ALERT"
    TRIGGER_RETRAINING = "TRIGGER_RETRAINING"

@task(name="Load Prediction Log", retries=2, retry_delay_seconds=[10, 30], tags=["monitoring", "io"])
def load_prediction_log_task(config_path: str = "configs/pipeline_params.yaml", days: int = 7) -> pd.DataFrame:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f).get("monitoring", {})
    
    # We instantiate DriftMonitor just to use its loading logic, without needing a full reference
    temp_monitor = DriftMonitor(reference_store=None, config=config)
    df = temp_monitor._load_current_data()
    return df

@task(name="Generate Drift Report", retries=1, retry_delay_seconds=[30], tags=["monitoring", "drift"])
def generate_drift_report_task(current_df: pd.DataFrame, config_path: str = "configs/pipeline_params.yaml") -> DriftResult:
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
        config = full_config.get("monitoring", {})
        mlflow_uri = full_config.get("mlflow", {}).get("tracking_uri")
        
    ref_store = ReferenceStore(mlflow_tracking_uri=mlflow_uri)
    monitor = DriftMonitor(reference_store=ref_store, config=config, mlflow_tracking_uri=mlflow_uri)
    
    result = monitor.generate_report(current_source=current_df)
    return result

@task(name="Evaluate Drift", retries=0, tags=["monitoring", "evaluation"])
def evaluate_drift_task(drift_result: DriftResult, config_path: str = "configs/pipeline_params.yaml") -> DriftAction:
    if drift_result is None:
        logger.warning("No drift result provided. Action: NONE")
        return DriftAction.NONE
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f).get("monitoring", {}).get("drift", {})
        
    alert_thresh = config.get("alert_threshold", 0.30)
    retrain_thresh = config.get("retrain_threshold", 0.50)
    
    fraction = drift_result.drift_fraction
    logger.info(f"Evaluating drift. Fraction: {fraction:.2f} | Alert Thresh: {alert_thresh} | Retrain Thresh: {retrain_thresh}")
    
    if fraction >= retrain_thresh:
        return DriftAction.TRIGGER_RETRAINING
    elif fraction >= alert_thresh:
        return DriftAction.ALERT
    
    return DriftAction.NONE

@task(name="Send Drift Alert", retries=2, retry_delay_seconds=[10, 30], tags=["monitoring", "alert"])
def send_drift_alert_task(drift_result: DriftResult, action: DriftAction):
    if drift_result is None:
        return
        
    msg = (
        f"🚨 *Model Drift Alert* 🚨\n"
        f"Action Recommended: {action.value}\n\n"
        f"Details:\n"
        f"- {drift_result.summary}\n"
        f"- Run ID: `{drift_result.mlflow_run_id}`\n"
        f"- Report Path: `{drift_result.report_path}`"
    )
    
    logger.warning(msg)
    
    # In production, use a Prefect Notification Block:
    # try:
    #     slack_block = SlackWebhook.load("drift-alerts")
    #     slack_block.notify(msg)
    # except Exception as e:
    #     logger.warning(f"Could not send Slack notification: {e}")

@task(name="Trigger Retraining", retries=3, retry_delay_seconds=[60, 120, 300], tags=["monitoring", "retraining"])
def trigger_retraining_task(config_path: str = "configs/pipeline_params.yaml"):
    logger.info("Exporting un-exported LLM Judge conflicts before retraining...")
    try:
        with open(config_path, "r") as f:
            judge_cfg = yaml.safe_load(f).get("llm_judge", {})
            c_db_path = judge_cfg.get("conflicts", {}).get("db_path", "artifacts/llm_judge/conflicts.db")
            min_conf = judge_cfg.get("conflicts", {}).get("min_judge_confidence", 0.60)
            
        from src.llm_judge.conflict_logger import ConflictLogger
        import os
        if os.path.exists(c_db_path):
            cl = ConflictLogger(db_path=c_db_path)
            conflicts_df = cl.get_training_candidates(min_confidence_gap=min_conf)
            if not conflicts_df.empty:
                export_path = "artifacts/llm_judge/exported_conflicts_for_training.csv"
                # Select important columns and rename label appropriately
                export_df = conflicts_df[['input_text', 'judge_prediction']].rename(
                    columns={'input_text': 'text', 'judge_prediction': 'sentiment'}
                )
                export_df.to_csv(export_path, index=False)
                cl.mark_exported(conflicts_df['entry_id'].tolist())
                logger.info(f"Exported {len(conflicts_df)} conflicts for retraining.")
            else:
                logger.info("No new conflicts to export.")
    except Exception as e:
        logger.error(f"Failed to export conflicts: {e}")
        
    logger.info("Triggering retraining flow deployment...")
    try:
        run = run_deployment(name="ReviewSentinel Training Pipeline/weekly-training", timeout=0)
        logger.info(f"Triggered retraining. Flow run ID: {run.id}")
    except Exception as e:
        logger.error(f"Failed to trigger retraining: {e}")
        raise
