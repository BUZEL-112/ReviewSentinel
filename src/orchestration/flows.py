import os
import yaml
import mlflow
import pandas as pd
import datetime
from prefect import task, flow
from prefect.exceptions import Abort

from src.data.load_data import LoadData
from src.data.clean_data import CleanDataBERT
from src.models.train_model import ModelTrainer
from src.models.evaluate_model import ModelEvaluator
from src.orchestration.validation import DataValidator
from src.utils.logger import logger
from prefect.cache_policies import NO_CACHE

@task(name="Load Raw Data", retries=3, retry_delay_seconds=[30, 60, 120], tags=["data", "ingestion"])
def load_data_task(config_path: str) -> pd.DataFrame:
    import yaml
    # Dynamically extract the core config path from the pipeline params
    with open(config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f).get("training_pipeline", {})
    main_cfg_path = pipeline_cfg.get("training", {}).get("config_path", "configs/config.yaml")
    
    loader = LoadData(main_cfg_path)
    return loader.load_data()

@task(name="Validate Data", retries=0, tags=["data", "validation"])
def validate_data_task(df: pd.DataFrame, config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f).get("orchestration", {}).get("validation", {})
    
    validator = DataValidator(validation_config=config)
    result = validator.validate(df)
    
    if not result.success:
        raise Abort(f"Data validation failed: {result.summary}")
    return result

@task(name="Clean Data", retries=2, retry_delay_seconds=15, tags=["data", "preprocessing"])
def clean_data_task(df: pd.DataFrame):
    cleaner = CleanDataBERT()
    # Returns (train_dataset, val_dataset, test_dataset, test_labels)
    return cleaner.prepare_datasets(df)

@task(name="Train Model", retries=1, retry_delay_seconds=300, cache_policy=NO_CACHE, tags=["model", "training"])
def train_model_task(df: pd.DataFrame, config_path: str):
    with open(config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f).get("training_pipeline", {})
    training_cfg = pipeline_cfg.get("training", {})
    
    target_column = training_cfg.get("target_column", "label")
    model_type = training_cfg.get("model_type", "distilbert")
    
    trainer = ModelTrainer(dataframe=df, yaml_config_path=config_path, target_column=target_column)
    model = trainer.train_model(model_type)
    return model, trainer

@task(name="Evaluate Model", retries=1, retry_delay_seconds=60, cache_policy=NO_CACHE, tags=["model", "evaluation"])
def evaluate_model_task(trainer, config_path: str):
    with open(config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f).get("training_pipeline", {})
    
    model_type = pipeline_cfg.get("training", {}).get("model_type", "distilbert")
    eval_cfg = pipeline_cfg.get("evaluation", {})
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(trainer.trainer, trainer.test_dataset, model_type)
    
    evaluator.save_results({model_type: metrics}, output_dir=eval_cfg.get("save_dir", "artifacts/evaluation"))
    return metrics

@task(name="Quality Gate", retries=0, tags=["model", "deployment"])
def quality_gate_task(metrics: dict, config_path: str) -> bool:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    gate_cfg = config.get("orchestration", {}).get("quality_gate", {})
    min_improvement = gate_cfg.get("min_f1_improvement", 0.01)
    baseline_metric = gate_cfg.get("baseline_metric", "f1")
    first_run_auto_deploy = gate_cfg.get("first_run_auto_deploy", True)
    
    experiment_name = config.get("mlflow", {}).get("experiment_name", "distilbert_training")
    mlflow.set_experiment(experiment_name)
    
    # Query MLflow for production baseline
    runs = mlflow.search_runs(
        filter_string="tags.is_production = 'true'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    new_score = metrics.get(baseline_metric, 0)
    
    if runs.empty:
        if first_run_auto_deploy:
            logger.info("No baseline found. First run auto-deploy enabled. Quality gate passed.")
            return True
        else:
            logger.info("No baseline found and auto-deploy disabled. Quality gate failed.")
            return False
            
    baseline_score = runs.iloc[0].get(f"metrics.eval_{baseline_metric}", 0)
    
    logger.info(f"Quality Gate - Baseline {baseline_metric}: {baseline_score:.4f} | New {baseline_metric}: {new_score:.4f}")
    
    if new_score >= baseline_score + min_improvement:
        logger.info("Quality gate passed. New model outperforms baseline.")
        return True
        
    logger.info("Quality gate failed. New model did not meet improvement threshold.")
    return False


@task(name="Deploy Model", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE, tags=["model", "deployment"])
def deploy_model_task(model, trainer, config_path: str):
    logger.info("Deploying model (Tagging in MLflow as production)")
    
    # We tag the most recent run with is_production=true
    with open(config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f).get("training_pipeline", {})
    model_type = pipeline_cfg.get("training", {}).get("model_type", "distilbert")
    eval_cfg = pipeline_cfg.get("evaluation", {})
    
    evaluator = ModelEvaluator()
    evaluator.save_best_model(model, output_dir=eval_cfg.get("best_model_dir", "artifacts/best_model"))
    
    # Retrieve active MLflow run id from trainer if possible, or tag the latest run
    # trainer creates a run in `train_model()`. We can find the latest run in the experiment.
    experiment_name = yaml.safe_load(open(config_path, "r")).get("mlflow", {}).get("experiment_name", "distilbert_training")
    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if not runs.empty:
        run_id = runs.iloc[0].run_id
        client = mlflow.tracking.MlflowClient()
        client.set_tag(run_id, "is_production", "true")
        logger.info(f"Tagged MLflow run {run_id} as production.")
        
        # Save Reference Data for Drift Monitoring
        try:
            from src.monitoring.reference_store import ReferenceStore
            dataset_version = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M")
            ref_store = ReferenceStore()
            # trainer.train_dataset should be available (from trainer instantiation in train_model_task)
            # To get raw dataframe that was used, we could access trainer.dataframe or reconstruct
            # CleanDataBERT prepares HuggingFace Datasets, so we convert back to Pandas if needed
            # For simplicity, we assume `trainer.dataframe` holds the original cleaned df 
            # if we pass it, but trainer does not store `dataframe` directly as attribute in current implementation.
            # Wait, ModelTrainer takes `dataframe=df` in its constructor. 
            if hasattr(trainer, "df") and isinstance(trainer.df, pd.DataFrame):
                ref_df = trainer.df
            else:
                # If dataframe wasn't saved on trainer, we use the huggingface dataset
                ref_df = pd.DataFrame(trainer.train_dataset)
                
            ref_store.save(ref_df, run_id, dataset_version)
        except Exception as e:
            logger.warning(f"Failed to save reference dataset for monitoring: {e}")
            
    else:
        logger.warning("Could not find MLflow run to tag as production.")


from src.orchestration.search_tasks import rebuild_search_index_task

@flow(name="ReviewSentinel Training Pipeline", description="Weekly automated training pipeline for ReviewSentinel", log_prints=True)
def training_flow(config_path: str = "configs/pipeline_params.yaml"):
    logger.info(f"Starting Prefect training flow with config: {config_path}")
    
    # 1. Load Data
    df = load_data_task(config_path)
    
    # 2. Validate Data
    validate_data_task(df, config_path)
    
    # 3. Clean Data (we run this to ensure logic is covered as requested, but ModelTrainer recleans)
    clean_data_task(df)
    
    # 4. Train Model
    model, trainer = train_model_task(df, config_path)
    
    # 5. Evaluate Model
    metrics = evaluate_model_task(trainer, config_path)
    
    # 6. Quality Gate
    should_deploy = quality_gate_task(metrics, config_path)
    
    # 7. Deploy Model Conditionally
    index_rebuilt = None
    if should_deploy:
        deploy_model_task(model, trainer, config_path)
        
        # 8. Rebuild Semantic Search Index
        index_rebuilt = rebuild_search_index_task(config_path)
        
    logger.info(
        f"Flow Summary - F1 Score: {metrics.get('f1', 0):.4f} | "
        f"Deployed: {should_deploy} | Index Rebuilt: {index_rebuilt}"
    )

from src.orchestration.monitoring_tasks import (
    load_prediction_log_task,
    generate_drift_report_task,
    evaluate_drift_task,
    send_drift_alert_task,
    trigger_retraining_task,
    DriftAction
)

@flow(name="ReviewSentinel Drift Monitoring", description="Weekly automated drift monitoring", log_prints=True)
def monitoring_flow(config_path: str = "configs/pipeline_params.yaml"):
    logger.info(f"Starting Prefect Drift Monitoring Flow with config: {config_path}")
    
    # 1. Load Prediction Logs
    df = load_prediction_log_task(config_path)
    
    if df is None or df.empty:
        logger.info("No prediction logs to monitor. Exiting.")
        return
        
    # 2. Generate Report
    result = generate_drift_report_task(df, config_path)
    
    if result is None:
        logger.info("Drift report generation skipped or failed.")
        return
        
    # 3. Evaluate Drift
    action = evaluate_drift_task(result, config_path)
    
    # 4 & 5. Alerts and Retraining
    if action == DriftAction.ALERT:
        send_drift_alert_task(result, action)
    elif action == DriftAction.TRIGGER_RETRAINING:
        send_drift_alert_task(result, action)
        trigger_retraining_task()
        
    logger.info(f"Drift Monitoring complete. Final Action: {action.value}")

from src.orchestration.judge_tasks import (
    check_ollama_health_task,
    dequeue_pending_task,
    run_judge_batch_task,
    process_results_task,
    log_judge_metrics_task
)

@flow(name="ReviewSentinel LLM Judge Processing", description="Async LLM Judge processing queue", log_prints=True)
def judge_processing_flow(config_path: str = "configs/pipeline_params.yaml"):
    logger.info("Starting LLM Judge Processing Flow...")
    
    # 1. Check Ollama health
    check_ollama_health_task(config_path)
    
    # 2. Dequeue pending entries
    import yaml
    with open(config_path, "r") as f:
        batch_size = yaml.safe_load(f).get("llm_judge", {}).get("queue", {}).get("batch_size", 50)
        
    entries = dequeue_pending_task(batch_size=batch_size, config_path=config_path)
    
    if not entries:
        logger.info("Queue empty, completing flow.")
        return
        
    # 3. Run inference
    results = run_judge_batch_task(entries, config_path)
    
    # 4. Process results
    summary = process_results_task(entries, results, config_path)
    
    # 5. Log metrics
    log_judge_metrics_task(summary, config_path)
    
    logger.info(f"LLM Judge flow complete. Processed: {summary['processed']}, Conflicts: {summary['conflicts']}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "monitor":
            monitoring_flow()
        elif sys.argv[1] == "judge":
            judge_processing_flow()
        else:
            training_flow()
    else:
        training_flow()
