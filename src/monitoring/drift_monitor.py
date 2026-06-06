import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
import pandas as pd
import mlflow
from evidently import Report
# from evidently.metric_preset import TextEvals
from evidently.presets import DataDriftPreset,TextEvals
# from evidently.metrics import DatasetSummaryMetric, ColumnDriftMetric
# from evidently.metrics.data_integrity.dataset_summary_metric import DatasetSummaryMetric
# from evidently.metrics.data_drift.column_drift_metric import ColumnDriftMetric
from evidently.legacy.metrics.data_integrity.dataset_summary_metric import DatasetSummaryMetric
from evidently.legacy.metrics.data_drift.column_drift_metric import ColumnDriftMetric
from src.monitoring.reference_store import ReferenceStore
from src.utils.logger import logger

@dataclass
class DriftResult:
    drift_detected: bool
    drift_score: float              
    features_drifted: int           
    total_features: int
    drift_fraction: float           
    predicted_label_drifted: bool   
    reference_row_count: int
    current_row_count: int
    report_path: str                
    mlflow_run_id: str              
    timestamp: str
    summary: str                    

class DriftMonitor:
    """
    Core monitoring logic to compare current prediction inputs against reference training data
    using Evidently AI statistical tests.
    """
    def __init__(self, reference_store: ReferenceStore, config: Dict[str, Any], mlflow_tracking_uri: Optional[str] = None):
        self.reference_store = reference_store
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri
        
        self.pred_log_path = self.config.get("prediction_log", {}).get("path", "artifacts/monitoring/prediction_log.jsonl")
        self.time_window_days = self.config.get("drift", {}).get("time_window_days", 7)
        self.min_current_samples = self.config.get("drift", {}).get("min_current_samples", 50)
        self.output_dir = self.config.get("reports", {}).get("output_dir", "artifacts/monitoring/reports")
        self.keep_last_n = self.config.get("reports", {}).get("keep_last_n", 12)
        
        os.makedirs(self.output_dir, exist_ok=True)
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def _load_current_data(self, source: Union[str, pd.DataFrame] = "prediction_log") -> Optional[pd.DataFrame]:
        """Loads and filters the current dataset from a JSONL log or raw DataFrame."""
        if isinstance(source, pd.DataFrame):
            return source

        if not os.path.exists(self.pred_log_path):
            logger.warning(f"Prediction log not found at {self.pred_log_path}")
            return None

        try:
            df = pd.read_json(self.pred_log_path, lines=True)
            if df.empty:
                return None
            
            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Filter by time window
            cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=self.time_window_days)
            
            # Make cutoff timezone-naive if df is naive, or df timezone-aware if cutoff is aware
            if df["timestamp"].dt.tz is None:
                cutoff_date = cutoff_date.replace(tzinfo=None)
                
            filtered_df = df[df["timestamp"] >= cutoff_date].copy()
            
            if filtered_df.empty:
                logger.info(f"No prediction logs found in the last {self.time_window_days} days.")
                return None
                
            # Rename columns to match reference (clean_text, predicted_label)
            if "input_text" in filtered_df.columns:
                filtered_df.rename(columns={"input_text": "clean_text"}, inplace=True)
                
            logger.info(f"Loaded {len(filtered_df)} predictions from the last {self.time_window_days} days.")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error loading prediction log: {e}")
            return None

    def _prune_old_reports(self):
        """Keeps only the keep_last_n recent HTML reports to prevent disk bloat."""
        try:
            reports = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) if f.endswith(".html")]
            if len(reports) > self.keep_last_n:
                reports.sort(key=os.path.getmtime)
                to_delete = reports[:-self.keep_last_n]
                for file_path in to_delete:
                    os.remove(file_path)
                logger.info(f"Pruned {len(to_delete)} old drift reports.")
        except Exception as e:
            logger.warning(f"Failed to prune old reports: {e}")

    def generate_report(self, current_source: Union[str, pd.DataFrame] = "prediction_log", reference_df: Optional[pd.DataFrame] = None) -> Optional[DriftResult]:
        """
        Executes Evidently tests on current vs reference data and logs the results to MLflow.
        """
        # Load Reference
        if reference_df is None:
            reference_df = self.reference_store.load()
            if reference_df is None:
                logger.error("Could not load reference data. Aborting drift report generation.")
                return None

        # Load Current
        current_df = self._load_current_data(current_source)
        if current_df is None:
            logger.error("Could not load current data. Aborting drift report generation.")
            return None

        # Validate minimum sample count
        ref_rows = len(reference_df)
        curr_rows = len(current_df)
        
        if curr_rows < self.min_current_samples:
            logger.warning(f"Current dataset has {curr_rows} rows, which is less than the required minimum of {self.min_current_samples}. Skipping report.")
            return None

        logger.info(f"Generating drift report: {curr_rows} current samples vs {ref_rows} reference samples.")

        # Ensure schema match
        for col in ["clean_text", "predicted_label"]:
            if col not in reference_df.columns or col not in current_df.columns:
                logger.warning(f"Missing required column '{col}' in datasets. Reference: {list(reference_df.columns)}, Current: {list(current_df.columns)}")

        # Configure Evidently Report
        report = Report(metrics=[
            TextEvals(column_name="clean_text"),
            DataDriftPreset(columns=["clean_text"]),
            DatasetSummaryMetric(),
            ColumnDriftMetric(column_name="predicted_label")
        ])

        # Run Report
        try:
            report.run(reference_data=reference_df, current_data=current_df)
        except Exception as e:
            logger.error(f"Evidently report execution failed: {e}")
            return None

        # Extract Verdict
        report_dict = report.as_dict()
        
        # We need to dig into the metrics array to find the DataDrift preset result
        # The exact structure depends on Evidently's output format, usually DataDriftPreset produces a dataset_drift metric
        dataset_drift = False
        drift_share = 0.0
        drifted_features = 0
        total_features = 1
        label_drifted = False

        for metric in report_dict["metrics"]:
            metric_type = metric["metric"]
            
            if metric_type == "DatasetDriftMetric":
                dataset_drift = metric["result"]["dataset_drift"]
                drift_share = metric["result"]["share_of_drifted_columns"]
                drifted_features = metric["result"]["number_of_drifted_columns"]
                total_features = metric["result"]["number_of_columns"]
            
            elif metric_type == "ColumnDriftMetric":
                if metric["result"]["column_name"] == "predicted_label":
                    label_drifted = metric["result"]["drift_detected"]

        # If DatasetDriftMetric wasn't found (sometimes it's grouped differently), calculate an approximation
        if total_features == 1 and drifted_features == 0 and drift_share == 0.0:
             drifted_features = 1 if label_drifted else 0
             total_features = 2 # text + label
             drift_share = drifted_features / total_features
             dataset_drift = drift_share > self.config.get("drift", {}).get("alert_threshold", 0.3)

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"drift_report_{timestamp}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Save HTML
        report.save_html(report_path)
        logger.info(f"Drift report saved to {report_path}")
        
        self._prune_old_reports()

        summary_text = (
            f"Drift detected: {dataset_drift} | "
            f"{drifted_features}/{total_features} features drifted ({drift_share:.1%}). "
            f"Label drift: {label_drifted}. "
            f"Current: {curr_rows} vs Reference: {ref_rows} samples."
        )

        # Log to MLflow
        mlflow.set_experiment("drift_monitoring")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_artifact(report_path)
            mlflow.log_metrics({
                "drift_score": drift_share,
                "drifted_features": drifted_features,
                "current_row_count": curr_rows,
                "label_drift_flag": 1.0 if label_drifted else 0.0
            })
            mlflow.set_tag("drift_detected", str(dataset_drift).lower())
            mlflow.set_tag("summary", summary_text)
            
        logger.info(summary_text)

        return DriftResult(
            drift_detected=dataset_drift,
            drift_score=drift_share,
            features_drifted=drifted_features,
            total_features=total_features,
            drift_fraction=drift_share,
            predicted_label_drifted=label_drifted,
            reference_row_count=ref_rows,
            current_row_count=curr_rows,
            report_path=report_path,
            mlflow_run_id=run_id,
            timestamp=timestamp,
            summary=summary_text
        )
