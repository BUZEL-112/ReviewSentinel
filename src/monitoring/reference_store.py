import os
import json
import datetime
import pandas as pd
import mlflow
from typing import Optional
from src.utils.logger import logger

class ReferenceStore:
    """
    Manages the reference dataset serialization (Parquet) and MLflow tracking.
    """
    def __init__(self, mlflow_tracking_uri: Optional[str] = None, local_cache_dir: str = "artifacts/monitoring/reference/"):
        self.local_cache_dir = local_cache_dir
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        os.makedirs(self.local_cache_dir, exist_ok=True)

    def save(self, df: pd.DataFrame, run_id: str, dataset_version: str):
        """
        Saves the reference dataset as a Parquet file locally and to MLflow,
        tagging the run so it can be retrieved dynamically.
        """
        if "clean_text" not in df.columns or "sentiment" not in df.columns:
            logger.warning("Reference dataset missing 'clean_text' or 'sentiment' columns. This may break drift monitoring.")
        
        # Keep only essential columns to save space
        cols_to_keep = ["clean_text", "sentiment"]
        available_cols = [c for c in cols_to_keep if c in df.columns]
        ref_df = df[available_cols].copy()
        # Rename sentiment to predicted_label for drift monitor matching (it compares to prediction outputs)
        if "sentiment" in ref_df.columns:
            ref_df.rename(columns={"sentiment": "predicted_label"}, inplace=True)
            
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Paths
        parquet_path = os.path.join(self.local_cache_dir, f"reference_{dataset_version}.parquet")
        meta_path = os.path.join(self.local_cache_dir, f"reference_{dataset_version}_meta.json")
        
        # Save Parquet
        ref_df.to_parquet(parquet_path, index=False)
        
        # Save Meta
        metadata = {
            "run_id": run_id,
            "dataset_version": dataset_version,
            "saved_at": timestamp,
            "row_count": len(ref_df),
            "column_list": list(ref_df.columns)
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Log to MLflow
        logger.info(f"Logging reference data to MLflow run {run_id}")
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(run_id, parquet_path, "reference_data")
        client.log_artifact(run_id, meta_path, "reference_data")
        
        # Tag run
        client.set_tag(run_id, "has_reference_data", "true")
        client.set_tag(run_id, "reference_version", dataset_version)
        logger.info(f"Reference dataset {dataset_version} saved and tracked.")

    def load(self, run_id: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieves the reference dataset from MLflow. If run_id is None, finds
        the latest production run with reference data. Local cache is checked first.
        """
        client = mlflow.tracking.MlflowClient()
        
        if not run_id:
            logger.info("No run_id provided. Searching for latest production run with reference data...")
            runs = mlflow.search_runs(
                filter_string="tags.is_production = 'true' and tags.has_reference_data = 'true'",
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs.empty:
                logger.error("Could not find any production run with reference data.")
                return None
            run_id = runs.iloc[0].run_id
            
        logger.info(f"Loading reference data for run {run_id}")
        run = client.get_run(run_id)
        dataset_version = run.data.tags.get("reference_version")
        
        if not dataset_version:
            logger.error(f"Run {run_id} missing 'reference_version' tag.")
            return None
            
        # Check cache
        cached_parquet_path = os.path.join(self.local_cache_dir, f"reference_{dataset_version}.parquet")
        if os.path.exists(cached_parquet_path):
            logger.info(f"Loading reference dataset from local cache: {cached_parquet_path}")
            return pd.read_parquet(cached_parquet_path)
            
        # Download from MLflow
        logger.info(f"Cache miss. Downloading reference dataset from MLflow artifacts...")
        try:
            download_dir = client.download_artifacts(run_id, "reference_data", self.local_cache_dir)
            # download_artifacts returns the local path to the directory downloaded
            downloaded_parquet = os.path.join(download_dir, f"reference_{dataset_version}.parquet")
            
            # Move out of the nested folder to the cache dir
            import shutil
            if os.path.exists(downloaded_parquet):
                shutil.move(downloaded_parquet, cached_parquet_path)
                
            return pd.read_parquet(cached_parquet_path)
        except Exception as e:
            logger.error(f"Failed to load reference data from MLflow: {e}")
            return None
