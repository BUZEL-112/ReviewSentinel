import yaml
import os
import logging
from pathlib import Path
import pandas as pd
import mlflow

from src.search.encoder import SentenceEncoder
from src.search.indexer import FAISSIndexer

logger = logging.getLogger(__name__)

class IndexBuilder:

    def build_from_dataframe(self, df: pd.DataFrame, output_dir: str, config_path: str = "configs/pipeline_params.yaml"):
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            cfg = full_cfg.get("semantic_search", {})

        encoder_cfg = cfg.get("encoder", {})
        model_name = encoder_cfg.get("model_name", "all-MiniLM-L6-v2")
        batch_size = encoder_cfg.get("batch_size", 64)

        encoder = SentenceEncoder(model_name=model_name, batch_size=batch_size)
        indexer = FAISSIndexer(index_dir=output_dir)
        indexer.build(df, encoder, text_column="clean_text")

        # Consistent env-var-first resolution, same pattern as train_model.py
        config_uri = full_cfg.get("mlflow", {}).get("tracking_uri", "./mlruns")
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI") or config_uri

        mlflow.set_tracking_uri(mlflow_uri)
        
        try:
            # THIS is the line that actually makes the network call and crashes if outside Docker
            mlflow.set_experiment("search_index")
            logger.info(f"Successfully connected to MLflow at {mlflow_uri}")
            
        except Exception as e:
            # 3. Catch the DNS/Connection error and fall back to localhost
            fallback_uri = "http://localhost:5000"
            logger.warning(f"Could not reach {mlflow}. Falling back to {fallback_uri}. Error: {e}")
            
            mlflow.set_tracking_uri(fallback_uri)
            # Retry the connection with localhost
            mlflow.set_experiment("search_index")
            logger.info("Successfully connected to MLflow via localhost.")

        with mlflow.start_run(run_name="build_index", nested=True):
            mlflow.log_params({
                "corpus_size": len(df),
                "model_name": model_name,
                "embedding_dim": encoder.get_embedding_dim(),
                "index_type": "IndexFlatIP"
            })
        logger.info("Search index build successfully logged to MLflow.")

    def build_from_pipeline(self, config_path: str, output_dir: str):
        from src.data.load_data import LoadData
        from src.data.clean_data import CleanDataBERT
        import yaml

        with open(config_path, "r") as f:
            pipeline_cfg = yaml.safe_load(f)
        core_config_path = (
            pipeline_cfg.get("training_pipeline", {})
                        .get("training", {})
                        .get("config_path", "configs/config.yaml")
        )

        loader = LoadData(core_config_path)
        df = loader.load_data()
        # df = df.sample(frac=0.001).copy()
        # df= df[:100]

        cleaner = CleanDataBERT()
        df["clean_text"] = df["text"].apply(cleaner._minimal_clean)

        # Cap corpus size — 50k is plenty for semantic search, full 700k takes ~2hrs on CPU
        max_index_rows = pipeline_cfg.get("semantic_search", {}).get("build", {}).get("max_index_rows", 500)
        if len(df) > max_index_rows:
            logger.info(f"Sampling {max_index_rows} rows from {len(df)} for index build...")
            df = df.sample(n=max_index_rows, random_state=42).reset_index(drop=True)

        self.build_from_dataframe(df, output_dir=output_dir, config_path=config_path)