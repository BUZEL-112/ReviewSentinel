import logging
from prefect import task
from src.search.index_builder import IndexBuilder

logger = logging.getLogger(__name__)

@task(name="Rebuild Search Index", retries=2, retry_delay_seconds=60, tags=["search", "index"])
def rebuild_search_index_task(config_path: str = "configs/pipeline_params.yaml", output_dir: str = "artifacts/search/"):
    """
    Rebuilds the FAISS semantic search index using the current training dataset.
    Catches and logs exceptions internally so that index build failures do not
    crash the main training deployment flow.
    """
    try:
        logger.info("Starting Semantic Search index rebuild...")
        builder = IndexBuilder()
        builder.build_from_pipeline(config_path=config_path, output_dir=output_dir)
        logger.info("Semantic Search index rebuilt successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to rebuild search index: {e}", exc_info=True)
        # We don't raise here because we want to allow_failure in the main flow.
        # Semantic search degrades gracefully if the index is missing/stale.
        return False
