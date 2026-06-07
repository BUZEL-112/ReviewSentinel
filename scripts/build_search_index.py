import argparse
import logging
from pathlib import Path
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(project_root)

from src.search.index_builder import IndexBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS Search Index")
    parser.add_argument("--config", type=str, default="configs/pipeline_params.yaml", help="Path to pipeline_params.yaml")
    parser.add_argument("--output-dir", type=str, default="artifacts/search/", help="Directory to save index artifacts")
    parser.add_argument("--rebuild-if-exists", action="store_true", help="Force rebuild even if index already exists")
    
    args = parser.parse_args()
    
    index_path = Path(args.output_dir) / "faiss.index"
    if index_path.exists() and not args.rebuild_if_exists:
        logger.info(f"Index already exists at {index_path}. Skipping build. Use --rebuild-if-exists to force.")
        return
        
    builder = IndexBuilder()
    builder.build_from_pipeline(config_path=args.config, output_dir=args.output_dir)
    
if __name__ == "__main__":
    main()
