import argparse
import sys
import os
from prefect import flow

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.orchestration.flows import training_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger the ReviewSentinel training flow locally.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/pipeline_params.yaml", 
        help="Path to the pipeline configuration YAML file."
    )
    args = parser.parse_args()
    
    print(f"Starting Prefect training flow with config: {args.config}")
    training_flow(config_path=args.config)
