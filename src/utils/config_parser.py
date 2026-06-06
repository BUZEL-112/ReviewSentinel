import sys
import yaml
from pathlib import Path
from src.utils.exception import CustomException

def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise CustomException(FileNotFoundError(f"Config file not found at {config_path}"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config