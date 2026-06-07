import sys
import yaml
import socket
from urllib.parse import urlparse
from pathlib import Path
from src.utils.exception import CustomException

def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise CustomException(FileNotFoundError(f"Config file not found at {config_path}"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



def is_host_reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        with socket.create_connection((host, port), timeout=timeout):
            return True

    except (socket.gaierror, socket.timeout, ConnectionRefusedError, OSError):
        return False


def resolve_tracking_uri(
    tracking_uri: str,
    fallback_uri: str = "http://localhost:5000",
) -> str:
    """
    Return the first reachable MLflow URI.
    """
    if is_host_reachable(tracking_uri):
        return tracking_uri

    if is_host_reachable(fallback_uri):
        return fallback_uri

    raise ConnectionError(
        f"Neither '{tracking_uri}' nor '{fallback_uri}' is reachable."
    )