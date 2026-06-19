import httpx
import datetime
from prefect import flow, task
from prefect.deployments import run_deployment
from src.utils.logger import logger 

@task(retries=2, retry_delay_seconds=5)
def check_api_health(api_url: str = "http://api:8000/health") -> bool:
    try:
        response = httpx.get(api_url, timeout=5.0)
        response.raise_for_status()
        health_data = response.json()
        
        if health_data.get("model_loaded") is True:
            logger.info("API is healthy and model is loaded.")
            return True
        else:
            logger.warning("API is up, but model is NOT loaded.")
            return False
    except Exception as e:
        logger.error(f"Failed to reach API health endpoint: {e}")
        return False

@flow(name="ReviewSentinel Watchdog")
def watchdog_flow():
    """Continuously monitors API health and triggers retraining if the model is missing."""
    is_healthy = check_api_health()
    
    if not is_healthy:
        # 1. Create a Time-Bucketed Idempotency Key
        # Generates a string bound to the current hour (e.g., "recovery-2026-06-09-10")
        current_hour = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        recovery_key = f"recovery-{current_hour}"
        
        logger.warning(f"Health check failed. Sending trigger with idempotency key: {recovery_key}")
        
        # 2. Trigger the deployment with the key
        run_deployment(
            name="ReviewSentinel Training Pipeline/weekly-training-pipeline",
            timeout=0, # Don't block the watchdog
            idempotency_key=recovery_key 
        )
        logger.info("Self-healing trigger sent (redundant triggers this hour will be dropped).")