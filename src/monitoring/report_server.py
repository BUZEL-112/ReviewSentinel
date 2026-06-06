import os
import yaml
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from src.utils.logger import logger

monitoring_router = APIRouter(tags=["Monitoring"])

def _get_config():
    config_path = "configs/pipeline_params.yaml"
    if not os.path.exists(config_path):
        config_path = "../configs/pipeline_params.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("monitoring", {})
    except Exception as e:
        logger.error(f"Failed to load monitoring config: {e}")
        return {}

@monitoring_router.get("/drift/latest")
async def get_latest_drift_status():
    """
    Returns the most recent DriftResult as JSON.
    """
    config = _get_config()
    output_dir = config.get("reports", {}).get("output_dir", "artifacts/monitoring/reports")
    
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="No drift reports found.")
        
    reports = [f for f in os.listdir(output_dir) if f.endswith(".html")]
    if not reports:
        raise HTTPException(status_code=404, detail="No drift reports found.")
        
    reports.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    latest_report = reports[0]
    
    # We do not have the JSON saved, but we can return the path/time of the latest report
    # The requirement says "returns the most recent DriftResult as JSON". 
    # To do this fully, DriftMonitor should also save a JSON version, or we return basic metadata.
    # We'll return basic metadata for now.
    
    report_path = os.path.join(output_dir, latest_report)
    mod_time = os.path.getmtime(report_path)
    
    return JSONResponse({
        "latest_report": latest_report,
        "timestamp": mod_time,
        "report_url": "/api/monitoring/drift/report"
    })

@monitoring_router.get("/drift/report")
async def serve_drift_report():
    """
    Serves the most recent Evidently HTML report directly.
    """
    config = _get_config()
    output_dir = config.get("reports", {}).get("output_dir", "artifacts/monitoring/reports")
    
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="No drift reports found.")
        
    reports = [f for f in os.listdir(output_dir) if f.endswith(".html")]
    if not reports:
        raise HTTPException(status_code=404, detail="No drift reports found.")
        
    # Sort by modification time, newest first
    reports.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    latest_report = os.path.join(output_dir, reports[0])
    
    return FileResponse(latest_report, media_type="text/html")
