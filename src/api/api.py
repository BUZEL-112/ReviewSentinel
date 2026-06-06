"""
Sentiment Analysis API - Production Grade

FastAPI application wrapping the DistilBERT + SetFit InferencePipeline.

Features:
    - Rich response schema: sentiment scores, aspect, processing time, model
      version, and edge-case flags.
    - Async inference via thread-pool executor so the event loop is never
      blocked by model forward passes.
    - Batch job queue: submit a batch, receive a job_id immediately, poll
      /predict/batch/{job_id} for results.
    - Production health endpoint: uptime, model version, dependency status.
    - Prometheus metrics at /metrics: request counters, latency histograms,
      confidence distribution, and error rates.
"""

import os
import sys
import uuid
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

import json
import gzip
import shutil

# --------------------------------------------------------------------------
# Project root path so src.* imports resolve when running from any directory
# --------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.pipeline.inference_pipeline import InferencePipeline
from src.explainability.shap_explainer import SentimentExplainer

# --------------------------------------------------------------------------
# Prometheus metrics (defined at module level — one instance for the process)
# --------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "sentiment_predictions_total",
    "Total number of prediction requests served.",
    ["endpoint", "label"],
)
PREDICTION_ERRORS = Counter(
    "sentiment_prediction_errors_total",
    "Total number of prediction errors.",
    ["endpoint"],
)
PREDICTION_LATENCY = Histogram(
    "sentiment_prediction_latency_seconds",
    "End-to-end prediction latency in seconds.",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
CONFIDENCE_DISTRIBUTION = Histogram(
    "sentiment_confidence_score",
    "Distribution of model confidence scores.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
BATCH_JOBS_ACTIVE = Gauge(
    "sentiment_batch_jobs_active",
    "Number of batch jobs currently being processed.",
)

# --------------------------------------------------------------------------
# Global state
# --------------------------------------------------------------------------
pipeline: Optional[InferencePipeline] = None
explainer: Optional[SentimentExplainer] = None
startup_time: float = time.time()

# Batch job store: {job_id: {"status": str, "result": list | None, "error": str | None}}
batch_jobs: Dict[str, dict] = {}

# Thread pool used to run synchronous inference off the event loop
_executor = ThreadPoolExecutor(max_workers=4)

CONFIG_PATHS = [
    "configs/pipeline_params.yaml",
    "configs/inference_pipeline.yaml",
    os.path.join(project_root, "configs", "pipeline_params.yaml"),
    os.path.join(project_root, "configs", "inference_pipeline.yaml"),
]

# Try to read a semantic version from a VERSION file at the project root
_version_file = os.path.join(project_root, "VERSION")
MODEL_VERSION = open(_version_file).read().strip() if os.path.exists(_version_file) else "unknown"


# --------------------------------------------------------------------------
# Pipeline initialization
# --------------------------------------------------------------------------
def _initialize_pipeline() -> bool:
    """Locate config and instantiate the InferencePipeline. Returns success flag."""
    global pipeline
    try:
        config_path = next((p for p in CONFIG_PATHS if os.path.exists(p)), None)
        if config_path is None:
            logger.error("No inference config found. Checked: " + ", ".join(CONFIG_PATHS))
            return False
        logger.info(f"Initializing InferencePipeline from: {config_path}")
        pipeline = InferencePipeline(config_path=config_path)
        logger.info("InferencePipeline ready.")
        return True
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
        pipeline = None
        return False


def _initialize_explainer():
    """Lazily initialize the SentimentExplainer."""
    global explainer, pipeline
    if explainer is None and pipeline is not None:
        try:
            logger.info("Initializing SentimentExplainer...")
            infer_cfg = pipeline.config.get("explainability", {})
            if not infer_cfg.get("enabled", True):
                logger.info("Explainability is disabled in config.")
                return
            
            max_evals = infer_cfg.get("max_evals", 500)
            top_k_tokens = infer_cfg.get("top_k_tokens", 10)
            from src.pipeline.inference_pipeline import SENTIMENT_MAP
            explainer = SentimentExplainer(
                model=pipeline.model,
                tokenizer=pipeline.tokenizer,
                device=pipeline.device,
                max_evals=max_evals,
                label_map=SENTIMENT_MAP,
                top_k_tokens=top_k_tokens
            )
            logger.info("SentimentExplainer ready.")
        except Exception as e:
            logger.error(f"Explainer init failed: {e}")


# --------------------------------------------------------------------------
# Prediction Logger
# --------------------------------------------------------------------------
class PredictionLogger:
    """Appends predictions to a JSONL file, rotating if it exceeds 100MB."""
    def __init__(self, log_path: str = "artifacts/monitoring/prediction_log.jsonl", max_size_mb: int = 100):
        self.log_path = log_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def _rotate_if_needed(self):
        if not os.path.exists(self.log_path):
            return
        if os.path.getsize(self.log_path) > self.max_size_bytes:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated_path = f"{self.log_path}.{timestamp}.gz"
            try:
                with open(self.log_path, 'rb') as f_in:
                    with gzip.open(rotated_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(self.log_path)
                logger.info(f"Rotated prediction log to {rotated_path}")
            except Exception as e:
                logger.error(f"Failed to rotate prediction log: {e}")

    def log_prediction(self, text: str, label: str, confidence: float, model_version: str):
        self._rotate_if_needed()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_text": text,
            "predicted_label": label,
            "confidence": float(confidence),
            "model_version": model_version
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

prediction_logger = PredictionLogger()

# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_time
    startup_time = time.time()
    logger.info("API startup — initializing pipeline...")
    _initialize_pipeline()
    yield
    logger.info("API shutdown.")


# --------------------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "a scalable machine learning pipeline for analysing sentiment of custumer review"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from src.monitoring.report_server import monitoring_router
    app.include_router(monitoring_router, prefix="/monitoring")
except Exception as e:
    logger.error(f"Could not load monitoring router: {e}")


# --------------------------------------------------------------------------
# Pydantic schemas
# --------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    title: str           = Field(..., description="Review title.")
    text: Optional[str]  = Field(None, description="Review body text (optional).")
    include_explanation: bool = Field(False, description="Whether to include SHAP explanations.")
    include_similar_reviews: bool = Field(False, description="Whether to return semantically similar historical reviews.")
    similar_reviews_count: int = Field(5, le=10, description="Number of similar reviews to return (max 10).")


class BatchPredictionRequest(BaseModel):
    items: List[SinglePredictionRequest] = Field(
        ..., min_length=1, description="List of title/text pairs to classify."
    )


class SentimentScores(BaseModel):
    """Full softmax distribution across all three sentiment classes."""
    negative: float = Field(..., description="Probability of negative sentiment.")
    neutral:  float = Field(..., description="Probability of neutral sentiment.")
    positive: float = Field(..., description="Probability of positive sentiment.")


class AspectScore(BaseModel):
    """Aspect classification result from the SetFit model."""
    label: str   = Field(..., description="Predicted aspect category.")


class PredictionFlags(BaseModel):
    """Edge-case flags raised during inference."""
    short_review:    bool = Field(..., description="Input is unusually short (< 10 chars).")
    low_confidence:  bool = Field(..., description="Winning confidence is below 0.60.")
    possible_sarcasm: bool = Field(..., description="Heuristic sarcasm pattern detected.")


class SearchResultItem(BaseModel):
    rank: int
    similarity_score: float
    clean_text: str
    raw_title: str
    raw_text: str
    known_sentiment: str
    rating: float
    sentiment_alignment: Optional[bool]


class SearchResponseModel(BaseModel):
    results: List[SearchResultItem]
    query_text: str
    top_k: int
    alignment_rate: Optional[float]
    alignment_signal: Optional[str]
    search_latency_ms: float


class PredictionResult(BaseModel):
    """Full per-sample prediction output."""
    text:             str                    = Field(..., description="Cleaned input text passed to the model.")
    label:            str                    = Field(..., description="Predicted sentiment: negative | neutral | positive.")
    confidence:       float                  = Field(..., description="Softmax probability for the winning class.")
    scores:           SentimentScores        = Field(..., description="Full class probability distribution.")
    aspect:           Optional[AspectScore]  = Field(None, description="Aspect prediction (if SetFit is enabled).")
    flags:            PredictionFlags        = Field(..., description="Edge-case quality flags.")
    processing_time_ms: float               = Field(..., description="End-to-end inference time in milliseconds.")
    model_version:    str                    = Field(..., description="Version tag of the loaded model.")
    explanation:      Optional[dict]         = Field(None, description="SHAP explanation, if requested.")
    similar_reviews:  Optional[SearchResponseModel] = Field(None, description="Semantically similar historical reviews.")


class BatchPredictionResponse(BaseModel):
    predictions:     List[PredictionResult]
    total_processed: int


class BatchJobSubmission(BaseModel):
    """Returned immediately when a batch job is accepted."""
    job_id:   str = Field(..., description="Poll /predict/batch/{job_id} for results.")
    status:   str = Field(..., description="'queued' on submission.")
    item_count: int


class BatchJobStatus(BaseModel):
    """Returned when polling a batch job."""
    job_id:  str
    status:  str  = Field(..., description="queued | processing | done | failed")
    predictions: Optional[List[PredictionResult]] = None
    error:   Optional[str]                        = None


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_version:  str
    model_dir:      Optional[str]
    aspect_enabled: bool
    uptime_seconds: float
    timestamp:      str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SARCASM_PATTERNS = re.compile(
    r"(just (wonderful|perfect|great)|oh great|thanks? (a lot|so much)|wow+|'ll?)",
    re.IGNORECASE,
)

def _detect_flags(text: str, confidence: float) -> PredictionFlags:
    """
    Applies lightweight heuristics to flag edge-case predictions.

    Args:
        text (str):       Cleaned input text.
        confidence (float): Model confidence for the winning class.

    Returns:
        PredictionFlags
    """
    return PredictionFlags(
        short_review    = len(text.strip()) < 10,
        low_confidence  = confidence < 0.60,
        possible_sarcasm = bool(_SARCASM_PATTERNS.search(text)),
    )


def _row_to_result(row: dict, processing_time_ms: float) -> PredictionResult:
    """Convert a raw inference row dict into a PredictionResult."""
    scores  = row["scores"]
    aspect  = AspectScore(label=row["aspect"]) if row.get("aspect") else None
    flags   = _detect_flags(row["text"], row["confidence"])

    return PredictionResult(
        text=row["text"],
        label=row["label"],
        confidence=row["confidence"],
        scores=SentimentScores(**scores),
        aspect=aspect,
        flags=flags,
        processing_time_ms=round(processing_time_ms, 2),
        model_version=MODEL_VERSION,
    )


def _sync_run(title: str, text: Optional[str], batch_mode: bool):
    """Synchronous inference call — intended to run in a thread pool."""
    return pipeline.run(title=title, text=text, batch_mode=batch_mode)


async def _async_run(title: str, text: Optional[str], batch_mode: bool):
    """Offload synchronous inference to the thread pool so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_run, title, text, batch_mode)


def _ensure_pipeline():
    """Raise 503 if pipeline is unavailable; attempt lazy init first."""
    global pipeline
    if pipeline is None:
        _initialize_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline unavailable — check server logs.",
        )


searcher = None

def _initialize_searcher():
    global searcher
    if searcher is None:
        logger.info("Initializing SemanticSearcher lazily...")
        try:
            from src.search.searcher import SemanticSearcher
            searcher = SemanticSearcher()
        except FileNotFoundError as e:
            logger.warning(f"Search index not found. Semantic search unavailable. ({e})")
            searcher = "UNAVAILABLE"
        except Exception as e:
            logger.error(f"Failed to initialize SemanticSearcher: {e}")
            searcher = "UNAVAILABLE"


# --------------------------------------------------------------------------
# Background task for batch jobs
# --------------------------------------------------------------------------
def _process_batch_job(job_id: str, request: BatchPredictionRequest) -> None:
    """
    Runs synchronous batch inference for a queued job.
    Called in a BackgroundTask so it does not block the response.
    """
    global batch_jobs
    BATCH_JOBS_ACTIVE.inc()
    batch_jobs[job_id]["status"] = "processing"

    try:
        sep    = pipeline.batch_separator
        titles = sep.join(item.title for item in request.items)
        texts  = sep.join(item.text or "" for item in request.items)

        t0 = time.perf_counter()
        result_df = pipeline.run(title=titles, text=texts, batch_mode=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_item_ms = elapsed_ms / max(len(result_df), 1)
        predictions = [
            _row_to_result(row, per_item_ms)
            for row in result_df.to_dict("records")
        ]

        # Update Prometheus counters and log predictions
        for p, req_item in zip(predictions, request.items):
            PREDICTIONS_TOTAL.labels(endpoint="batch", label=p.label).inc()
            CONFIDENCE_DISTRIBUTION.observe(p.confidence)
            prediction_logger.log_prediction(
                text=p.text,
                label=p.label,
                confidence=p.confidence,
                model_version=p.model_version
            )
            
            # Local LLM Judge confidence check
            try:
                from src.utils.config_parser import load_config
                judge_cfg = load_config("configs/pipeline_params.yaml").get("llm_judge", {})
                lower = judge_cfg.get("confidence_window", {}).get("lower", 0.40)
                upper = judge_cfg.get("confidence_window", {}).get("upper", 0.60)
                q_db_path = judge_cfg.get("queue", {}).get("db_path", "artifacts/llm_judge/review_queue.db")
                
                if lower <= p.confidence <= upper:
                    from src.llm_judge.queue_manager import QueueManager
                    from src.llm_judge import QueueEntry
                    import uuid
                    from datetime import datetime, timezone
                    
                    qm = QueueManager(db_path=q_db_path)
                    entry = QueueEntry(
                        entry_id=str(uuid.uuid4()),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        input_text=p.text,
                        raw_title=req_item.title,
                        raw_text=req_item.text or "",
                        model_prediction=p.label,
                        model_confidence=p.confidence,
                        model_probabilities=p.scores.dict(),
                        model_version=p.model_version,
                        status="pending"
                    )
                    # For batch, we do it inline here since it's already in a BackgroundTask
                    qm.enqueue(entry)
            except Exception as e:
                logger.error(f"Failed LLM Judge queueing in batch: {e}")

        PREDICTION_LATENCY.labels(endpoint="batch").observe(elapsed_ms / 1000)
        batch_jobs[job_id]["status"] = "done"
        batch_jobs[job_id]["result"] = [p.model_dump() for p in predictions]

    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        PREDICTION_ERRORS.labels(endpoint="batch").inc()
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"]  = str(e)
    finally:
        BATCH_JOBS_ACTIVE.dec()


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Production health check used by Kubernetes liveness/readiness probes.
    Returns model status, version, and uptime.
    """
    return HealthResponse(
        status         = "healthy" if pipeline is not None else "unhealthy",
        model_loaded   = pipeline is not None,
        model_version  = MODEL_VERSION,
        model_dir      = getattr(pipeline, "model_dir", None),
        aspect_enabled = getattr(pipeline, "aspect_model", None) is not None,
        uptime_seconds = round(time.time() - startup_time, 1),
        timestamp      = datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics", tags=["Observability"], response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus exposition endpoint.
    Grafana scrapes this to populate dashboards in Phase 5.

    Exposed metrics:
        - sentiment_predictions_total (counter, by endpoint + label)
        - sentiment_prediction_errors_total (counter, by endpoint)
        - sentiment_prediction_latency_seconds (histogram, by endpoint)
        - sentiment_confidence_score (histogram)
        - sentiment_batch_jobs_active (gauge)
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResult, response_model_exclude_none=True, tags=["Inference"])
async def predict_single(request: SinglePredictionRequest, background_tasks: BackgroundTasks):
    """
    Async single-sample sentiment prediction.

    The model forward pass runs in a thread-pool executor so the asyncio
    event loop is never blocked, allowing concurrent requests to proceed.
    """
    _ensure_pipeline()
    t0 = time.perf_counter()
    try:
        result_df = await _async_run(
            title=request.title,
            text=request.text,
            batch_mode=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        row = result_df.to_dict("records")[0]
        result = _row_to_result(row, elapsed_ms)

        # Handle optional SHAP explanation
        if request.include_explanation:
            _initialize_explainer()
            if explainer is not None:
                try:
                    combined_text = pipeline._build_texts(request.title, request.text, batch_mode=False)[0]
                    result.explanation = explainer.explain(combined_text, target_class=result.label)
                except Exception as e:
                    logger.warning(f"Failed to generate SHAP explanation: {e}")

        PREDICTIONS_TOTAL.labels(endpoint="single", label=result.label).inc()
        CONFIDENCE_DISTRIBUTION.observe(result.confidence)
        PREDICTION_LATENCY.labels(endpoint="single").observe(elapsed_ms / 1000)

        # Optional Semantic Search Enrichment
        if request.include_similar_reviews:
            _initialize_searcher()
            if searcher is not None and searcher != "UNAVAILABLE":
                try:
                    search_resp = searcher.search(
                        query_text=result.text, 
                        model_prediction=result.label, 
                        top_k=request.similar_reviews_count
                    )
                    result.similar_reviews = SearchResponseModel(
                        results=[
                            SearchResultItem(
                                rank=r.rank,
                                similarity_score=r.similarity_score,
                                clean_text=r.clean_text,
                                raw_title=r.raw_title,
                                raw_text=r.raw_text,
                                known_sentiment=r.known_sentiment,
                                rating=r.rating,
                                sentiment_alignment=r.sentiment_alignment
                            ) for r in search_resp.results
                        ],
                        query_text=search_resp.query_text,
                        top_k=search_resp.top_k,
                        alignment_rate=search_resp.alignment_rate,
                        alignment_signal=search_resp.alignment_signal,
                        search_latency_ms=search_resp.search_latency_ms
                    )
                except Exception as e:
                    logger.warning(f"Semantic search failed: {e}")

        # Non-blocking prediction log write
        background_tasks.add_task(
            prediction_logger.log_prediction,
            text=result.text,
            label=result.label,
            confidence=result.confidence,
            model_version=result.model_version
        )
        
        # Local LLM Judge confidence check
        try:
            from src.utils.config_parser import load_config
            judge_cfg = load_config("configs/pipeline_params.yaml").get("llm_judge", {})
            lower = judge_cfg.get("confidence_window", {}).get("lower", 0.40)
            upper = judge_cfg.get("confidence_window", {}).get("upper", 0.60)
            q_db_path = judge_cfg.get("queue", {}).get("db_path", "artifacts/llm_judge/review_queue.db")
            
            if lower <= result.confidence <= upper:
                from src.llm_judge.queue_manager import QueueManager
                from src.llm_judge import QueueEntry
                import uuid
                from datetime import datetime, timezone
                
                qm = QueueManager(db_path=q_db_path)
                entry = QueueEntry(
                    entry_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    input_text=result.text,
                    raw_title=request.title,
                    raw_text=request.text or "",
                    model_prediction=result.label,
                    model_confidence=result.confidence,
                    model_probabilities=result.scores.dict(),
                    model_version=result.model_version,
                    status="pending"
                )
                background_tasks.add_task(qm.enqueue, entry)
        except Exception as e:
            logger.error(f"Failed LLM Judge queueing in single predict: {e}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.labels(endpoint="single").inc()
        logger.error(f"Single prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchJobSubmission, status_code=202, tags=["Inference"])
async def submit_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch of reviews for async sentiment prediction.

    Returns a job_id immediately (HTTP 202). Poll
    GET /predict/batch/{job_id} to retrieve results when the job is done.
    """
    _ensure_pipeline()

    job_id = str(uuid.uuid4())
    batch_jobs[job_id] = {"status": "queued", "result": None, "error": None}
    background_tasks.add_task(_process_batch_job, job_id, request)

    logger.info(f"Batch job {job_id} queued — {len(request.items)} item(s).")
    return BatchJobSubmission(
        job_id=job_id,
        status="queued",
        item_count=len(request.items),
    )


@app.get("/predict/batch/{job_id}", response_model=BatchJobStatus, tags=["Inference"])
async def get_batch_result(job_id: str):
    """
    Poll the status and results of a submitted batch job.

    Status values:
        - queued     — job accepted, not yet started
        - processing — inference in progress
        - done       — results available in the 'predictions' field
        - failed     — an error occurred; check the 'error' field
    """
    job = batch_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return BatchJobStatus(
        job_id=job_id,
        status=job["status"],
        predictions=job["result"],
        error=job["error"],
    )


@app.get("/test", tags=["Health"])
async def test_endpoint():
    """Lightweight reachability check."""
    return {"message": "API is running.", "timestamp": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.api.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )