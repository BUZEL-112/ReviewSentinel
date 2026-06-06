from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class QueueEntry:
    entry_id: str
    timestamp: str
    input_text: str
    raw_title: str
    raw_text: str
    model_prediction: str
    model_confidence: float
    model_probabilities: Dict[str, float]
    model_version: str
    status: str
    judge_prediction: Optional[str] = None
    judge_reasoning: Optional[str] = None
    is_conflict: Optional[bool] = None

@dataclass
class JudgmentResult:
    entry_id: str
    judge_prediction: Optional[str]
    judge_confidence: Optional[float]
    judge_reasoning: Optional[str]
    is_conflict: Optional[bool]
    raw_response: str
    parse_success: bool
    latency_ms: float
