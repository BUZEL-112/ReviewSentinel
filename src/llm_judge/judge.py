import time
import requests
from typing import List, Dict, Any
from src.llm_judge import QueueEntry, JudgmentResult
from src.llm_judge.prompt_builder import PromptBuilder
from src.utils.logger import logger

class LLMJudge:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", model_name: str = "mistral", config: Dict[str, Any] = None):
        self.config = config or {}
        
        ollama_cfg = self.config.get("ollama", {})
        self.ollama_base_url = ollama_cfg.get("base_url", ollama_base_url)
        self.model_name = ollama_cfg.get("model_name", model_name)
        self.temperature = ollama_cfg.get("temperature", 0.1)
        self.timeout = ollama_cfg.get("timeout_seconds", 30)

    def _check_ollama_health(self) -> bool:
        try:
            resp = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                if any(self.model_name in m for m in models):
                    return True
                else:
                    logger.warning(f"Ollama is reachable, but model '{self.model_name}' is not found.")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.ollama_base_url}: {e}")
            return False

    def _call_ollama(self, prompt: str, system_prompt: str) -> str | None:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/chat", 
                json=payload, 
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama inference call failed: {e}")
            return None

    def judge(self, entry: QueueEntry) -> JudgmentResult:
        prompt = PromptBuilder.build_judgment_prompt(entry)
        
        t0 = time.perf_counter()
        raw_response = self._call_ollama(prompt, PromptBuilder.SYSTEM_PROMPT)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if raw_response is None:
            return JudgmentResult(
                entry_id=entry.entry_id,
                judge_prediction=None, judge_confidence=None, judge_reasoning=None,
                is_conflict=None, raw_response="", parse_success=False, latency_ms=latency_ms
            )
            
        parsed = PromptBuilder.parse_response(raw_response)
        
        if not parsed:
            return JudgmentResult(
                entry_id=entry.entry_id,
                judge_prediction=None, judge_confidence=None, judge_reasoning=None,
                is_conflict=None, raw_response=raw_response, parse_success=False, latency_ms=latency_ms
            )
            
        judge_pred = parsed["sentiment"].lower()
        is_conflict = (judge_pred != entry.model_prediction)
        
        return JudgmentResult(
            entry_id=entry.entry_id,
            judge_prediction=judge_pred,
            judge_confidence=parsed["confidence"],
            judge_reasoning=parsed["reasoning"],
            is_conflict=is_conflict,
            raw_response=raw_response,
            parse_success=True,
            latency_ms=latency_ms
        )

    def judge_batch(self, entries: List[QueueEntry]) -> List[JudgmentResult]:
        # Process sequentially to avoid out-of-memory errors on Ollama with local GPUs
        results = []
        for entry in entries:
            result = self.judge(entry)
            results.append(result)
        return results
