import json
import re
from typing import Optional, Dict, Any
from src.llm_judge import QueueEntry
from src.utils.logger import logger

class PromptBuilder:
    SYSTEM_PROMPT = """You are a sentiment analysis expert reviewing product reviews. Your task is to 
determine the sentiment of a customer review as either "positive", "neutral", or "negative".

You must respond with ONLY a JSON object in this exact format, with no additional text:
{"sentiment": "<positive|neutral|negative>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}

Definitions:
- positive: the reviewer is satisfied with the product overall
- neutral: the reviewer has mixed feelings or is neither satisfied nor dissatisfied  
- negative: the reviewer is dissatisfied with the product overall"""

    @classmethod
    def build_judgment_prompt(cls, entry: QueueEntry) -> str:
        return f"""Review Title: {entry.raw_title}
Review Text: {entry.raw_text}

A sentiment classifier predicted: {entry.model_prediction} (confidence: {entry.model_confidence:.2f})

The classifier's full probability distribution:
- positive: {entry.model_probabilities.get('positive', 0.0):.3f}
- neutral: {entry.model_probabilities.get('neutral', 0.0):.3f}  
- negative: {entry.model_probabilities.get('negative', 0.0):.3f}

This review was flagged for secondary review because the classifier was uncertain.
Please provide your independent assessment."""

    @classmethod
    def parse_response(cls, raw_output: str) -> Optional[Dict[str, Any]]:
        if not raw_output:
            return None
            
        raw_output = raw_output.strip()
        
        # 1st tier: direct JSON parse
        try:
            parsed = json.loads(raw_output)
            if "sentiment" in parsed and "confidence" in parsed and "reasoning" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
            
        # 2nd tier: Regex fallback for Markdown or preambles
        try:
            match = re.search(r'\{[^}]+\}', raw_output)
            if match:
                parsed = json.loads(match.group(0))
                if "sentiment" in parsed and "confidence" in parsed and "reasoning" in parsed:
                    return parsed
        except Exception:
            pass
            
        # 3rd tier: Graceful failure
        logger.warning(f"Failed to parse judge output: {raw_output}")
        return None
