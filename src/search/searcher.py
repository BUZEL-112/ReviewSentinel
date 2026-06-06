import time
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.search.encoder import SentenceEncoder
from src.search.indexer import FAISSIndexer

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    review_id: int
    rank: int
    similarity_score: float
    clean_text: str
    raw_title: str
    raw_text: str
    known_sentiment: str
    rating: float
    sentiment_alignment: Optional[bool]


@dataclass
class SearchResponse:
    results: List[SearchResult]
    query_text: str
    top_k: int
    alignment_rate: Optional[float]
    alignment_signal: Optional[str]
    search_latency_ms: float


class SemanticSearcher:
    """
    Provides the query interface for FAISS semantic search.
    Holds the loaded index and metadata in memory.
    """
    def __init__(self, index_dir: str = "artifacts/search/", encoder: Optional[SentenceEncoder] = None, default_top_k: int = 5, self_match_threshold: float = 0.999):
        self.indexer = FAISSIndexer(index_dir=index_dir)
        self.encoder = encoder or SentenceEncoder()
        self.default_top_k = default_top_k
        self.self_match_threshold = self_match_threshold
        
        logger.info(f"Loading FAISS index from {index_dir}...")
        self.index, self.metadata, self.encoder_info = self.indexer.load()
        self.indexer.validate_index(self.index, self.encoder_info, self.encoder.model_name)
        logger.info("Searcher initialized successfully.")

    def search(self, query_text: str, model_prediction: Optional[str] = None, top_k: Optional[int] = None) -> SearchResponse:
        t0 = time.perf_counter()
        k = top_k or self.default_top_k
        
        # 1. Encode query
        query_vector = self.encoder.encode([query_text], normalize=True)
        
        # 2. Search index (with buffer for self-matches)
        buffer_k = k + 2
        scores, ids = self.index.search(query_vector, buffer_k)
        
        # 3. Filter and build results
        results = []
        rank = 1
        
        for i in range(len(ids[0])):
            review_id = ids[0][i]
            score = scores[0][i]
            
            if review_id == -1:
                continue  # not enough results
                
            if score > self.self_match_threshold:
                continue  # self-match
                
            if len(results) >= k:
                break
                
            meta = self.metadata.get(review_id, {})
            known_sentiment = meta.get("sentiment", "")
            
            alignment = None
            if model_prediction is not None and known_sentiment:
                alignment = (known_sentiment.lower() == model_prediction.lower())
                
            results.append(SearchResult(
                review_id=int(review_id),
                rank=rank,
                similarity_score=float(score),
                clean_text=meta.get("clean_text", ""),
                raw_title=meta.get("raw_title", ""),
                raw_text=meta.get("raw_text", ""),
                known_sentiment=known_sentiment,
                rating=float(meta.get("rating", 0.0)),
                sentiment_alignment=alignment
            ))
            rank += 1
            
        # 4. Compute alignment signal
        alignment_rate = None
        alignment_signal = None
        
        if model_prediction is not None and len(results) > 0:
            alignments = [r.sentiment_alignment for r in results if r.sentiment_alignment is not None]
            if alignments:
                alignment_rate = sum(alignments) / len(alignments)
                if alignment_rate >= 0.8:
                    alignment_signal = "STRONG_SUPPORT"
                elif alignment_rate < 0.4:
                    alignment_signal = "CONTRADICTS"
                else:
                    alignment_signal = "MIXED"
                    
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return SearchResponse(
            results=results,
            query_text=query_text,
            top_k=len(results),
            alignment_rate=alignment_rate,
            alignment_signal=alignment_signal,
            search_latency_ms=latency_ms
        )
