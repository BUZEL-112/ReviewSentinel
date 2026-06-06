import logging
from typing import List
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading SentenceTransformer '{self.model_name}' on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("SentenceTransformer loaded.")

    def get_embedding_dim(self) -> int:
        """Returns the dimensionality of the embeddings produced by the model."""
        dummy = self.model.encode(["dummy string"], normalize_embeddings=True)
        return dummy.shape[1]

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encodes a list of strings into a numpy array of embeddings.
        Normalizes to unit length by default for cosine similarity via inner product.
        """
        if not texts:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)
            
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size, 
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    def encode_for_index(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        """
        Specialized method for bulk encoding during index building.
        Handles batching internally and returns the full float32 matrix.
        """
        texts = df[text_column].tolist()
        logger.info(f"Encoding {len(texts)} documents for FAISS index...")
        
        # We can use the model's builtin batching & progress bar or do it manually to log.
        # The sentence-transformers encode method already batches internally, 
        # so we just call it directly.
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        logger.info("Encoding complete.")
        return embeddings.astype(np.float32)
