import os
import json
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import faiss

from src.search.encoder import SentenceEncoder

logger = logging.getLogger(__name__)

class FAISSIndexer:
    """
    Handles the creation, persistence, and loading of the FAISS index
    and the associated metadata store (Parquet).
    """
    def __init__(self, index_dir: str = "artifacts/search/"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss.index"
        self.metadata_path = self.index_dir / "metadata.parquet"
        self.encoder_info_path = self.index_dir / "encoder_info.json"

    def build(self, df: pd.DataFrame, encoder: SentenceEncoder, text_column: str = "clean_text"):
        """
        Builds the FAISS index from the dataframe using the provided encoder.
        Saves the index, metadata parquet, and encoder_info to disk.
        """
        logger.info("Starting FAISS index build process...")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Encode texts
        embeddings = encoder.encode_for_index(df, text_column)
        
        # 2. Verify float32
        assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings"
        
        # 3. Create integer IDs
        n_reviews = len(df)
        ids = np.arange(n_reviews, dtype=np.int64)
        
        # 4. Initialize FAISS index
        embedding_dim = encoder.get_embedding_dim()
        logger.info(f"Initializing IndexFlatIP with dimension {embedding_dim}...")
        base_index = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIDMap(base_index)
        
        # 5. Add vectors
        index.add_with_ids(embeddings, ids)
        logger.info(f"Added {index.ntotal} vectors to index.")
        
        # 6. Save FAISS index
        faiss.write_index(index, str(self.index_path))
        logger.info(f"Saved FAISS index to {self.index_path}")
        
        # 7. Prepare and save metadata
        metadata_df = df.copy()
        metadata_df['review_id'] = ids
        
        # Ensure we only keep columns we might need for display/search
        keep_cols = ['review_id', 'clean_text']
        if 'raw_title' in metadata_df.columns: keep_cols.append('raw_title')
        if 'raw_text' in metadata_df.columns: keep_cols.append('raw_text')
        if 'title' in metadata_df.columns and 'raw_title' not in keep_cols:
            metadata_df = metadata_df.rename(columns={'title': 'raw_title'})
            keep_cols.append('raw_title')
        if 'text' in metadata_df.columns and 'raw_text' not in keep_cols:
            metadata_df = metadata_df.rename(columns={'text': 'raw_text'})
            keep_cols.append('raw_text')
        if 'sentiment' in metadata_df.columns: keep_cols.append('sentiment')
        if 'label' in metadata_df.columns and 'sentiment' not in keep_cols:
            metadata_df = metadata_df.rename(columns={'label': 'sentiment'})
            keep_cols.append('sentiment')
        if 'rating' in metadata_df.columns: keep_cols.append('rating')
            
        # Ensure these columns actually exist
        keep_cols = [c for c in keep_cols if c in metadata_df.columns]
        
        metadata_df[keep_cols].to_parquet(self.metadata_path, index=False)
        logger.info(f"Saved metadata to {self.metadata_path}")
        
        # 8. Save encoder info
        import datetime
        encoder_info = {
            "model_name": encoder.model_name,
            "embedding_dim": embedding_dim,
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "corpus_size": n_reviews
        }
        with open(self.encoder_info_path, 'w') as f:
            json.dump(encoder_info, f, indent=2)
        logger.info(f"Saved encoder info to {self.encoder_info_path}")

    def load(self) -> Tuple[faiss.Index, Dict[int, dict], dict]:
        """
        Loads the FAISS index, metadata, and encoder info from disk.
        Returns the index, a dictionary mapping review_id -> metadata_row, and encoder_info.
        """
        if not self.index_path.exists() or not self.metadata_path.exists() or not self.encoder_info_path.exists():
            raise FileNotFoundError(f"One or more index files missing in {self.index_dir}")
            
        with open(self.encoder_info_path, 'r') as f:
            encoder_info = json.load(f)
            
        index = faiss.read_index(str(self.index_path))
        
        metadata_df = pd.read_parquet(self.metadata_path)
        # Convert to O(1) lookup dict: {review_id: {col: val, ...}}
        metadata_dict = metadata_df.set_index('review_id').to_dict('index')
        
        logger.info(f"Loaded index with {index.ntotal} vectors from {self.index_dir}")
        return index, metadata_dict, encoder_info

    def validate_index(self, index: faiss.Index, encoder_info: dict, current_model_name: str):
        """
        Validates that the loaded index was built using the currently configured encoder model.
        """
        built_model = encoder_info.get("model_name")
        if built_model != current_model_name:
            msg = (f"Index was built with {built_model} but current encoder is {current_model_name}. "
                   "Rebuild the index with `python scripts/build_search_index.py`.")
            logger.error(msg)
            raise ValueError(msg)
