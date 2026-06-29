# ADR 004: FAISS Flat Index over HNSW

**Date:** June 2026  
**Status:** Accepted

## Context

ReviewSentinel's semantic search feature (`include_similar_reviews: true`) requires retrieving the $K$ most similar historical reviews to an incoming query. 

We generate embeddings for reviews using a sentence-transformer model and must search these embeddings rapidly. 

We evaluated two FAISS index types:
1. **IndexFlatIP (Exact Search):** Computes the inner product between the query vector and *every* vector in the dataset.
2. **HNSW (Approximate Nearest Neighbour):** Builds a hierarchical graph to navigate to the nearest neighbours logarithmically, sacrificing perfect accuracy for speed.

## Decision

We chose **IndexFlatIP** (exact search).

## Rationale

1. **Dataset Scale:** Our current training dataset (`All_Beauty`) is relatively small (tens of thousands of rows). At this scale, a brute-force linear scan of high-dimensional vectors takes less than 10ms on a modern CPU.
2. **Accuracy:** IndexFlatIP guarantees 100% recall. The alignment signal we compute relies on finding the *true* nearest neighbours. HNSW approximation errors could artificially skew the alignment signal.
3. **Memory and Build Time:** HNSW indices require significantly more RAM to hold the graph structure and take longer to build. IndexFlatIP requires exactly $N \times D \times 4$ bytes of memory and builds instantaneously.
4. **Simplicity:** IndexFlatIP requires no hyperparameter tuning (unlike HNSW, which requires tuning `M` and `efConstruction`).

## Consequences

- **Positive:** We get perfectly accurate semantic similarity scores with zero parameter tuning.
- **Negative:** Search time scales linearly $O(N)$ with the size of the dataset. If the training corpus grows beyond ~500,000 records, the 10ms latency budget will be blown, and API latency will degrade unacceptably. At that point, this decision must be revisited and the system migrated to HNSW or an inverted file index (IVF).
