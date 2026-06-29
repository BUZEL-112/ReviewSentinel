# System Design

This is not a general-purpose solution. It's optimized for:
- Single-node deployment
- ≤ 50k training reviews
- ≤ 100 predictions/sec
- 4-hour LLM judge SLA
- Weekly retraining

## Known Bottlenecks
1. **FAISS flat index:** O(n) rebuild on deploy. The index is fully rebuilt rather than incrementally updated.
2. **SQLite single-writer queue:** The LLM judge queue uses SQLite, which locks for writes and limits concurrent judge processing.
3. **Ollama sequential LLM inference:** The local LLM judge processes evaluation requests sequentially.

## Scaling Paths
To move past these limitations, refer to the rationale and paths out lined in:
- [ADR 004: FAISS flat index over HNSW](../06-decisions/004-faiss-flat-index-over-hnsw.md)
- [Deployment FAQ](../07-faq/deployment.md)
