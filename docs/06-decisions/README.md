# Architecture Decision Records (ADRs)

This directory contains the historical design decisions made for ReviewSentinel. We use ADRs to document *why* a particular technology or pattern was chosen, especially when it involves significant trade-offs.

## Decision Map: Which decisions affect me?

| Persona | Relevant ADRs |
|---------|---------------|
| **DevOps / Infra** | [003 (SQLite Judge Queue)](003-sqlite-for-llm-judge-queue.md), [005 (MinIO over S3)](005-minio-over-s3.md), [006 (Kind for Local K8s)](006-kind-for-local-k8s.md) |
| **ML Engineer** | [001 (DistilBERT over Larger Models)](001-distilbert-over-larger-models.md), [004 (FAISS Flat Index)](004-faiss-flat-index-over-hnsw.md), [007 (Three-Tier Prompt Parsing)](007-three-tier-prompt-parsing.md) |
| **API Consumer** | [002 (Prefect over Alternatives)](002-prefect-over-alternatives.md), [005 (MinIO over S3)](005-minio-over-s3.md) |

## Full Index
- **001:** [DistilBERT over Larger Models](001-distilbert-over-larger-models.md)
- **002:** [Prefect over Alternatives](002-prefect-over-alternatives.md)
- **003:** [SQLite for LLM Judge Queue](003-sqlite-for-llm-judge-queue.md)
- **004:** [FAISS Flat Index over HNSW](004-faiss-flat-index-over-hnsw.md)
- **005:** [MinIO over S3](005-minio-over-s3.md)
- **006:** [Kind for Local K8s](006-kind-for-local-k8s.md)
- **007:** [Three-Tier Prompt Parsing](007-three-tier-prompt-parsing.md)
- **008:** [Documentation Structure](008-documentation-structure.md)
