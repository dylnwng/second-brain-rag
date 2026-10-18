---
tags: [career, projects]
created: 2026-08-15
---

# RAG Project

A retrieval-augmented Q&A system over my notes, built on [[google-cloud|Google Cloud]].

## Goals

Ship a measured, honest evaluation. Recall@5 matters more than demo polish.

## Stack

### Storage

Postgres with pgvector, HNSW index, cosine distance.

### Models

Gemini for generation and Gemini embeddings for retrieval.

```python
# not a heading, just a comment inside a code fence
print("hello")
```

## Empty Section

## Open Questions

- Should chunks overlap across headings? No.
- How big should chunks be? About 500 tokens.
