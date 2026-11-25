"""Pure retrieval-stage functions, shared by the API and the eval harness.

Step 1a: only the manual RRF lives here so far. It mirrors
lancedb.rerankers.RRFReranker (score = sum of 1 / (rank + K), rank from 1).
It is used for per-stage tracing in eval, NOT in the production path.
"""

import math

import lancedb
from lancedb.rerankers import CrossEncoderReranker, RRFReranker

RRF_K = 60  # same default as lancedb RRFReranker
DB_URI = "vector_database"
TABLE_NAME = "flight_incident"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIELDS = ["id", "title", "text"]

_cross_encoder = None


def open_table(uri: str = DB_URI):
    """Open the table once; callers pass it to every stage function."""
    return lancedb.connect(uri).open_table(TABLE_NAME)


def dense(table, query: str, k: int) -> list[dict]:
    """Vector search only."""
    return table.search(query, query_type="vector").select(FIELDS).limit(k).to_list()


def bm25(table, query: str, k: int) -> list[dict]:
    """Full-text (BM25) search only. Requires the FTS index on `text`."""
    return table.search(query, query_type="fts").select(FIELDS).limit(k).to_list()


def hybrid_rrf(table, query: str, k: int) -> list[dict]:
    """Production path: LanceDB hybrid search fused with its own RRFReranker."""
    builder = table.search(
        query, query_type="hybrid", vector_column_name="vector", fts_columns="text"
    )
    return builder.rerank(RRFReranker()).select(FIELDS).limit(k).to_list()


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder rerank. Adds `cross_encoder_score` (sigmoid, 0..1) to each doc."""
    global _cross_encoder
    if not candidates:
        return []
    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker(model_name=CROSS_ENCODER_NAME).model
    pairs = [[query, f"{d['title']} - {d['text']}"] for d in candidates]
    scores = _cross_encoder.predict(pairs)
    scored = []
    for doc, score in zip(candidates, scores):
        scored.append({**doc, "cross_encoder_score": 1 / (1 + math.exp(-float(score)))})
    scored.sort(key=lambda d: d["cross_encoder_score"], reverse=True)
    return scored[:top_k]


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse several best-first id lists into one list of (id, score), best first.

    Ties keep first-appearance order (a stable sort), so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
