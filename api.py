import lancedb
import json
import pandas
import math

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from lancedb.rerankers import RRFReranker, CrossEncoderReranker

from data_models import FlightDocument

app = FastAPI()

uri = "vector_database"

cross_wrapper = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

async def connect_vector_db(path):
    vector_db = await lancedb.connect_async(uri=path)
    return vector_db


async def _open_table():
    db = await connect_vector_db(uri)
    return await db.open_table("flight_incident")


@app.get("/search")
async def search_incident(query: str, top_k: int = 5):
    """
    Pure dense (vector) search. top_k is configurable so the eval can measure MRR / Recall@k.
    """
    incident_table = await _open_table()
    query_builder = await incident_table.search(query)
    return await query_builder.limit(top_k).to_list()


async def rrf_candidates(query: str, n: int = 1000):
    """
    Stage 1: dense + BM25 fused with RRF. Returns candidates ordered by RRF score.
    """
    incident_table = await _open_table()
    query_builder = await incident_table.search(
        query, query_type="hybrid", vector_column_name="vector", fts_columns="text"
    )
    return await (
        query_builder.rerank(RRFReranker())
        .select(["id", "title", "text"])
        .limit(n)
        .to_list()
    )


def cross_encode(query: str, docs: list[dict]) -> list[dict]:
    """
    Stage 2: score (query, doc) pairs jointly with the cross-encoder, sorted best-first.
    """
    pairs = [[query, f"{doc['title']} - {doc['text']}"] for doc in docs]
    scores = cross_wrapper.model.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["cross_encoder_score"] = 1 / (1 + math.exp(-float(score)))
    return sorted(docs, key=lambda d: d["cross_encoder_score"], reverse=True)


@app.get("/hybrid-search")
async def hybrid_search_incident(query: str, top_k: int = 5, rerank_top_n: int = 50):
    """
    Hybrid pipeline: RRF(dense + BM25) -> cross-encoder on the top `rerank_top_n` -> top_k.
    """
    results = await rrf_candidates(query)
    if not results:
        return []
    return cross_encode(query, results[:rerank_top_n])[:top_k]
