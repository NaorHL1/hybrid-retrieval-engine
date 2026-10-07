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


@app.get("/search")
async def search_incident(query: str):
    """
    Gets a text input and returns results from vector search
    """
    db = await connect_vector_db(uri)
    incident_table = await db.open_table("flight_incident")
    query_builder = await incident_table.search(query)
    results = await query_builder.limit(5).to_list()
    return results


@app.get("/hybrid-search")
async def hybrid_search_incident(query: str):
    """
    Gets a text input and returns results from vector search
    """
    db = await connect_vector_db(uri)
    incident_table = await db.open_table("flight_incident")

    rrf_reranker = RRFReranker()
    query_builder = await incident_table.search(
        query, query_type="hybrid", vector_column_name="vector", fts_columns="text"
    )

    results = await query_builder.rerank(rrf_reranker).select(["id", "title", "text"]).limit(1000).to_list()

    if not results:
        return []
    
    top_50_results = results[:50]

    pairs = [[query, f"{doc['title']} - {doc['text']}"] for doc in top_50_results]
    scores = cross_wrapper.model.predict(pairs)

    for doc, score in zip(top_50_results, scores):
        probability = 1 / (1 + math.exp(-float(score)))
        doc["cross_encoder_score"] = probability

    final_top_5 = sorted(top_50_results, key=lambda x: x["cross_encoder_score"], reverse=True)[:5]

    return final_top_5
