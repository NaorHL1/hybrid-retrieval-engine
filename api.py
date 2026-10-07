import lancedb
import json
import pandas

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from lancedb.rerankers import RRFReranker
from lancedb.rerankers import CrossEncoderReranker

from data_models import FlightDocument

app = FastAPI()

uri = "vector_database"


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
    query_builder = await incident_table.search(
        query, query_type="hybrid", vector_column_name="vector", fts_columns="text"
    )

    rrf_reranker = RRFReranker()
    cross_encoder = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")

    results = query_builder.rerank(rrf_reranker).limit(1000)
    top_50_results = await results.rerank(cross_encoder).select(["id", "title", "text"]).limit(50).to_list()

    if not top_50_results:
        return []

    return top_50_results
