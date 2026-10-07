import lancedb
import json
import pandas

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from lancedb.rerankers import RRFReranker

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
async def search_incident(query: str):
    """
    Gets a text input and returns results from vector search
    """
    db = await connect_vector_db(uri)
    incident_table = await db.open_table("flight_incident")
    query_builder = await incident_table.search(
        query, query_type="hybrid", vector_column_name="vector", fts_columns="text"
    )

    reranker = RRFReranker()

    results = await query_builder.rerank(reranker).limit(3).to_list()
    return results
