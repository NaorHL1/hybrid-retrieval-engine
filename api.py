import lancedb
import json

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from data_models import FlightDocument

app = FastAPI()

uri = "vector_database"

async def connect_vector_db(path):
    vector_db = await lancedb.connect_async(uri = path)
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
