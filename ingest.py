import lancedb
import json
import pandas
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from sentence_transformers import SentenceTransformer


uri = "vector_database"
db = lancedb.connect(uri)

model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class FlightDocument(LanceModel):
    id:str
    title:str
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


# Create A Tabler

with open("data/flydubai_flight_1073_Incident.json", "r") as file:
    data = json.loads(file.read())

incident_table = db.create_table("flight_incident", schema=FlightDocument, mode="overwrite")

incident_table.add(data)

more_data = [
        {
    "id": "doc_006",
    "title": "Flight Data Recorder (Black Box) Telemetry Analysis",
    "text": "Analysis of the Digital Flight Data Recorder (DFDR) retrieved from the Boeing 737 MAX indicates deliberate manual disengagement of the autopilot system at FL360. Severe control column inputs were logged immediately prior to the steep dive, confirming intentional pitch-down commands rather than an automated trim malfunction or MCAS failure."
},
{
    "id": "doc_007",
    "title": "Diplomatic & Extradition Memorandum",
    "text": "The Ministry of Foreign Affairs confirmed that bilateral discussions between the UAE and Saudi Arabian legal authorities are underway regarding jurisdiction and extradition. Given that the hijacking attempt occurred in international airspace before diverting to Tabuk, both nations are coordinating under the Tokyo Convention of 1963 regarding offenses committed on board aircraft."
}
]

incident_table.add(more_data)

incident_table.to_pandas()


incident_table.create_fts_index("text")
print(incident_table.to_pandas().head())