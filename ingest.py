import lancedb
import json
import pandas

from data_models import FlightDocument

def setup_vector_db(path):
  vector_db = lancedb.connect(uri = path)
  vector_db.create_table("flight_incident", schema=FlightDocument, mode="overwrite")
  return vector_db

def ingest_data_to_vector_db(table):
    with open("data/flydubai_flight_1073_Incident.json", "r") as file:
        data = json.loads(file.read())
    table.add(data)

def add_data_to_vector_db(table):
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
   table.add(more_data)

def create_fts_vector_db(table):
   table.create_fts_index("text")

if __name__ == "__main__":
   uri = "vector_database"
   vector_db = setup_vector_db(uri)
   ingest_data_to_vector_db(vector_db["flight_incident"])
   create_fts_vector_db(vector_db["flight_incident"])