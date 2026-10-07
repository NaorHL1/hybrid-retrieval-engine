from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from sentence_transformers import SentenceTransformer

model = get_registry().get("sentence-transformers").create(name="BAAI/bge-small-en-v1.5", device="cpu")

class FlightDocument(LanceModel):
    id:str
    title:str
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()


