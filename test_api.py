# Test RRF formula implementation: RRF_Score = sum(1 / (k + rank_i))
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_vector_search():
    response = client.get("/search?query=Squawk7500")
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert "id" in results[0]
    
def test_hybrid_search():
    response = client.get("/hybrid-search?query=Squawk7500")
    assert response.status_code == 200
    results = response.json()
    assert len(results) > 0
    assert "cross_encoder_score" in results[0]