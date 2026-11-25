"""API tests. Quality tests use the real index; concurrency tests patch the cross-encoder with
a deterministic slow stub so they test the event loop and semaphore, not the machine's speed."""
import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import api

pytestmark = pytest.mark.integration  # needs vector_database/ and the downloaded models

client = TestClient(api.app)


def test_vector_search_contract():
    results = client.get("/search?query=A6-FKF").json()
    assert 0 < len(results) <= api.TOP_K
    assert {"id", "title", "text"} <= set(results[0])
    assert "vector" not in results[0]  # payload: no 384-float embedding per hit


def test_hybrid_search_beats_dense_on_a_lookalike_identifier():
    # With look-alike distractors (A6-FKE/FKG/FKH) in the corpus, dense search alone ranks a
    # look-alike first for "A6-FKF" (see eval/FINDINGS.md). The full pipeline must not.
    hybrid_ids = [r["id"] for r in client.get("/hybrid-search?query=A6-FKF").json()]
    assert hybrid_ids[0] == "doc_0003"


def test_hybrid_search_returns_sorted_scores():
    results = client.get("/hybrid-search?query=Squawk 7500 and 7700 emergency broadcast").json()
    scores = [r["cross_encoder_score"] for r in results]
    assert 0 < len(results) <= api.TOP_K
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert "doc_0008" in [r["id"] for r in results[:3]]


def test_hybrid_search_dentist_query_hits_either_valid_doc():
    # doc_0010 and doc_0142 both say a dentist treated the pilot (see eval/FINDINGS.md).
    q = "Who was the medical professional that treated the wounded pilot?"
    ids = [r["id"] for r in client.get("/hybrid-search", params={"query": q}).json()[:3]]
    assert {"doc_0010", "doc_0142"} & set(ids)


async def _run(requests_):
    """Fire (name, url, delay) requests against the ASGI app in one event loop."""
    done = {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t") as c:
        async def call(name, url, delay):
            await asyncio.sleep(delay)
            r = await c.get(url)
            assert r.status_code == 200
            done[name] = time.perf_counter()
        await asyncio.gather(*(call(*r) for r in requests_))
    return done


def test_slow_rerank_does_not_block_the_event_loop(monkeypatch):
    client.get("/hybrid-search?query=warm")  # load models before timing anything

    def slow_rerank(query, candidates, top_k):
        time.sleep(0.8)
        return candidates[:top_k]

    monkeypatch.setattr(api, "rerank", slow_rerank)
    done = asyncio.run(_run([
        ("hybrid", "/hybrid-search?query=Shin Bet", 0.0),
        ("search", "/search?query=A6-FKF", 0.4),  # arrives while the rerank is sleeping
    ]))
    assert done["search"] < done["hybrid"]


def test_rerank_calls_are_limited_by_the_semaphore(monkeypatch):
    client.get("/hybrid-search?query=warm")
    lock, state = threading.Lock(), {"now": 0, "max": 0}

    def tracked_rerank(query, candidates, top_k):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.3)
        with lock:
            state["now"] -= 1
        return candidates[:top_k]

    monkeypatch.setattr(api, "rerank", tracked_rerank)
    asyncio.run(_run([(f"h{i}", f"/hybrid-search?query=Shin Bet {i}", 0.0) for i in range(4)]))
    assert state["max"] == api.CE_CONCURRENCY == 1


def test_table_is_opened_once():
    first = api.get_table()
    client.get("/search?query=A6-FKF")
    client.get("/hybrid-search?query=A6-FKF")
    assert api.get_table() is first
