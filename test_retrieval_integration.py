"""Checks that our manual rrf_fuse reproduces LanceDB's RRFReranker on the real index.

If this fails, per-stage traces built on rrf_fuse can no longer be trusted.
Run: pytest -m integration
"""
import os

import pytest

from retrieval import DB_URI, bm25, dense, hybrid_rrf, open_table, rrf_fuse

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.path.isdir(DB_URI), reason="vector_database/ not built"),
]

QUERIES = [
    "Who was the medical professional that treated the wounded pilot?",
    "How did a passenger know how to stabilize the plane controls from a TV show?",
    "Why did the attacker abort a previous hijacking attempt on an earlier flight?",
    "Squawk 7500 and 7700 emergency broadcast",
    "A6-FKF",
    "Shin Bet",
    "+971 54 232 3578",
    "Air Transport Law 1980",
    "Iberia Airlines v. Fleischer Peled",
    "Dunedin, New Zealand SpiceJet 11 years",
]


@pytest.fixture(scope="module")
def table():
    return open_table()


@pytest.mark.parametrize("query", QUERIES)
def test_manual_rrf_matches_lancedb_order(table, query):
    n = table.count_rows()  # large enough that no leg is truncated
    dense_ids = [d["id"] for d in dense(table, query, n)]
    bm25_ids = [d["id"] for d in bm25(table, query, n)]
    lancedb_ids = [d["id"] for d in hybrid_rrf(table, query, n)]
    mine = [doc_id for doc_id, _ in rrf_fuse([dense_ids, bm25_ids])]
    assert mine == lancedb_ids
