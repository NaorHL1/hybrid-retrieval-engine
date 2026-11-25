"""FastAPI front for the retrieval pipeline. All retrieval logic lives in retrieval.py,
so the API and the eval (eval/run.py) always run the same code.

Concurrency design:
- The LanceDB table is opened once and reused (not once per request).
- Dense/RRF search is fast (tens of ms) and runs inline in the handler.
- The cross-encoder is CPU-bound and takes seconds, so it runs in a thread executor behind a
  semaphore: the event loop stays free, and concurrent rerank calls queue instead of
  fighting over the same CPU cores.
"""
import asyncio
import os
import weakref
from contextlib import asynccontextmanager

from fastapi import FastAPI

from retrieval import dense, hybrid_rrf, open_table, rerank

RRF_POOL = 1000   # fuse over the whole corpus, then cut (same as eval/run.py)
CANDIDATES = 50   # what the cross-encoder sees
TOP_K = 5
CE_CONCURRENCY = int(os.environ.get("CE_CONCURRENCY", "1"))

_table = None
_semaphores: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def get_table():
    """Open the LanceDB table once and reuse it. Lazy so direct callers (benchmark.py)
    and TestClient without a lifespan still work."""
    global _table
    if _table is None:
        _table = open_table()
    return _table


def ce_semaphore() -> asyncio.Semaphore:
    """One semaphore per event loop (a semaphore must not be shared across loops)."""
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = _semaphores[loop] = asyncio.Semaphore(CE_CONCURRENCY)
    return sem


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_table()
    # Load the cross-encoder now so the first user request does not pay for it.
    await asyncio.get_running_loop().run_in_executor(
        None, rerank, "warmup", [{"title": "", "text": ""}], 1
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/search")
async def search_incident(query: str):
    """Vector (dense) search only."""
    return dense(get_table(), query, TOP_K)


@app.get("/hybrid-search")
async def hybrid_search_incident(query: str):
    """Dense + BM25 fused with RRF, then cross-encoder rerank of the top candidates."""
    candidates = hybrid_rrf(get_table(), query, RRF_POOL)[:CANDIDATES]
    if not candidates:
        return []
    loop = asyncio.get_running_loop()
    async with ce_semaphore():
        return await loop.run_in_executor(None, rerank, query, candidates, TOP_K)
