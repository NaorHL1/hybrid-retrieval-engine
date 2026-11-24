"""Retrieval eval: per-query-type metrics for vector vs hybrid, plus a per-stage miss trace.

    uv run python evaluate.py                 # full report
    uv run python evaluate.py --trace "A6-FKF" --expect doc_0003
    uv run python evaluate.py --json out.json # save per-query ranks for diffing between runs
"""
import argparse
import asyncio
import json
import time

from eval_set import EVAL_SET
from metrics import rank_of, summarize

DEPTH = 10  # how deep we look for MRR / Recall


async def run_pipeline(name, fn, query):
    start = time.perf_counter()
    results = await fn(query, top_k=DEPTH)
    latency = time.perf_counter() - start
    return [doc["id"] for doc in results], latency


async def run_eval(k: int = 3):
    # imported here so `metrics.py` stays importable (and testable) without loading any model
    from api import hybrid_search_incident, search_incident

    pipelines = {"vector": search_incident, "hybrid": hybrid_search_incident}
    rows = {name: [] for name in pipelines}
    latencies = {name: [] for name in pipelines}
    per_query = []

    for query, relevant, qtype in EVAL_SET:
        entry = {"query": query, "type": qtype, "relevant": relevant}
        for name, fn in pipelines.items():
            ranked, latency = await run_pipeline(name, fn, query)
            rows[name].append({"type": qtype, "ranked": ranked, "relevant": set(relevant)})
            latencies[name].append(latency)
            entry[name] = {"rank": rank_of(ranked, set(relevant)), "top": ranked[:k]}
        per_query.append(entry)

    reports = {name: summarize(rows[name], k=k) for name in pipelines}
    return reports, latencies, per_query


def print_report(reports, latencies, per_query, k=3):
    types = ["ALL"] + sorted(t for t in reports["vector"] if t != "ALL")
    print(f"\n{'type':<12}{'n':>3}  " + "  ".join(f"{m + ' ' + p:>14}" for m in (f'hit@{k}', 'mrr') for p in ('vector', 'hybrid')))
    for t in types:
        n = reports["vector"][t]["n"]
        cells = [reports[p][t][m] for m in (f"hit@{k}", "mrr") for p in ("vector", "hybrid")]
        print(f"{t:<12}{n:>3}  " + "  ".join(f"{c:>14.2f}" for c in cells))
    for name, vals in latencies.items():
        print(f"avg latency {name}: {1000 * sum(vals) / len(vals):.0f} ms")

    print("\nMisses (expected doc not in top-%d):" % k)
    for e in per_query:
        for name in ("vector", "hybrid"):
            rank = e[name]["rank"]
            if rank is None or rank > k:
                print(f"  [{name:<6}] ({e['type']}) rank={rank}  {e['query']}")


async def trace(query: str, expected: set[str], depth: int = 50):
    """Where does the expected doc sit after EACH stage? Shows which stage loses it."""
    from api import _open_table, cross_encode, rrf_candidates

    table = await _open_table()
    vec = await (await table.search(query)).limit(depth).to_list()
    fts = await (await table.search(query, query_type="fts", fts_columns="text")).limit(depth).to_list()
    rrf = await rrf_candidates(query)
    final = cross_encode(query, rrf[:50])

    stages = {
        "dense (vector)": [d["id"] for d in vec],
        "BM25 (fts)": [d["id"] for d in fts],
        "RRF fusion": [d["id"] for d in rrf],
        "cross-encoder (top 50 of RRF)": [d["id"] for d in final],
    }
    print(f"\nQuery: {query}\nExpected: {sorted(expected)}\n")
    for stage, ids in stages.items():
        r = rank_of(ids, expected)
        print(f"  {stage:<32} rank = {r if r else 'not in top ' + str(len(ids))}")
    print("\nTop-3 after cross-encoder:")
    for d in final[:3]:
        print(f"  {d['id']}  ce={d['cross_encoder_score']:.3f}  {d['text'][:90]!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", help="query to trace through every stage")
    parser.add_argument("--expect", nargs="+", help="expected doc ids for --trace")
    parser.add_argument("--json", help="write per-query results to this file")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    if args.trace:
        asyncio.run(trace(args.trace, set(args.expect or [])))
    else:
        reports, latencies, per_query = asyncio.run(run_eval(args.k))
        print_report(reports, latencies, per_query, args.k)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"reports": reports, "per_query": per_query}, f, indent=2)
