"""Per-stage retrieval eval.  Run:  python -m eval.run

Configurations (the RRF stage mirrors api.py: fuse with limit 1000, then cut to 50):
  dense    dense(q, 50)
  bm25     bm25(q, 50)
  rrf      hybrid_rrf(q, 1000)[:50]
  rrf+ce   cross-encoder rerank of those same 50 candidates

Writes eval/results.json with aggregate metrics and per-query ranks (for traces).
MRR is computed over the top 50 (a doc ranked below 50 counts as 0).
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("LANCE_LOG", "error")  # silence lance scanner deprecation spam

from eval.metrics import bootstrap_ci, first_rank, hit_at_k, reciprocal_rank, relevant_set
from retrieval import bm25, dense, hybrid_rrf, open_table, rerank

EVAL_DIR = Path(__file__).parent
QUERIES = EVAL_DIR / "queries.jsonl"
RESULTS = EVAL_DIR / "results.json"
RRF_POOL = 1000   # same as api.py
CANDIDATES = 50   # same as api.py: what the cross-encoder sees
CONFIGS = ["dense", "bm25", "rrf", "rrf+ce"]
MODES = {"strict": False, "lenient": True}


def load_queries(path: Path = QUERIES) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_query(table, q: str) -> tuple[dict, dict]:
    """Return ranked id lists per config and per-stage latency in ms."""
    timings = {}
    t0 = time.perf_counter(); d = dense(table, q, CANDIDATES); timings["dense"] = time.perf_counter() - t0
    t0 = time.perf_counter(); b = bm25(table, q, CANDIDATES); timings["bm25"] = time.perf_counter() - t0
    t0 = time.perf_counter(); fused = hybrid_rrf(table, q, RRF_POOL); timings["rrf"] = time.perf_counter() - t0
    cands = fused[:CANDIDATES]
    t0 = time.perf_counter(); ce = rerank(q, cands, CANDIDATES); timings["ce"] = time.perf_counter() - t0
    ranked = {
        "dense": [x["id"] for x in d],
        "bm25": [x["id"] for x in b],
        "rrf": [x["id"] for x in cands],
        "rrf+ce": [x["id"] for x in ce],
        "rrf_full": [x["id"] for x in fused],  # kept for traces, not scored
    }
    return ranked, {k: v * 1000 for k, v in timings.items()}


def aggregate(per_query: list[dict], scope: str) -> dict:
    rows = [r for r in per_query if scope == "all" or r["bucket"] == scope]
    out = {"n": len(rows)}
    for mode in MODES:
        out[mode] = {}
        for cfg in CONFIGS:
            m = {}
            for name in ("hit@1", "hit@3", "mrr"):
                mean, lo, hi = bootstrap_ci([r["metrics"][mode][cfg][name] for r in rows])
                m[name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            out[mode][cfg] = m
        mean, lo, hi = bootstrap_ci([r["metrics"][mode]["rrf"]["recall@50"] for r in rows])
        out[mode]["candidate_recall@50"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main() -> int:
    queries = load_queries()
    table = open_table()
    rerank("warmup", [{"title": "", "text": ""}], 1)  # load the model outside the timings

    per_query, latencies = [], {k: [] for k in ("dense", "bm25", "rrf", "ce")}
    for qd in queries:
        ranked, ms = run_query(table, qd["query"])
        for k, v in ms.items():
            latencies[k].append(v)
        metrics, ranks = {}, {}
        for mode, lenient in MODES.items():
            rel = relevant_set(qd, lenient)
            metrics[mode] = {}
            for cfg in CONFIGS:
                ids = ranked[cfg]
                metrics[mode][cfg] = {
                    "hit@1": hit_at_k(ids, rel, 1),
                    "hit@3": hit_at_k(ids, rel, 3),
                    "mrr": reciprocal_rank(ids, rel),
                    "recall@50": hit_at_k(ids, rel, CANDIDATES),
                }
            ranks[mode] = {cfg: first_rank(ranked[cfg], rel) for cfg in CONFIGS + ["rrf_full"]}
        per_query.append({
            "id": qd["id"], "bucket": qd["bucket"], "query": qd["query"],
            "expected_id": qd["expected_id"], "also_relevant": qd.get("also_relevant", []),
            "ranks": ranks, "top3": {cfg: ranked[cfg][:3] for cfg in CONFIGS}, "metrics": metrics,
        })

    scopes = ["all"] + sorted({q["bucket"] for q in queries})
    summary = {s: aggregate(per_query, s) for s in scopes}
    latency = {k: {"p50_ms": round(statistics.median(v), 1),
                   "max_ms": round(max(v), 1)} for k, v in latencies.items()}

    RESULTS.write_text(json.dumps({
        "config": {"rrf_pool": RRF_POOL, "candidates": CANDIDATES, "n_queries": len(queries),
                   "bootstrap": {"n": 1000, "seed": 0, "ci": 0.95}},
        "summary": summary, "latency": latency, "per_query": per_query,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    print_report(summary, latency, scopes)
    print(f"\nwrote {RESULTS}")
    return 0


def print_report(summary: dict, latency: dict, scopes: list[str]) -> None:
    for mode in MODES:
        print(f"\n=== {mode.upper()} (n={summary['all']['n']}) ===")
        head = f"{'config':8} | {'Hit@1':>5} | {'Hit@3  [95% CI]':>18} | {'MRR':>5} | " + " | ".join(
            f"{s[:5]:>5}" for s in scopes if s != "all")
        print(head + "   <- Hit@3 per bucket")
        print("-" * len(head))
        for cfg in CONFIGS:
            a = summary["all"][mode][cfg]
            lo, hi = a["hit@3"]["ci95"]
            per_bucket = " | ".join(f"{summary[s][mode][cfg]['hit@3']['mean']:>5.2f}" for s in scopes if s != "all")
            print(f"{cfg:8} | {a['hit@1']['mean']:>5.2f} | {a['hit@3']['mean']:>5.2f}  [{lo:.2f}-{hi:.2f}]   | "
                  f"{a['mrr']['mean']:>5.2f} | {per_bucket}")
        r = summary["all"][mode]["candidate_recall@50"]
        print(f"candidate Recall@50 (rrf top-50 reaches cross-encoder): {r['mean']:.2f} "
              f"[{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}]")
    print("\n=== latency per stage (ms) ===")
    for k, v in latency.items():
        print(f"{k:6} p50={v['p50_ms']:>7}  max={v['max_ms']:>7}")


if __name__ == "__main__":
    sys.exit(main())
