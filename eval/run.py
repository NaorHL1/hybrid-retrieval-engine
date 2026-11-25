"""Per-stage retrieval eval.

    python -m eval.run                   # score, print a table, write eval/results.json
    python -m eval.run --write-baseline  # also freeze eval/baseline.json (deliberate act)

Configurations (the RRF stage mirrors api.py: fuse with limit 1000, then cut to 50):
  dense    dense(q, 50)
  bm25     bm25(q, 50)
  rrf      hybrid_rrf(q, 1000)[:50]
  rrf+ce   cross-encoder rerank of those same 50 candidates

MRR is computed over the top 50 (a doc ranked below 50 counts as 0).
"""
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("LANCE_LOG", "error")  # silence lance scanner deprecation spam

from eval.fingerprint import corpus_fingerprint, file_sha256
from eval.gate import THRESHOLD, gated_values
from eval.metrics import (
    bootstrap_ci, first_rank, hit_at_k, paired_bootstrap_diff, reciprocal_rank, relevant_set,
)
from retrieval import (
    CROSS_ENCODER_NAME, FIELDS, bm25, dense, hybrid_rrf, open_table, rerank,
)

EVAL_DIR = Path(__file__).parent
QUERIES = EVAL_DIR / "queries.jsonl"
RESULTS = EVAL_DIR / "results.json"
BASELINE = EVAL_DIR / "baseline.json"
RRF_POOL = 1000   # same as api.py
CANDIDATES = 50   # same as api.py: what the cross-encoder sees
CONFIGS = ["dense", "bm25", "rrf", "rrf+ce"]
MODES = {"strict": False, "lenient": True}
PAIRS = [("bm25", "dense"), ("rrf", "bm25"), ("rrf+ce", "rrf")]  # (better?, baseline)


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


def _ci(values: list[float]) -> dict:
    mean, lo, hi = bootstrap_ci(values)
    return {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def aggregate(per_query: list[dict], scope: str) -> dict:
    rows = [r for r in per_query if scope == "all" or r["bucket"] == scope]
    out = {"n": len(rows)}
    for mode in MODES:
        out[mode] = {}
        for cfg in CONFIGS:
            out[mode][cfg] = {name: _ci([r["metrics"][mode][cfg][name] for r in rows])
                              for name in ("hit@1", "hit@3", "mrr")}
        out[mode]["candidate_recall@50"] = _ci([r["metrics"][mode]["rrf"]["recall@50"] for r in rows])
    return out


def paired_comparisons(per_query: list[dict]) -> dict:
    """Difference between neighboring stages per metric and scope, with a paired bootstrap CI.
    Layout: paired[mode][scope]["better - base"][metric] = {diff, ci95}."""
    scopes = ["all"] + sorted({r["bucket"] for r in per_query})
    out = {}
    for mode in MODES:
        out[mode] = {}
        for scope in scopes:
            rows = [r for r in per_query if scope == "all" or r["bucket"] == scope]
            out[mode][scope] = {}
            for better, base in PAIRS:
                cell = {}
                for metric in ("hit@1", "hit@3", "mrr"):
                    a = [r["metrics"][mode][better][metric] for r in rows]
                    b = [r["metrics"][mode][base][metric] for r in rows]
                    mean, lo, hi = paired_bootstrap_diff(a, b)
                    cell[metric] = {"diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
                out[mode][scope][f"{better} - {base}"] = cell
    return out


def fingerprints(table) -> dict:
    rows = table.search().select(FIELDS).limit(table.count_rows()).to_list()
    return {
        "corpus": corpus_fingerprint(rows),
        "queries_sha256": file_sha256(QUERIES),
        "rrf_pool": RRF_POOL, "candidates": CANDIDATES, "cross_encoder": CROSS_ENCODER_NAME,
    }


def run_eval() -> dict:
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
    return {
        "config": {"rrf_pool": RRF_POOL, "candidates": CANDIDATES, "n_queries": len(queries),
                   "bootstrap": {"n": 1000, "seed": 0, "ci": 0.95}},
        "fingerprint": fingerprints(table),
        "summary": {s: aggregate(per_query, s) for s in scopes},
        "paired": paired_comparisons(per_query),
        "latency": {k: {"p50_ms": round(statistics.median(v), 1), "max_ms": round(max(v), 1)}
                    for k, v in latencies.items()},
        "per_query": per_query,
    }


def write_baseline(results: dict, path: Path = BASELINE) -> dict:
    ref = {mode: {cfg: {m: results["summary"]["all"][mode][cfg][m]["mean"] for m in ("hit@1", "hit@3", "mrr")}
                  for cfg in CONFIGS} for mode in MODES}
    baseline = {
        "created": datetime.date.today().isoformat(),
        "threshold": THRESHOLD,
        "note": "Refreeze only on purpose: python -m eval.run --write-baseline, and review the diff.",
        "fingerprint": results["fingerprint"],
        "gated": gated_values(results["summary"]),
        "reference_overall": ref,
    }
    path.write_text(json.dumps(baseline, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return baseline


def print_report(results: dict) -> None:
    summary, scopes = results["summary"], list(results["summary"])
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
        print("paired differences between neighboring stages (same queries). '*' = 95% CI excludes 0.")
        print("  CAUTION: 36 intervals per mode; about 2 will exclude 0 by chance alone.")
        print(f"  {'comparison':16} {'scope':>10} | {'Hit@1':>9} | {'Hit@3':>9} | {'MRR':>9}")
        for scope, pairs in results["paired"][mode].items():
            for name, cell in pairs.items():
                cols = []
                for metric in ("hit@1", "hit@3", "mrr"):
                    p = cell[metric]; star = "*" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else " "
                    cols.append(f"{p['diff']:+.2f}{star}")
                print(f"  {name:16} {scope:>10} | {cols[0]:>9} | {cols[1]:>9} | {cols[2]:>9}")
    print("\n=== latency per stage (ms, single run: indicative only) ===")
    for k, v in results["latency"].items():
        print(f"{k:6} p50={v['p50_ms']:>7}  max={v['max_ms']:>7}")


def main(argv: list[str]) -> int:
    results = run_eval()
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print_report(results)
    print(f"\nwrote {RESULTS}")
    if "--write-baseline" in argv:
        write_baseline(results)
        print(f"wrote {BASELINE} (gated: {gated_values(results['summary'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
