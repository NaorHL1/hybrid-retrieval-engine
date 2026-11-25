"""Cross-encoder candidate-count sweep.   python -m eval.sweep

For each K, rerank the top-K RRF candidates (RRF fused with limit 1000, as in production) and
report quality (deterministic, one pass) and cross-encoder latency (measured REPEATS times,
interleaved across K to spread machine noise; we report median and minimum, not one run).
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("LANCE_LOG", "error")

from eval.metrics import bootstrap_ci, first_rank, hit_at_k, paired_bootstrap_diff, reciprocal_rank, relevant_set
from eval.run import QUERIES, RRF_POOL, load_queries
from retrieval import hybrid_rrf, open_table, rerank

KS = [5, 10, 20, 50]
REPEATS = 3
OUT = QUERIES.parent / "sweep_results.json"
MODES = {"strict": False, "lenient": True}


def main() -> int:
    queries, table = load_queries(), open_table()
    rerank("warmup", [{"title": "", "text": ""}], 1)
    fused = {q["id"]: hybrid_rrf(table, q["query"], RRF_POOL) for q in queries}

    times = {k: {q["id"]: [] for q in queries} for k in KS}  # seconds, per query per repeat
    ranked = {k: {} for k in KS}
    for rep in range(REPEATS):
        for qd in queries:
            for k in KS:
                cands = fused[qd["id"]][:k]
                t0 = time.perf_counter()
                out = rerank(qd["query"], cands, k)
                times[k][qd["id"]].append(time.perf_counter() - t0)
                if rep == 0:
                    ranked[k][qd["id"]] = [x["id"] for x in out]
        print(f"repeat {rep + 1}/{REPEATS} done", flush=True)

    results = {"ks": KS, "repeats": REPEATS, "n_queries": len(queries), "by_k": {}}
    per_q = {m: {k: {} for k in KS} for m in MODES}
    for mode, lenient in MODES.items():
        for k in KS:
            for qd in queries:
                rel, ids = relevant_set(qd, lenient), ranked[k][qd["id"]]
                cand_ids = [x["id"] for x in fused[qd["id"]][:k]]
                per_q[mode][k][qd["id"]] = {
                    "hit@1": hit_at_k(ids, rel, 1), "hit@3": hit_at_k(ids, rel, 3),
                    "mrr": reciprocal_rank(ids, rel), "cand_recall": hit_at_k(cand_ids, rel, k),
                }
    order = [q["id"] for q in queries]
    for k in KS:
        entry = {}
        for mode in MODES:
            for metric in ("hit@1", "hit@3", "mrr", "cand_recall"):
                vals = [per_q[mode][k][i][metric] for i in order]
                mean, lo, hi = bootstrap_ci(vals)
                entry[f"{metric} {mode}"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            vs50 = paired_bootstrap_diff([per_q[mode][k][i]["hit@3"] for i in order],
                                         [per_q[mode][50][i]["hit@3"] for i in order])
            entry[f"hit@3 {mode} minus K=50"] = {"diff": round(vs50[0], 4), "ci95": [round(vs50[1], 4), round(vs50[2], 4)]}
        per_query_median = [statistics.median(times[k][i]) for i in order]
        run_totals = [sum(times[k][i][r] for i in order) / len(order) for r in range(REPEATS)]
        entry["latency_ms"] = {
            "median_per_query": round(statistics.median(per_query_median) * 1000, 1),
            "mean_per_query_by_repeat": [round(t * 1000, 1) for t in run_totals],
            "min_mean_per_query": round(min(run_totals) * 1000, 1),
        }
        results["by_k"][str(k)] = entry
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"\n{'K':>3} | {'cand recall':>11} | {'Hit@1':>5} {'Hit@3':>5} {'MRR':>5} (lenient) | "
          f"{'Hit@3 strict':>12} | {'Hit@3 - K50 [paired 95% CI]':>29} | cross-encoder ms/query (median, min-of-repeats)")
    for k in KS:
        e = results["by_k"][str(k)]
        d = e["hit@3 lenient minus K=50"]
        print(f"{k:>3} | {e['cand_recall lenient']['mean']:>11.2f} | {e['hit@1 lenient']['mean']:>5.2f} "
              f"{e['hit@3 lenient']['mean']:>5.2f} {e['mrr lenient']['mean']:>5.2f}          | "
              f"{e['hit@3 strict']['mean']:>12.2f} | {d['diff']:>+8.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]   | "
              f"{e['latency_ms']['median_per_query']:>8} , {e['latency_ms']['min_mean_per_query']:>8}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
