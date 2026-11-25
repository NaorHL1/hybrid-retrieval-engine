"""Regression gate: pure functions, no DB or models.

Gated metrics (overall, lenient): production pipeline Hit@3 and the candidate Recall@50
(does the right document even reach the cross-encoder). A drop of MORE than THRESHOLD fails;
exactly THRESHOLD passes. Improvements never fail.
"""
THRESHOLD = 0.05
EPS = 1e-9


def gated_values(summary: dict) -> dict[str, float]:
    s = summary["all"]["lenient"]
    return {
        "rrf+ce hit@3 (lenient)": s["rrf+ce"]["hit@3"]["mean"],
        "candidate recall@50 (lenient)": s["candidate_recall@50"]["mean"],
    }


def stale_reasons(baseline_fp: dict, current_fp: dict) -> list[str]:
    """Differences that make a comparison meaningless (data or settings changed)."""
    reasons = []
    for key in sorted(set(baseline_fp) | set(current_fp)):
        if baseline_fp.get(key) != current_fp.get(key):
            reasons.append(f"{key}: baseline={baseline_fp.get(key)!r} current={current_fp.get(key)!r}")
    return reasons


def regressions(baseline: dict[str, float], current: dict[str, float],
                threshold: float = THRESHOLD) -> list[str]:
    failures = []
    for name, base in baseline.items():
        if name not in current:
            failures.append(f"{name}: missing from current results")
            continue
        drop = base - current[name]
        if drop > threshold + EPS:
            failures.append(f"{name}: {base:.4f} -> {current[name]:.4f} (drop {drop:.4f} > {threshold})")
    return failures
