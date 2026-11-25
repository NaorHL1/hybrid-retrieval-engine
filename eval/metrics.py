"""Pure metric functions for the retrieval eval. No DB, no models.

Strict vs lenient: strict counts only `expected_id`; lenient also counts every id in
`also_relevant` (near-duplicate docs that answer the same question).
"""
import random


def relevant_set(query: dict, lenient: bool) -> set[str]:
    ids = {query["expected_id"]}
    if lenient:
        ids |= set(query.get("also_relevant", []))
    return ids


def first_rank(ranked_ids: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant id, or None if absent."""
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return rank
    return None


def hit_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    rank = first_rank(ranked_ids, relevant)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    rank = first_rank(ranked_ids, relevant)
    return 0.0 if rank is None else 1.0 / rank


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over per-query values. Fixed seed => reproducible."""
    if not values:
        raise ValueError("values must not be empty")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(sum(rng.choices(values, k=size)) / size for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(values) / size, lo, hi


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) per query and its percentile CI. Resamples *queries*, so both configs
    always see the same queries; this is the right way to ask "is config a really better than b?".
    If the CI excludes 0, the gap is unlikely to be query-sampling noise."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length (one value per query)")
    return bootstrap_ci([x - y for x, y in zip(a, b)], n_boot=n_boot, seed=seed, alpha=alpha)
