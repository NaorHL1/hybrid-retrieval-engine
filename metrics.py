"""Pure retrieval metrics. No models or DB, so they are unit-testable anywhere."""
from collections import defaultdict


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(doc_id in relevant for doc_id in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def rank_of(ranked: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant doc, or None if it is not in the list."""
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return position
    return None


def summarize(rows: list[dict], k: int = 3) -> dict:
    """rows: [{"type": str, "ranked": [ids], "relevant": set[ids]}] -> metrics per type + overall."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
        groups["ALL"].append(row)
    report = {}
    for name, items in groups.items():
        n = len(items)
        report[name] = {
            "n": n,
            f"hit@{k}": sum(hit_at_k(r["ranked"], r["relevant"], k) for r in items) / n,
            f"recall@{k}": sum(recall_at_k(r["ranked"], r["relevant"], k) for r in items) / n,
            "mrr": sum(reciprocal_rank(r["ranked"], r["relevant"]) for r in items) / n,
        }
    return report
