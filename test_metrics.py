from metrics import hit_at_k, recall_at_k, reciprocal_rank, rank_of, summarize


def test_hit_and_rank():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, 3) == 1.0
    assert hit_at_k(ranked, {"d"}, 3) == 0.0
    assert rank_of(ranked, {"c"}) == 3
    assert rank_of(ranked, {"zzz"}) is None


def test_mrr_rewards_higher_rank():
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a", "x"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_recall_with_multiple_relevant():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == 2 / 3


def test_summarize_splits_by_type():
    rows = [
        {"type": "semantic", "ranked": ["a"], "relevant": {"a"}},
        {"type": "identifier", "ranked": ["x", "y", "z", "a"], "relevant": {"a"}},
    ]
    report = summarize(rows, k=3)
    assert report["semantic"]["hit@3"] == 1.0
    assert report["identifier"]["hit@3"] == 0.0
    assert report["ALL"]["n"] == 2
