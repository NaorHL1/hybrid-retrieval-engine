import pytest

from retrieval import rrf_fuse


def test_single_list_keeps_order():
    fused = rrf_fuse([["a", "b", "c"]])
    assert [doc for doc, _ in fused] == ["a", "b", "c"]


def test_exact_scores_k60():
    # "a" is rank 1 in both lists, "b" rank 2 in one, "c" rank 2 in the other.
    fused = dict(rrf_fuse([["a", "b"], ["a", "c"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 62)


def test_doc_in_both_lists_beats_doc_ranked_first_in_one():
    # "x" is #1 only in dense; "y" is #2 in both. 2/62 > 1/61.
    fused = rrf_fuse([["x", "y"], ["z", "y"]])
    assert fused[0][0] == "y"


def test_ties_are_deterministic_first_seen_first():
    fused = rrf_fuse([["b"], ["a"]])
    assert [doc for doc, _ in fused] == ["b", "a"]


def test_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_invalid_k():
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
