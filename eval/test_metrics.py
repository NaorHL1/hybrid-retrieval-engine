import pytest

from eval.metrics import (
    bootstrap_ci, first_rank, hit_at_k, relevant_set, reciprocal_rank,
)

Q = {"expected_id": "a", "also_relevant": ["b"]}


def test_relevant_set_strict_vs_lenient():
    assert relevant_set(Q, lenient=False) == {"a"}
    assert relevant_set(Q, lenient=True) == {"a", "b"}


def test_relevant_set_missing_also_relevant():
    assert relevant_set({"expected_id": "a"}, lenient=True) == {"a"}


def test_first_rank_is_one_based_and_none_when_absent():
    assert first_rank(["x", "a", "y"], {"a"}) == 2
    assert first_rank(["x", "y"], {"a"}) is None


def test_first_rank_takes_earliest_of_several_relevant():
    assert first_rank(["x", "b", "a"], {"a", "b"}) == 2


def test_hit_at_k_boundary():
    ranked = ["x", "y", "a"]
    assert hit_at_k(ranked, {"a"}, 3) == 1.0
    assert hit_at_k(ranked, {"a"}, 2) == 0.0


def test_lenient_can_rescue_a_strict_miss():
    ranked = ["b", "x", "y", "a"]
    assert hit_at_k(ranked, relevant_set(Q, False), 1) == 0.0
    assert hit_at_k(ranked, relevant_set(Q, True), 1) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_bootstrap_constant_values_have_zero_width():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_bootstrap_is_reproducible_with_fixed_seed():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert bootstrap_ci(vals, seed=7) == bootstrap_ci(vals, seed=7)


def test_bootstrap_interval_contains_mean_and_stays_in_range():
    mean, lo, hi = bootstrap_ci([1.0] * 6 + [0.0] * 4)
    assert mean == pytest.approx(0.6)
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])
