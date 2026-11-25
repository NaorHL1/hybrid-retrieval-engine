import pytest

from eval.fingerprint import corpus_fingerprint, file_sha256
from eval.gate import THRESHOLD, gated_values, regressions, stale_reasons
from eval.metrics import paired_bootstrap_diff


# ---- gate ----------------------------------------------------------------
def test_drop_of_exactly_threshold_passes():
    assert regressions({"m": 0.90}, {"m": 0.85}) == []


def test_drop_beyond_threshold_fails():
    assert len(regressions({"m": 0.90}, {"m": 0.849})) == 1


def test_improvement_never_fails():
    assert regressions({"m": 0.80}, {"m": 0.99}) == []


def test_two_of_45_queries_pass_three_fail():
    # one query is worth 1/45 = 0.0222: two lost queries (0.044) pass, three (0.067) fail
    base = 40 / 45
    assert regressions({"m": base}, {"m": 38 / 45}) == []
    assert len(regressions({"m": base}, {"m": 37 / 45})) == 1


def test_missing_metric_is_a_failure():
    assert regressions({"m": 0.9}, {}) != []


def test_each_metric_is_judged_separately():
    out = regressions({"a": 0.9, "b": 0.9}, {"a": 0.9, "b": 0.5})
    assert len(out) == 1 and out[0].startswith("b")


def test_gated_values_reads_the_lenient_overall_summary():
    summary = {"all": {"lenient": {
        "rrf+ce": {"hit@3": {"mean": 0.9}}, "candidate_recall@50": {"mean": 0.97},
        "dense": {"hit@3": {"mean": 0.1}}}}}
    assert gated_values(summary) == {"rrf+ce hit@3 (lenient)": 0.9, "candidate recall@50 (lenient)": 0.97}


def test_threshold_matches_the_agreed_value():
    assert THRESHOLD == 0.05


# ---- staleness -------------------------------------------------------------
def test_stale_reasons_empty_when_equal_and_explains_differences():
    fp = {"corpus": {"n_docs": 3, "sha256": "x"}, "queries_sha256": "q"}
    assert stale_reasons(fp, dict(fp)) == []
    other = {"corpus": {"n_docs": 4, "sha256": "x"}, "queries_sha256": "q"}
    reasons = stale_reasons(fp, other)
    assert len(reasons) == 1 and reasons[0].startswith("corpus")


# ---- fingerprints ----------------------------------------------------------
ROWS = [{"id": "b", "title": "t2", "text": "two"}, {"id": "a", "title": "t1", "text": "one"}]


def test_corpus_fingerprint_ignores_row_order():
    assert corpus_fingerprint(ROWS) == corpus_fingerprint(list(reversed(ROWS)))


def test_corpus_fingerprint_changes_when_text_title_or_count_changes():
    base = corpus_fingerprint(ROWS)["sha256"]
    assert corpus_fingerprint([{**ROWS[0], "text": "TWO"}, ROWS[1]])["sha256"] != base
    assert corpus_fingerprint([{**ROWS[0], "title": "T2"}, ROWS[1]])["sha256"] != base
    assert corpus_fingerprint(ROWS[:1])["n_docs"] == 1


def test_corpus_fingerprint_field_boundaries_are_unambiguous():
    # "ab"+"c" must not hash like "a"+"bc"
    assert corpus_fingerprint([{"id": "ab", "title": "c", "text": ""}]) != \
        corpus_fingerprint([{"id": "a", "title": "bc", "text": ""}])


def test_file_sha256_ignores_line_ending_style(tmp_path):
    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes(b"a\nb\n")
    crlf.write_bytes(b"a\r\nb\r\n")
    assert file_sha256(lf) == file_sha256(crlf)


# ---- paired bootstrap --------------------------------------------------------
def test_paired_diff_identical_configs_is_zero():
    assert paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1]) == (0.0, 0.0, 0.0)


def test_paired_diff_consistently_better_excludes_zero():
    mean, lo, hi = paired_bootstrap_diff([1.0] * 10, [0.0] * 10)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)


def test_paired_diff_mixed_results_ci_contains_zero():
    a = [1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1, 0, 1]
    mean, lo, hi = paired_bootstrap_diff(a, b)
    assert mean == 0.0 and lo < 0.0 < hi


def test_paired_diff_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])
