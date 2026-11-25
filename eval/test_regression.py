"""Regression gate: reruns the full eval and compares with eval/baseline.json.

    pytest eval/test_regression.py        (about a minute: it runs the cross-encoder 45 times)

Fails if (a) the baseline is missing or stale (data/queries/settings changed: refreeze on
purpose), or (b) the gated overall lenient metrics drop by more than 0.05.
"""
import json

import pytest

from eval.gate import gated_values, regressions, stale_reasons
from eval.run import BASELINE, run_eval

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def results():
    return run_eval()


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE.exists():
        pytest.fail(f"{BASELINE.name} is missing. Create it deliberately: python -m eval.run --write-baseline")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_baseline_is_not_stale(results, baseline):
    reasons = stale_reasons(baseline["fingerprint"], results["fingerprint"])
    assert not reasons, (
        "baseline.json no longer matches the corpus, queries or settings, so scores are not "
        "comparable. Review the change, then refreeze: python -m eval.run --write-baseline\n  "
        + "\n  ".join(reasons)
    )


def test_no_regression_against_baseline(results, baseline):
    failures = regressions(baseline["gated"], gated_values(results["summary"]), baseline["threshold"])
    assert not failures, "retrieval quality regressed:\n  " + "\n  ".join(failures)
