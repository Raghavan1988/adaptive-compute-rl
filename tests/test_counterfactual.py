"""Tests for counterfactual value-of-compute analysis (M1, Question 1).

The heterogeneity and non-monotonicity logic is the scientific core of M1, so it is
tested on synthetic rows with hand-checked expected categorizations.
"""

import pytest

from when_to_think.evaluation import (
    accuracy_by_budget,
    per_example_accuracy,
    summarize_counterfactuals,
)


def _row(example_id, budget, correct, *, reasoning_tokens=None, sample_index=0):
    tokens = budget if reasoning_tokens is None else reasoning_tokens
    return {
        "example_id": example_id,
        "budget": budget,
        "correct": correct,
        "reasoning_tokens": tokens,
        "total_generated_tokens": tokens + 2,
        "sample_index": sample_index,
    }


def test_heterogeneous_mix_of_helped_hurt_unchanged():
    rows = [
        _row("A", 0, True), _row("A", 100, True),    # unchanged (already correct)
        _row("B", 0, False), _row("B", 100, True),   # helped
        _row("C", 0, True), _row("C", 100, False),   # hurt + non-monotone
    ]
    s = summarize_counterfactuals(rows)
    voc = s["value_of_compute"]
    assert s["n_examples"] == 3
    assert voc["counts"] == {
        "helped": 1, "hurt": 1, "unchanged": 1,
        "unchanged_correct": 1, "unchanged_wrong": 0,
    }
    assert voc["nonmonotone_examples"] == 1
    assert voc["heterogeneous"] is True


def test_homogeneous_all_helped_is_not_heterogeneous():
    rows = [
        _row("A", 0, False), _row("A", 100, True),
        _row("B", 0, False), _row("B", 100, True),
    ]
    voc = summarize_counterfactuals(rows)["value_of_compute"]
    assert voc["counts"]["helped"] == 2
    assert voc["heterogeneous"] is False  # all examples in one bucket


def test_nonmonotone_detected_even_when_endpoints_equal():
    # Wrong at min and max (delta 0 => unchanged) but correct in the middle: a
    # correct->wrong drop that the no-monotonicity check must still flag.
    rows = [
        _row("X", 0, False), _row("X", 50, True), _row("X", 100, False),
    ]
    voc = summarize_counterfactuals(rows)["value_of_compute"]
    assert voc["counts"]["unchanged"] == 1
    assert voc["counts"]["unchanged_wrong"] == 1
    assert voc["nonmonotone_examples"] == 1


def test_accuracy_by_budget_values():
    rows = [
        _row("A", 0, True), _row("B", 0, False),      # budget 0: 1/2
        _row("A", 100, True), _row("B", 100, True),   # budget 100: 2/2
    ]
    by_budget = accuracy_by_budget(rows)
    assert by_budget[0]["accuracy"] == pytest.approx(0.5)
    assert by_budget[100]["accuracy"] == pytest.approx(1.0)
    assert by_budget[0]["n"] == 2


def test_per_example_accuracy_averages_samples():
    rows = [
        _row("A", 100, True, sample_index=0),
        _row("A", 100, False, sample_index=1),
    ]
    per_ex = per_example_accuracy(rows)
    assert per_ex["A"][100] == pytest.approx(0.5)


def test_requires_at_least_two_budgets():
    with pytest.raises(ValueError, match="two distinct budgets"):
        summarize_counterfactuals([_row("A", 0, True)])
