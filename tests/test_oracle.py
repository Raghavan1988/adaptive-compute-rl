"""Tests for oracle allocation (M2). The oracle is the upper bound on adaptive compute.

The three PLAN-required toy trajectories map directly onto the argmax-with-cheaper-
tiebreak rule (STOP = cheaper budget, CONTINUE = more expensive budget):
  - STOP correct  / CONTINUE correct -> STOP   (both right => take the cheaper)
  - STOP wrong    / CONTINUE correct -> CONTINUE iff the accuracy gain outweighs lambda*cost
  - STOP wrong    / CONTINUE wrong   -> cheaper action (equal accuracy => cheaper wins)
"""

import pytest

from when_to_think.evaluation import (
    oracle_allocation,
    oracle_frontier,
    per_example_budget_stats,
    summarize_oracle,
)

STOP, CONTINUE = 10, 100  # cheaper vs more-expensive budget (reasoning_tokens = budget)


def _row(example_id, budget, correct, *, sample_index=0, reasoning_tokens=None):
    tokens = budget if reasoning_tokens is None else reasoning_tokens
    return {
        "example_id": example_id,
        "budget": budget,
        "correct": correct,
        "reasoning_tokens": tokens,
        "total_generated_tokens": tokens + 2,
        "sample_index": sample_index,
    }


# --------------------------------------------------------------------------- #
# Required toy trajectories
# --------------------------------------------------------------------------- #

def test_stop_correct_continue_correct_takes_cheaper():
    rows = [_row("A", STOP, True), _row("A", CONTINUE, True)]
    # Any positive penalty prefers the cheaper budget; at lambda=0 the tie breaks cheaper.
    for lam in (0.0, 1e-4):
        alloc = oracle_allocation(rows, lam)
        assert alloc["budget_choices"]["A"] == STOP


def test_stop_wrong_continue_correct_depends_on_cost():
    rows = [_row("A", STOP, False), _row("A", CONTINUE, True)]
    # Gain = 1.0 accuracy; cost = lambda * (CONTINUE - STOP) = 90*lambda.
    # Small lambda: continuing is worth it.
    assert oracle_allocation(rows, 1e-4)["budget_choices"]["A"] == CONTINUE
    # Large lambda (cost 0.02*90 = 1.8 > 1.0 gain): stop instead.
    assert oracle_allocation(rows, 0.02)["budget_choices"]["A"] == STOP


def test_stop_wrong_continue_wrong_takes_cheaper():
    rows = [_row("A", STOP, False), _row("A", CONTINUE, False)]
    for lam in (0.0, 1e-4, 0.02):
        assert oracle_allocation(rows, lam)["budget_choices"]["A"] == STOP


# --------------------------------------------------------------------------- #
# Aggregation and frontier
# --------------------------------------------------------------------------- #

def test_per_example_stats_average_samples():
    rows = [
        _row("A", 100, True, sample_index=0),
        _row("A", 100, False, sample_index=1),
    ]
    stats = per_example_budget_stats(rows)
    assert stats["A"][100]["accuracy"] == pytest.approx(0.5)
    assert stats["A"][100]["mean_reasoning_tokens"] == pytest.approx(100.0)
    assert stats["A"][100]["n"] == 2


def _heterogeneous_rows():
    # P is helped only by the largest budget; Q is already correct at 0; R is never correct.
    return [
        _row("P", 0, False), _row("P", 50, False), _row("P", 100, True),
        _row("Q", 0, True), _row("Q", 50, True), _row("Q", 100, True),
        _row("R", 0, False), _row("R", 50, False), _row("R", 100, False),
    ]


def test_accuracy_max_oracle_picks_cheapest_correct_per_example():
    alloc = oracle_allocation(_heterogeneous_rows(), 0.0)
    assert alloc["budget_choices"] == {"P": 100, "Q": 0, "R": 0}
    assert alloc["mean_accuracy"] == pytest.approx(2 / 3)
    # (100 + 0 + 0) / 3 examples.
    assert alloc["mean_reasoning_tokens"] == pytest.approx(100 / 3)
    assert alloc["budget_histogram"] == {0: 2, 100: 1}


def test_frontier_is_pareto_monotone_with_endpoints():
    frontier = oracle_frontier(_heterogeneous_rows())
    # Sorted by compute, accuracy must be non-decreasing (Pareto frontier).
    frontier = sorted(frontier, key=lambda p: p["mean_reasoning_tokens"])
    accs = [p["mean_accuracy"] for p in frontier]
    assert accs == sorted(accs)
    # Cheapest vertex = everyone at budget 0 (acc 1/3 @ 0 tokens).
    assert frontier[0]["mean_reasoning_tokens"] == pytest.approx(0.0)
    assert frontier[0]["mean_accuracy"] == pytest.approx(1 / 3)
    # Richest vertex = accuracy-max oracle (acc 2/3 @ 100/3 tokens).
    assert frontier[-1]["mean_accuracy"] == pytest.approx(2 / 3)
    assert frontier[-1]["mean_reasoning_tokens"] == pytest.approx(100 / 3)


def test_summary_reports_savings_at_matched_accuracy():
    summary = summarize_oracle(_heterogeneous_rows())
    assert summary["best_fixed_budget"] == 100  # highest-accuracy fixed budget
    # Oracle reaches 2/3 accuracy at 100/3 tokens; fixed b=100 needs 100 tokens.
    assert summary["compute_saved_vs_best_fixed"] == pytest.approx(100 - 100 / 3)
    assert summary["compute_saved_fraction_vs_best_fixed"] == pytest.approx(2 / 3)
    assert summary["oracle_dominates_fixed"] is True


def test_oracle_does_not_beat_fixed_when_cheapest_already_solves():
    # Everything correct at budget 0 already: no allocation can save compute.
    rows = [
        _row("A", 0, True), _row("A", 100, True),
        _row("B", 0, True), _row("B", 100, True),
    ]
    summary = summarize_oracle(rows)
    assert summary["best_fixed_budget"] == 0
    assert summary["compute_saved_vs_best_fixed"] == pytest.approx(0.0)
    assert summary["oracle_dominates_fixed"] is False


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        oracle_allocation([_row("A", 0, True)], -1e-3)


def test_requires_two_budgets():
    with pytest.raises(ValueError, match="two distinct budgets"):
        summarize_oracle([_row("A", 0, True), _row("A", 0, False, sample_index=1)])
