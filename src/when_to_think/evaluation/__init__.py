"""Deterministic evaluation and machine-readable result files."""

from when_to_think.evaluation.counterfactual import (
    accuracy_by_budget,
    load_runs,
    per_example_accuracy,
    summarize_counterfactuals,
)
from when_to_think.evaluation.evaluate import run_evaluation
from when_to_think.evaluation.fixed_budget_eval import run_fixed_budget_sweep
from when_to_think.evaluation.oracle import (
    compute_savings_at_matched_accuracy,
    fixed_budget_points,
    oracle_allocation,
    oracle_frontier,
    per_example_budget_stats,
    summarize_oracle,
)

__all__ = [
    "accuracy_by_budget",
    "compute_savings_at_matched_accuracy",
    "fixed_budget_points",
    "load_runs",
    "oracle_allocation",
    "oracle_frontier",
    "per_example_accuracy",
    "per_example_budget_stats",
    "run_evaluation",
    "run_fixed_budget_sweep",
    "summarize_counterfactuals",
    "summarize_oracle",
]
