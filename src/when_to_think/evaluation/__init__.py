"""Deterministic evaluation and machine-readable result files."""

from when_to_think.evaluation.counterfactual import (
    accuracy_by_budget,
    load_runs,
    per_example_accuracy,
    summarize_counterfactuals,
)
from when_to_think.evaluation.evaluate import run_evaluation
from when_to_think.evaluation.fixed_budget_eval import run_fixed_budget_sweep

__all__ = [
    "accuracy_by_budget",
    "load_runs",
    "per_example_accuracy",
    "run_evaluation",
    "run_fixed_budget_sweep",
    "summarize_counterfactuals",
]
