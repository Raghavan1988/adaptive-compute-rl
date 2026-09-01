"""Analyze the counterfactual fixed-budget dataset (M1, Question 1).

Answers: does additional reasoning have *heterogeneous* value across examples? We
compute, per example, the accuracy at each budget (averaged over counterfactual
samples), then categorize how accuracy changes from the cheapest to the most
expensive budget:

- helped     : more compute raised accuracy
- hurt       : more compute LOWERED accuracy (a correct→wrong flip; the project
               makes no monotonicity assumption — AGENTS.md §4.5, these are kept)
- unchanged  : no change (further split into already-correct / never-correct)

"Heterogeneous" means examples do not all fall in one bucket: some benefit from
compute and others do not (or are hurt). If value is NOT heterogeneous, the M1 exit
says to flag it — the project hypothesis would be weak.

These functions are pure (operate on the per-example rows read from the run's
JSONL) so the aggregate numbers are reproducible from the result files, never typed
by hand (AGENTS.md §18).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_runs(runs_path: str | Path) -> list[dict[str, Any]]:
    """Read a `fixed_budget_runs.jsonl` file into a list of row dicts."""
    lines = Path(runs_path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def accuracy_by_budget(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Aggregate accuracy and mean compute per budget — the accuracy-vs-compute curve."""
    out: dict[int, dict[str, float]] = {}
    for budget in sorted({row["budget"] for row in rows}):
        at_budget = [row for row in rows if row["budget"] == budget]
        out[budget] = {
            "accuracy": _mean([1.0 if row["correct"] else 0.0 for row in at_budget]),
            "mean_reasoning_tokens": _mean([row["reasoning_tokens"] for row in at_budget]),
            "mean_total_tokens": _mean(
                [row.get("total_generated_tokens", row["reasoning_tokens"]) for row in at_budget]
            ),
            "n": len(at_budget),
        }
    return out


def per_example_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """Per-example accuracy at each budget, averaged over counterfactual samples."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        eid = row["example_id"]
        grouped.setdefault(eid, {}).setdefault(row["budget"], []).append(
            1.0 if row["correct"] else 0.0
        )
    return {
        eid: {budget: _mean(vals) for budget, vals in budgets.items()}
        for eid, budgets in grouped.items()
    }


def summarize_counterfactuals(
    rows: list[dict[str, Any]], *, unchanged_tol: float = 0.0
) -> dict[str, Any]:
    """Summarize heterogeneity of the value of compute across examples (Question 1)."""
    budgets = sorted({row["budget"] for row in rows})
    if len(budgets) < 2:
        raise ValueError("Need at least two distinct budgets to assess value of compute")
    per_ex = per_example_accuracy(rows)
    min_budget, max_budget = budgets[0], budgets[-1]

    helped = hurt = unchanged = 0
    unchanged_correct = unchanged_wrong = 0
    nonmonotone = 0
    deltas: list[float] = []

    for accs in per_ex.values():
        if min_budget not in accs or max_budget not in accs:
            continue  # example missing an endpoint budget; skip from this comparison
        delta = accs[max_budget] - accs[min_budget]
        deltas.append(delta)
        if delta > unchanged_tol:
            helped += 1
        elif delta < -unchanged_tol:
            hurt += 1
        else:
            unchanged += 1
            if accs[min_budget] >= 0.5:
                unchanged_correct += 1
            else:
                unchanged_wrong += 1

        # Non-monotone: accuracy drops for some larger budget (more tokens hurt).
        seq = [accs[b] for b in budgets if b in accs]
        if any(seq[j] < seq[i] for i in range(len(seq)) for j in range(i + 1, len(seq))):
            nonmonotone += 1

    n = len(deltas)
    categories_present = sum(1 for c in (helped, hurt, unchanged) if c > 0)

    return {
        "n_examples": n,
        "budgets": budgets,
        "accuracy_by_budget": accuracy_by_budget(rows),
        "value_of_compute": {
            "definition": "acc(max_budget) - acc(min_budget) per example",
            "min_budget": min_budget,
            "max_budget": max_budget,
            "mean_delta": _mean(deltas),
            "counts": {
                "helped": helped,
                "hurt": hurt,
                "unchanged": unchanged,
                "unchanged_correct": unchanged_correct,
                "unchanged_wrong": unchanged_wrong,
            },
            "fraction_helped": helped / n if n else math.nan,
            "fraction_hurt": hurt / n if n else math.nan,
            "fraction_unchanged": unchanged / n if n else math.nan,
            "nonmonotone_examples": nonmonotone,
            # Heterogeneous iff examples do not all fall in one bucket.
            "heterogeneous": categories_present >= 2,
        },
    }
