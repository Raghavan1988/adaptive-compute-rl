"""Oracle allocation: the upper bound on adaptive compute allocation (M2).

The oracle is an *omniscient* per-example budget allocator. Given the counterfactual
fixed-budget dataset (M1), it sees each example's outcome at every budget and, for a
compute penalty ``lambda``, picks per example the budget that maximizes

    value(example, budget) = accuracy(example, budget) - lambda * tokens(example, budget)

ties broken toward *fewer* tokens. Sweeping ``lambda`` from 0 to infinity traces the
oracle's accuracy-vs-compute Pareto frontier:

- lambda = 0      -> accuracy-maximizing oracle: the cheapest budget achieving each
                     example's best accuracy ("cheapest budget that is correct" when
                     accuracy is binary, i.e. ``num_samples == 1``).
- lambda -> infinity -> the cheapest budget everywhere (budget 0).

The oracle is **not a deployable method** — it peeks at correctness, so it cannot be
compared to a real policy as if it were one. It is the *ceiling*: if even an omniscient
router cannot meaningfully beat fixed budgets, an adaptive policy cannot either, and the
M2 exit says to stop and reconsider before building probes or RL.

Design notes tied to research invariants:
- **Matched comparison (AGENTS.md §4.1).** Savings are reported at *matched accuracy*:
  for each fixed-budget baseline we find the cheapest oracle allocation reaching the same
  accuracy, and report the compute it saves — never oracle-vs-fixed at different accuracy.
- **No monotonicity assumption (AGENTS.md §4.5).** ``accuracy(example, budget)`` can fall
  as budget rises; the argmax handles that directly (a bigger budget that hurts is simply
  never chosen). Nothing here assumes more compute helps.
- **Purity (AGENTS.md §18).** These functions operate on the per-example rows read from
  the run's JSONL, so every number is reproducible from the result files.

Multiple counterfactual samples are aggregated to expected values: ``accuracy`` is the
mean correctness over samples (an estimate of P(correct | example, budget)) and ``tokens``
is the mean ``reasoning_tokens`` over samples. With ``num_samples == 1`` these reduce to
the raw per-run values.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from when_to_think.evaluation.counterfactual import accuracy_by_budget

# Values within this tolerance are treated as equal, so floating-point noise does not
# spuriously flip an oracle choice or drop a Pareto vertex.
_EPS = 1e-9


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def per_example_budget_stats(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, float]]]:
    """Per example, per budget: expected accuracy and mean reasoning tokens over samples."""
    grouped: dict[str, dict[int, dict[str, list[float]]]] = {}
    for row in rows:
        eid = row["example_id"]
        budget = row["budget"]
        cell = grouped.setdefault(eid, {}).setdefault(budget, {"correct": [], "tokens": []})
        cell["correct"].append(1.0 if row["correct"] else 0.0)
        cell["tokens"].append(float(row["reasoning_tokens"]))
    return {
        eid: {
            budget: {
                "accuracy": _mean(cell["correct"]),
                "mean_reasoning_tokens": _mean(cell["tokens"]),
                "n": len(cell["correct"]),
            }
            for budget, cell in by_budget.items()
        }
        for eid, by_budget in grouped.items()
    }


def oracle_allocation(rows: list[dict[str, Any]], lambda_compute: float) -> dict[str, Any]:
    """Oracle choice per example at one compute penalty; return choices + aggregates.

    Per example, choose the budget maximizing ``accuracy - lambda * tokens``. Budgets are
    scanned in ascending order and a later (larger) budget replaces the incumbent only if
    it is *strictly* better, so ties are broken toward the cheaper budget.
    """
    if lambda_compute < 0:
        raise ValueError("lambda_compute must be non-negative")
    stats = per_example_budget_stats(rows)

    choices: dict[str, int] = {}
    accuracies: list[float] = []
    computes: list[float] = []
    for eid, by_budget in stats.items():
        best_budget: int | None = None
        best_value = -math.inf
        for budget in sorted(by_budget):  # ascending => cheapest wins ties
            cell = by_budget[budget]
            value = cell["accuracy"] - lambda_compute * cell["mean_reasoning_tokens"]
            if value > best_value + _EPS:
                best_value = value
                best_budget = budget
        choices[eid] = best_budget
        accuracies.append(by_budget[best_budget]["accuracy"])
        computes.append(by_budget[best_budget]["mean_reasoning_tokens"])

    histogram = Counter(choices.values())
    return {
        "lambda_compute": lambda_compute,
        "n_examples": len(choices),
        "mean_accuracy": _mean(accuracies),
        "mean_reasoning_tokens": _mean(computes),
        "budget_choices": choices,
        # JSON-friendly {budget: count}, sorted by budget for stable output.
        "budget_histogram": {int(b): int(histogram[b]) for b in sorted(histogram)},
    }


def _candidate_lambdas(rows: list[dict[str, Any]]) -> list[float]:
    """Breakpoint lambdas at which some example switches its argmax, plus segment probes.

    An example switches the budget it prefers exactly at a lambda equal to the slope
    ``(delta accuracy) / (delta tokens)`` between two of its budgets. Evaluating the oracle
    at every such breakpoint AND at midpoints between consecutive breakpoints visits every
    segment of the piecewise-constant frontier, so the exact frontier vertices are found
    without depending on an arbitrary lambda grid resolution.
    """
    stats = per_example_budget_stats(rows)
    slopes: set[float] = set()
    for by_budget in stats.values():
        budgets = sorted(by_budget)
        for i in range(len(budgets)):
            for j in range(i + 1, len(budgets)):
                lo, hi = by_budget[budgets[i]], by_budget[budgets[j]]
                dt = hi["mean_reasoning_tokens"] - lo["mean_reasoning_tokens"]
                da = hi["accuracy"] - lo["accuracy"]
                if dt > _EPS and da > _EPS:  # paying more only switches when it also helps
                    slopes.add(da / dt)

    breakpoints = sorted({0.0} | slopes)
    probes: set[float] = set(breakpoints)
    for a, b in zip(breakpoints, breakpoints[1:], strict=False):
        probes.add((a + b) / 2.0)
    # A lambda beyond the largest breakpoint forces the cheapest budget everywhere.
    probes.add((breakpoints[-1] + 1.0) if breakpoints else 1.0)
    return sorted(probes)


def _pareto_reduce(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only Pareto-optimal (min compute, max accuracy) points; sort by compute.

    A point is dominated if another has accuracy >= it AND compute <= it, with at least
    one strict — that other point is at least as accurate for no more compute.
    """
    kept: list[dict[str, Any]] = []
    for p in points:
        pc, pa = p["mean_reasoning_tokens"], p["mean_accuracy"]
        dominated = any(
            q["mean_accuracy"] >= pa - _EPS
            and q["mean_reasoning_tokens"] <= pc + _EPS
            and (
                q["mean_accuracy"] > pa + _EPS
                or q["mean_reasoning_tokens"] < pc - _EPS
            )
            for q in points
            if q is not p
        )
        if not dominated:
            kept.append(p)

    # Deduplicate points that coincide in (compute, accuracy) — same frontier vertex
    # reached by different lambdas — keeping the one found at the smallest lambda.
    unique: list[dict[str, Any]] = []
    for p in sorted(kept, key=lambda x: (x["mean_reasoning_tokens"], -x["mean_accuracy"])):
        if unique and (
            abs(p["mean_reasoning_tokens"] - unique[-1]["mean_reasoning_tokens"]) <= _EPS
            and abs(p["mean_accuracy"] - unique[-1]["mean_accuracy"]) <= _EPS
        ):
            continue
        unique.append(p)
    return unique


def oracle_frontier(
    rows: list[dict[str, Any]], lambdas: list[float] | None = None
) -> list[dict[str, Any]]:
    """The oracle accuracy-vs-compute Pareto frontier.

    ``lambdas`` defaults to the exact data-derived breakpoints (see ``_candidate_lambdas``);
    pass an explicit list to force a specific penalty grid. Each frontier point drops the
    per-example ``budget_choices`` (kept only in ``oracle_allocation``) to stay compact.
    """
    grid = _candidate_lambdas(rows) if lambdas is None else sorted(set(lambdas))
    points = []
    for lam in grid:
        alloc = oracle_allocation(rows, lam)
        points.append(
            {
                "lambda_compute": alloc["lambda_compute"],
                "mean_accuracy": alloc["mean_accuracy"],
                "mean_reasoning_tokens": alloc["mean_reasoning_tokens"],
                "n_examples": alloc["n_examples"],
                "budget_histogram": alloc["budget_histogram"],
            }
        )
    return _pareto_reduce(points)


def fixed_budget_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed-budget baseline points (one per budget): accuracy and mean compute."""
    by_budget = accuracy_by_budget(rows)
    return [
        {
            "budget": budget,
            "accuracy": by_budget[budget]["accuracy"],
            "mean_reasoning_tokens": by_budget[budget]["mean_reasoning_tokens"],
            "n": by_budget[budget]["n"],
        }
        for budget in sorted(by_budget)
    ]


def compute_savings_at_matched_accuracy(
    fixed_points: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    *,
    accuracy_tol: float = 0.0,
) -> list[dict[str, Any]]:
    """For each fixed-budget baseline, the cheapest oracle allocation of equal accuracy.

    Matched-accuracy comparison (AGENTS.md §4.1): we never credit the oracle for being more
    accurate at more compute. For each fixed point we find the minimum-compute oracle point
    whose accuracy is at least the baseline's (within ``accuracy_tol``) and report the
    compute it saves. The oracle's accuracy-max point is >= every fixed budget's accuracy,
    so a match always exists.
    """
    out: list[dict[str, Any]] = []
    for fp in fixed_points:
        target = fp["accuracy"] - accuracy_tol
        candidates = [p for p in frontier if p["mean_accuracy"] >= target - _EPS]
        if not candidates:
            out.append({
                "fixed_budget": fp["budget"],
                "fixed_accuracy": fp["accuracy"],
                "fixed_compute": fp["mean_reasoning_tokens"],
                "oracle_compute": None,
                "oracle_accuracy": None,
                "compute_saved": None,
                "compute_saved_fraction": None,
            })
            continue
        best = min(candidates, key=lambda p: p["mean_reasoning_tokens"])
        saved = fp["mean_reasoning_tokens"] - best["mean_reasoning_tokens"]
        denom = fp["mean_reasoning_tokens"]
        out.append({
            "fixed_budget": fp["budget"],
            "fixed_accuracy": fp["accuracy"],
            "fixed_compute": fp["mean_reasoning_tokens"],
            "oracle_compute": best["mean_reasoning_tokens"],
            "oracle_accuracy": best["mean_accuracy"],
            "compute_saved": saved,
            "compute_saved_fraction": (saved / denom) if denom > _EPS else math.nan,
        })
    return out


def summarize_oracle(
    rows: list[dict[str, Any]],
    *,
    lambdas: list[float] | None = None,
    accuracy_tol: float = 0.0,
) -> dict[str, Any]:
    """Assemble the machine-readable M2 oracle summary (aggregates + frontier + savings)."""
    budgets = sorted({row["budget"] for row in rows})
    if len(budgets) < 2:
        raise ValueError("Need at least two distinct budgets to build an oracle")

    fixed = fixed_budget_points(rows)
    frontier = oracle_frontier(rows, lambdas)
    savings = compute_savings_at_matched_accuracy(fixed, frontier, accuracy_tol=accuracy_tol)

    acc_max = oracle_allocation(rows, 0.0)  # accuracy-maximizing oracle (lambda = 0)

    # Headline "beats fixed budgets?": at the accuracy of the best fixed budget, does the
    # oracle reach it for strictly less compute? Uses the matched-accuracy savings so the
    # comparison is fair. This is the M2 exit gate.
    best_fixed = max(fixed, key=lambda p: (p["accuracy"], -p["mean_reasoning_tokens"]))
    matched_best = next(s for s in savings if s["fixed_budget"] == best_fixed["budget"])
    saved_fractions = [
        s["compute_saved_fraction"]
        for s in savings
        if s["compute_saved_fraction"] is not None and not math.isnan(s["compute_saved_fraction"])
    ]
    max_saved_fraction = max(saved_fractions, default=0.0)
    oracle_dominates_fixed = bool(
        matched_best["compute_saved"] is not None and matched_best["compute_saved"] > _EPS
    )

    return {
        "n_examples": acc_max["n_examples"],
        "budgets": budgets,
        "fixed_budget_points": fixed,
        "oracle_frontier": frontier,
        "accuracy_max_oracle": {
            "lambda_compute": acc_max["lambda_compute"],
            "mean_accuracy": acc_max["mean_accuracy"],
            "mean_reasoning_tokens": acc_max["mean_reasoning_tokens"],
            "budget_histogram": acc_max["budget_histogram"],
        },
        "savings_at_matched_accuracy": savings,
        "best_fixed_budget": best_fixed["budget"],
        "compute_saved_vs_best_fixed": matched_best["compute_saved"],
        "compute_saved_fraction_vs_best_fixed": matched_best["compute_saved_fraction"],
        "max_compute_saved_fraction": max_saved_fraction,
        # M2 exit gate: does the omniscient allocator beat fixed budgets at matched accuracy?
        "oracle_dominates_fixed": oracle_dominates_fixed,
    }
