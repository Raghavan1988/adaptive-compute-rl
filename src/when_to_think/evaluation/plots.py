"""Plot the accuracy-vs-compute frontier from a counterfactual summary (M1).

The plot is generated entirely from the machine-readable summary (itself derived
from the run's JSONL), never from hand-typed numbers (AGENTS.md §18, §19). matplotlib
is imported lazily so the rest of the package does not depend on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_accuracy_vs_compute(
    summary: dict[str, Any],
    out_path: str | Path,
    *,
    x_key: str = "mean_reasoning_tokens",
    title: str = "Accuracy vs. compute (fixed budgets)",
) -> Path:
    """Render accuracy against mean compute per budget; return the written path."""
    import matplotlib

    matplotlib.use("Agg")  # headless; no display needed
    import matplotlib.pyplot as plt

    by_budget = summary["accuracy_by_budget"]
    budgets = sorted(by_budget, key=lambda b: int(b))
    xs = [by_budget[b][x_key] for b in budgets]
    ys = [by_budget[b]["accuracy"] for b in budgets]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    for b, x, y in zip(budgets, xs, ys, strict=True):
        ax.annotate(f"budget={b}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(f"Mean compute ({x_key})")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_oracle_frontier(
    summary: dict[str, Any],
    out_path: str | Path,
    *,
    x_key: str = "mean_reasoning_tokens",
    title: str = "Oracle vs. fixed budgets (accuracy-compute frontier)",
) -> Path:
    """Plot the oracle Pareto frontier against the fixed-budget baseline points (M2).

    Both series come from the machine-readable oracle summary (AGENTS.md §18): fixed
    budgets as labeled markers, the oracle frontier as a line. A dominating oracle sits
    up-and-to-the-left (more accuracy per token).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; no display needed
    import matplotlib.pyplot as plt

    fixed = sorted(summary["fixed_budget_points"], key=lambda p: p[x_key])
    frontier = sorted(summary["oracle_frontier"], key=lambda p: p[x_key])

    fig, ax = plt.subplots(figsize=(6, 4))

    fx = [p[x_key] for p in fixed]
    fy = [p["accuracy"] for p in fixed]
    ax.plot(fx, fy, marker="s", linestyle="--", color="tab:gray", label="fixed budget")
    for p in fixed:
        ax.annotate(
            f"b={p['budget']}", (p[x_key], p["accuracy"]),
            textcoords="offset points", xytext=(5, -10), fontsize=8, color="tab:gray",
        )

    ox = [p[x_key] for p in frontier]
    oy = [p["mean_accuracy"] for p in frontier]
    ax.plot(ox, oy, marker="o", color="tab:blue", label="oracle (upper bound)")

    ax.set_xlabel(f"Mean compute ({x_key})")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
