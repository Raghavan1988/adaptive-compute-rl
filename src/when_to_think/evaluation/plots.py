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
    for b, x, y in zip(budgets, xs, ys):
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
