"""M4 plots: the headline accuracy-vs-compute Pareto curve (adaptive vs fixed vs oracle).

Generated entirely from the machine-readable sweep results (§18). matplotlib is imported
lazily. The adaptive frontier carries bootstrap error bars on accuracy (§20).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_policy_frontier(
    results: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "Adaptive policy vs fixed budgets (accuracy-compute frontier)",
) -> Path:
    """Plot fixed-budget, oracle, and adaptive frontiers on one accuracy-vs-compute axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = sorted(results["fixed_frontier"], key=lambda p: p["mean_reasoning_tokens"])
    oracle = sorted(results["oracle_frontier"], key=lambda p: p["mean_reasoning_tokens"])
    adaptive = sorted(results["adaptive_frontier"], key=lambda p: p["mean_reasoning_tokens"])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot([p["mean_reasoning_tokens"] for p in fixed], [p["accuracy"] for p in fixed],
            marker="s", linestyle="--", color="tab:gray", label="fixed budget")
    ax.plot([p["mean_reasoning_tokens"] for p in oracle], [p["accuracy"] for p in oracle],
            marker="^", linestyle=":", color="tab:green", label="oracle (upper bound)")

    ax_x = [p["mean_reasoning_tokens"] for p in adaptive]
    ax_y = [p["accuracy"] for p in adaptive]
    yerr_lo = [p["accuracy"] - p["accuracy_ci"][0] for p in adaptive]
    yerr_hi = [p["accuracy_ci"][1] - p["accuracy"] for p in adaptive]
    ax.errorbar(ax_x, ax_y, yerr=[yerr_lo, yerr_hi], marker="o", color="tab:blue",
                capsize=3, label="adaptive policy")
    for p in adaptive:
        if p["collapsed"]:
            ax.annotate("collapsed", (p["mean_reasoning_tokens"], p["accuracy"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color="tab:red")

    ax.set_xlabel("Mean reasoning tokens (compute)")
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
