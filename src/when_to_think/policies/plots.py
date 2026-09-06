"""M4 plots: the headline accuracy-vs-compute Pareto curve (adaptive vs fixed vs oracle).

Generated entirely from the machine-readable sweep results (§18). matplotlib is imported
lazily. The adaptive frontier carries bootstrap error bars on accuracy (§20).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_error_taxonomy(
    taxonomy: dict[str, Any],
    out_path: str | Path,
    *,
    title: str = "Policy error taxonomy",
) -> Path:
    """Bar chart of episode counts per error category (M5), colored by fault type."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["optimal_stop", "correct_but_late", "premature_stop",
             "overthought_to_wrong", "unavoidable_wrong"]
    # green = good, orange = wasteful-but-right, red = policy accuracy fault, gray = model limit.
    colors = {"optimal_stop": "tab:green", "correct_but_late": "tab:orange",
              "premature_stop": "tab:red", "overthought_to_wrong": "tab:purple",
              "unavoidable_wrong": "tab:gray"}
    counts = taxonomy["category_counts"]
    values = [counts[c] for c in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(order)), values, color=[colors[c] for c in order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.replace("_", "\n") for c in order], fontsize=8)
    ax.set_ylabel("Episodes")
    ax.set_title(
        f"{title}  (acc={taxonomy['policy_accuracy']:.3f}, "
        f"ceiling={taxonomy['ceiling_accuracy']:.3f}, "
        f"recoverable lost={taxonomy['recoverable_accuracy_lost']:.3f})"
    )
    ax.grid(True, axis="y", alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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
