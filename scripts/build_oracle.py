"""Thin entry point: build the M2 oracle allocation from an M1 fixed-budget run.

Reads ``<run-dir>/fixed_budget_runs.jsonl``, computes the omniscient per-example budget
allocation and its accuracy-compute Pareto frontier, writes machine-readable
``oracle_summary.json`` and per-example ``oracle_allocation.jsonl``, renders
``oracle_frontier.png``, and prints the M2 exit verdict (does the oracle beat fixed
budgets at matched accuracy?). All numbers are derived from the result file, never
hand-typed (AGENTS.md §18).

Usage:
    python scripts/build_oracle.py --run-dir results/<run_id>
    python scripts/build_oracle.py --run-dir results/<run_id> --lambdas 0,1e-4,5e-4,1e-3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when_to_think.evaluation import (
    load_runs,
    oracle_allocation,
    summarize_oracle,
)
from when_to_think.evaluation.fixed_budget_eval import RUNS_FILENAME
from when_to_think.evaluation.plots import plot_oracle_frontier


def _parse_lambdas(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="M1 run directory")
    parser.add_argument(
        "--lambdas",
        default=None,
        help="Comma-separated compute penalties for the frontier "
        "(default: exact data-derived breakpoints)",
    )
    parser.add_argument(
        "--accuracy-tol",
        type=float,
        default=0.0,
        help="Slack when matching the oracle to a fixed budget's accuracy (default 0)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot (data only)")
    args = parser.parse_args()

    rows = load_runs(args.run_dir / RUNS_FILENAME)
    lambdas = _parse_lambdas(args.lambdas)
    summary = summarize_oracle(rows, lambdas=lambdas, accuracy_tol=args.accuracy_tol)

    summary_path = args.run_dir / "oracle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    # Per-example allocation of the accuracy-maximizing oracle (lambda = 0): a
    # machine-readable record of which budget the oracle assigned to each example.
    acc_max = oracle_allocation(rows, 0.0)
    alloc_path = args.run_dir / "oracle_allocation.jsonl"
    with open(alloc_path, "w") as out:
        for example_id, budget in sorted(acc_max["budget_choices"].items()):
            out.write(json.dumps({"example_id": example_id, "oracle_budget": budget}) + "\n")
    print(f"Wrote {alloc_path}")

    amo = summary["accuracy_max_oracle"]
    print("\n=== Oracle allocation (M2 upper bound) ===")
    print(f"  examples                 : {summary['n_examples']}")
    print(f"  budgets                  : {summary['budgets']}")
    print("  fixed-budget baseline    :")
    for fp in summary["fixed_budget_points"]:
        print(f"    budget {fp['budget']:>4}: acc={fp['accuracy']:.3f} "
              f"@ {fp['mean_reasoning_tokens']:.1f} tokens")
    print(f"  accuracy-max oracle      : acc={amo['mean_accuracy']:.3f} "
          f"@ {amo['mean_reasoning_tokens']:.1f} tokens  (budget mix {amo['budget_histogram']})")
    best_fixed = summary["best_fixed_budget"]
    saved = summary["compute_saved_vs_best_fixed"]
    saved_frac = summary["compute_saved_fraction_vs_best_fixed"]
    print(f"  vs best fixed (b={best_fixed}) at matched accuracy: "
          f"saves {saved:.1f} tokens ({saved_frac:+.1%})")
    print(f"  max compute saved (any matched budget): {summary['max_compute_saved_fraction']:+.1%}")
    print(f"  ORACLE BEATS FIXED       : {summary['oracle_dominates_fixed']}")
    if not summary["oracle_dominates_fixed"]:
        print("  ⚠️  The oracle does NOT beat fixed budgets at matched accuracy — per the M2 "
              "exit, stop and reconsider before building probes or RL (see PLAN.md).")

    if not args.no_plot:
        plot_path = plot_oracle_frontier(summary, args.run_dir / "oracle_frontier.png")
        print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
