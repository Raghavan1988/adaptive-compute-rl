"""Thin entry point: summarize an M1 fixed-budget run and plot accuracy vs. compute.

Reads `<run-dir>/fixed_budget_runs.jsonl`, writes a machine-readable
`summary.json`, and renders `accuracy_vs_compute.png` — all derived from the result
file, never hand-typed (AGENTS.md §18). Prints the heterogeneity verdict (M1 exit,
Question 1).

Usage:
    python scripts/summarize_fixed_budgets.py --run-dir results/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when_to_think.evaluation import load_runs, summarize_counterfactuals
from when_to_think.evaluation.fixed_budget_eval import RUNS_FILENAME
from when_to_think.evaluation.plots import plot_accuracy_vs_compute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory from the sweep")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot (data only)")
    args = parser.parse_args()

    rows = load_runs(args.run_dir / RUNS_FILENAME)
    summary = summarize_counterfactuals(rows)

    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    voc = summary["value_of_compute"]
    print("\n=== Value of compute (Question 1) ===")
    print(f"  examples            : {summary['n_examples']}")
    print(f"  budgets             : {summary['budgets']}")
    print(f"  helped / hurt / unch: {voc['counts']['helped']} / {voc['counts']['hurt']} / "
          f"{voc['counts']['unchanged']}")
    print(f"  non-monotone (more tokens hurt): {voc['nonmonotone_examples']}")
    print(f"  HETEROGENEOUS value : {voc['heterogeneous']}")
    if not voc["heterogeneous"]:
        print("  ⚠️  Value of compute is NOT heterogeneous — the M1 hypothesis is weak; "
              "reconsider before M2+ (see PLAN.md).")

    if not args.no_plot:
        plot_path = plot_accuracy_vs_compute(summary, args.run_dir / "accuracy_vs_compute.png")
        print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
