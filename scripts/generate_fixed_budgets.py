"""Thin entry point: generate the M1 counterfactual fixed-budget dataset.

Runs the same test examples at every budget in `generation.fixed_budgets`, with
`generation.num_samples` counterfactual samples each, and writes per-run rows +
decision-point hidden states to `results/<run_id>/`.

Usage:
    python scripts/generate_fixed_budgets.py --config configs/experiment/gsm8k_smoke.yaml
    python scripts/generate_fixed_budgets.py --config configs/experiment/gsm8k_smoke.yaml \
        --set generation.fixed_budgets='[0, 128, 256, 512]' --set generation.num_samples=3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from when_to_think.config import add_config_args, load_config_from_args
from when_to_think.evaluation import run_fixed_budget_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    run_dir = run_fixed_budget_sweep(cfg, repo_dir=Path(__file__).resolve().parent.parent)

    print(f"Fixed-budget sweep complete. Results written to: {run_dir}")
    print(f"  per-run rows: {run_dir / 'fixed_budget_runs.jsonl'}")
    print(f"  hidden states: {run_dir / 'hidden_states'}")
    print("Next: python scripts/summarize_fixed_budgets.py --run-dir", run_dir)


if __name__ == "__main__":
    main()
