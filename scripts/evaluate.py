"""Thin entry point: run the M0 single-pass fixed-budget evaluation from config.

Usage:
    python scripts/evaluate.py --config configs/experiment/gsm8k_smoke.yaml
    python scripts/evaluate.py --config configs/experiment/gsm8k_smoke.yaml \
        --set data.max_test_examples=20 --set generation.max_reasoning_budget=256

All experimental quantities come from the config (AGENTS.md §21); this script only
parses arguments and delegates to `when_to_think`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from when_to_think.config import add_config_args, load_config_from_args
from when_to_think.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    run_dir = run_evaluation(cfg, repo_dir=Path(__file__).resolve().parent.parent)

    print(f"Run complete. Results written to: {run_dir}")
    print(f"  per-example: {run_dir / 'eval.jsonl'}")
    print(f"  hidden states: {run_dir / 'hidden_states'}")
    print(f"  run record: {run_dir / 'run_record.json'}")


if __name__ == "__main__":
    main()
