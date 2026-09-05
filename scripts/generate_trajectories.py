"""Thin entry point: generate M4 STOP/CONTINUE trajectories with the SLM.

One coherent reasoning rollout per example, checkpointed every `decision_interval`
tokens (hidden state + provisional answer at each). Writes them under
`results/<run_id>/trajectories/`. Defaults to all splits — the policy needs TRAIN data.

Usage:
    python scripts/generate_trajectories.py --config configs/experiment/gsm8k_m4.yaml
    python scripts/generate_trajectories.py --config configs/experiment/gsm8k_m4.yaml \
        --splits train,val,test --set data.max_train_examples=300
"""

from __future__ import annotations

import argparse
from pathlib import Path

from when_to_think.config import add_config_args, load_config_from_args
from when_to_think.policies.generate import run_trajectory_generation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--splits", default="train,val,test",
        help="Comma-separated splits to generate (default: train,val,test).",
    )
    args = parser.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    cfg = load_config_from_args(args)
    run_dir = run_trajectory_generation(
        cfg, repo_dir=Path(__file__).resolve().parent.parent, splits=splits
    )

    print(f"Trajectory generation complete. Results written to: {run_dir}")
    print(f"  splits: {list(splits)}")
    print(f"  trajectories: {run_dir / 'trajectories'}")
    print("Next: python scripts/train_policy.py --run-dir", run_dir,
          "--config configs/experiment/gsm8k_m4.yaml")


if __name__ == "__main__":
    main()
