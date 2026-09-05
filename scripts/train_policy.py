"""Thin entry point: train + evaluate the M4 STOP/CONTINUE policy (Q3/Q4).

Reads a trajectory run (generate it with `generate_trajectories.py`), trains one policy
per compute penalty in `reward.lambda_compute_sweep` on TRAIN, rolls each out greedily on
TEST, and writes the headline accuracy-vs-compute frontier (adaptive vs fixed vs oracle)
with bootstrap CIs and collapse diagnostics. All numbers derive from result files (§18).

Usage:
    python scripts/train_policy.py --run-dir results/<run_id> \
        --config configs/experiment/gsm8k_m4.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when_to_think.config import add_config_args, load_config_from_args
from when_to_think.policies.data import load_trajectories
from when_to_think.policies.experiment import run_policy_sweep
from when_to_think.policies.plots import plot_policy_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--run-dir", required=True, type=Path, help="Trajectory run directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip the plot (data only)")
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    trajectories = load_trajectories(args.run_dir)
    results = run_policy_sweep(trajectories, cfg)

    # Per-episode records go to their own JSONL; keep policy_results.json compact.
    pred_path = args.run_dir / "policy_episodes.jsonl"
    with open(pred_path, "w") as out:
        for block in results["per_lambda"].values():
            for ep in block.pop("episodes"):
                out.write(json.dumps({"lambda_compute": block["lambda_compute"], **ep}) + "\n")

    results_path = args.run_dir / "policy_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {pred_path}")

    print("\n=== Adaptive STOP/CONTINUE policy (M4, Q3/Q4) ===")
    print(f"  split sizes (trajectories): {results['split_sizes']}")
    print("  fixed-budget frontier (step: acc @ tokens):")
    for p in results["fixed_frontier"]:
        print(f"    step {p['step']}: acc={p['accuracy']:.3f} @ {p['mean_reasoning_tokens']:.1f}")
    print("  adaptive frontier (lambda: acc @ tokens [collapse?]):")
    for p, orc in zip(results["adaptive_frontier"], results["oracle_frontier"], strict=True):
        flag = "  COLLAPSED" if p["collapsed"] else ""
        print(f"    lam={p['lambda_compute']}: acc={p['accuracy']:.3f} "
              f"@ {p['mean_reasoning_tokens']:.1f}  (oracle {orc['accuracy']:.3f} "
              f"@ {orc['mean_reasoning_tokens']:.1f}){flag}")
    print(f"  best accuracy gain at matched compute: "
          f"{results['best_accuracy_gain_at_matched_compute']:+.3f}")
    print(f"  ADAPTIVE BEATS FIXED (matched compute): {results['adaptive_beats_fixed']}")
    if results["any_collapsed"]:
        print("  ⚠️  At least one lambda collapsed to ~always-STOP or ~always-CONTINUE — "
              "rising reward is not learning (AGENTS.md §16). Inspect before trusting.")
    if not results["adaptive_beats_fixed"]:
        print("  ⚠️  The policy does not beat fixed budgets at matched compute on this sweep — "
              "report honestly (PLAN.md M4 exit).")

    if not args.no_plot:
        plot_path = plot_policy_frontier(results, args.run_dir / "policy_frontier.png")
        print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
