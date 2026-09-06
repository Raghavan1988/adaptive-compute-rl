"""Thin entry point: M5 analysis over a run directory's result files.

Runs whichever analyses the run supports, deriving every number from the machine-readable
outputs (§18), never by hand:

- ``probe_predictions.jsonl`` (M3)  -> calibration of the probe's ``fixes_incorrect``
  probability: ECE, Brier, reliability diagram.
- ``policy_episodes.jsonl`` + ``trajectories/`` (M4) -> the STOP/CONTINUE error taxonomy
  (per lambda): where the policy makes expensive mistakes, and the recoverable accuracy
  it leaves on the table.

Usage:
    python scripts/analyze.py --run-dir results/<run_id>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from when_to_think.evaluation.calibration import calibration_summary
from when_to_think.evaluation.plots import plot_reliability_diagram
from when_to_think.policies.data import load_trajectories
from when_to_think.policies.error_taxonomy import summarize_error_taxonomy
from when_to_think.policies.plots import plot_error_taxonomy


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze_probe_calibration(run_dir: Path, *, no_plot: bool) -> bool:
    pred_path = run_dir / "probe_predictions.jsonl"
    if not pred_path.exists():
        return False
    rows = [r for r in _read_jsonl(pred_path) if r.get("target") == "fixes_incorrect"]
    if not rows:
        print("  (no fixes_incorrect predictions to calibrate)")
        return False
    y = [int(r["fixes_incorrect"]) for r in rows]
    prob = [float(r["probe_prediction"]) for r in rows]
    summary = calibration_summary(y, prob)

    out = run_dir / "calibration_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"  probe calibration: ECE={summary['ece']:.3f}  Brier={summary['brier']:.3f}  "
          f"base_rate={summary['base_rate']:.3f}  (n={summary['n']})")
    if not no_plot:
        p = plot_reliability_diagram(summary, run_dir / "reliability_diagram.png",
                                     title="Probe calibration (fixes_incorrect)")
        print(f"Wrote {p}")
    return True


def analyze_policy_errors(run_dir: Path, *, no_plot: bool) -> bool:
    ep_path = run_dir / "policy_episodes.jsonl"
    if not ep_path.exists() or not (run_dir / "trajectories").exists():
        return False
    trajectories = load_trajectories(run_dir)
    episodes = _read_jsonl(ep_path)

    by_lambda: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_lambda[str(ep.get("lambda_compute", "na"))].append(ep)

    summaries = {}
    ep_out = run_dir / "error_taxonomy_episodes.jsonl"
    with open(ep_out, "w") as handle:
        for lam, eps in sorted(by_lambda.items()):
            summary = summarize_error_taxonomy(trajectories, eps)
            for row in summary.pop("per_episode"):
                handle.write(json.dumps({"lambda_compute": lam, **row}) + "\n")
            summaries[lam] = summary

    out = run_dir / "error_taxonomy.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {ep_out}")
    for lam, s in sorted(summaries.items()):
        print(f"  lambda={lam}: acc={s['policy_accuracy']:.3f} "
              f"(ceiling {s['ceiling_accuracy']:.3f}), "
              f"recoverable lost={s['recoverable_accuracy_lost']:.3f}, "
              f"wasted tokens/ep={s['mean_wasted_tokens']:.1f}")
        print(f"    categories: {s['category_counts']}")
        if not no_plot:
            plot_error_taxonomy(s, run_dir / f"error_taxonomy_lambda_{lam}.png",
                                title=f"Policy error taxonomy (lambda={lam})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to analyze")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots (data only)")
    args = parser.parse_args()

    print(f"=== M5 analysis: {args.run_dir} ===")
    did_probe = analyze_probe_calibration(args.run_dir, no_plot=args.no_plot)
    did_policy = analyze_policy_errors(args.run_dir, no_plot=args.no_plot)
    if not (did_probe or did_policy):
        raise SystemExit(
            "Nothing to analyze: need probe_predictions.jsonl (M3) or "
            "policy_episodes.jsonl + trajectories/ (M4) in the run directory."
        )


if __name__ == "__main__":
    main()
