"""Thin entry point: train + evaluate the M3 value-of-compute probe.

Reads a fixed-budget run that covers TRAIN/VAL/TEST (generate it with
``generate_fixed_budgets.py --splits train,val,test``), fits the probe on train,
selects the layer and regularization on val, and scores test exactly once for the
probe and the baselines. Writes machine-readable ``probe_results.json`` and per-example
``probe_predictions.jsonl``, renders layer-wise and probe-vs-baseline plots, and prints
the Question 2 verdict — described as *decodability*, never mechanism (CLAUDE.md).

Probe hyperparameters come from the ``probe:`` section of ``--config`` (experimental
quantities live in config, not in the script). With no ``--config`` the ProbeConfig
defaults are used.

Usage:
    python scripts/train_probe.py --run-dir results/<run_id> --config configs/experiment/gsm8k_m3.yaml
    python scripts/train_probe.py --run-dir results/<run_id>           # ProbeConfig defaults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from when_to_think.config import ProbeConfig, load_config
from when_to_think.evaluation import load_runs
from when_to_think.evaluation.fixed_budget_eval import RUNS_FILENAME
from when_to_think.probes.plots import plot_layerwise_val, plot_probe_vs_baselines
from when_to_think.probes.train import train_probe
from when_to_think.representations import HiddenStateReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Fixed-budget run directory")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Experiment YAML supplying the probe: section (default: ProbeConfig defaults)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plots (data only)")
    args = parser.parse_args()

    probe_cfg = load_config(args.config).probe if args.config else ProbeConfig()

    rows = load_runs(args.run_dir / RUNS_FILENAME)
    reader = HiddenStateReader(args.run_dir / "hidden_states")

    missing = [layer for layer in probe_cfg.layers if layer not in reader.layers]
    if missing:
        raise SystemExit(
            f"probe.layers {missing} were not stored in this run (stored: {reader.layers}). "
            "Re-run the sweep with representation.layers covering the layers you want to probe."
        )

    results = train_probe(rows, reader, probe_cfg)

    # Keep results.json compact: per-example predictions go to their own JSONL.
    pred_path = args.run_dir / "probe_predictions.jsonl"
    with open(pred_path, "w") as out:
        for target, tr in results["targets"].items():
            for pred in tr.pop("predictions"):
                out.write(json.dumps({"target": target, **pred}) + "\n")

    results_path = args.run_dir / "probe_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {pred_path}")

    print("\n=== Value-of-compute probe (M3, Question 2) ===")
    print(f"  stored layers          : {results['stored_layers']}")
    for target, tr in results["targets"].items():
        metric = "R²" if target == "value_of_compute" else "AUROC"
        key = "r2" if target == "value_of_compute" else "auroc"
        probe_v = tr["hidden_state_probe"]["test"].get(key)
        base_v = tr["input_only_baseline"]["test"].get(key)
        n = tr["n_instances"]
        print(f"\n  target: {target}  ({tr['definition']})")
        print(f"    instances (tr/va/te): {n['train']} / {n['val']} / {n['test']}")
        print(f"    selected layer / alpha: {tr['selected_layer']} / {tr['selected_alpha']}")
        print(f"    hidden-state probe  {metric}: {probe_v:.3f}")
        print(f"    input-only baseline {metric}: {base_v:.3f}")
        print(f"    HIDDEN STATE BEATS INPUT BASELINE: {tr['hidden_state_beats_input_baseline']}"
              f"  (margin {tr['decodability_margin']:+.3f})")
        if not tr["hidden_state_beats_input_baseline"]:
            print("    ⚠️  Value of compute is not more decodable from the hidden state than "
                  "from the input alone for this target — report honestly (PLAN.md M3).")
    print("\n  (entropy / verbalized-confidence baselines not yet collected — see "
          "probes/baselines.py; Q2 is answered vs input-only for now.)")

    if not args.no_plot:
        for target, tr in results["targets"].items():
            lp = plot_layerwise_val(tr, args.run_dir / f"probe_layerwise_{target}.png")
            bp = plot_probe_vs_baselines(tr, args.run_dir / f"probe_vs_baselines_{target}.png")
            print(f"Wrote {lp}\nWrote {bp}")


if __name__ == "__main__":
    main()
