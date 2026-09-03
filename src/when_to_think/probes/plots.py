"""Plots for the M3 probe, generated from the machine-readable results (AGENTS.md §18).

matplotlib is imported lazily so importing the probes package never pulls it in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_layerwise_val(
    target_result: dict[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Bar chart of val selection score per candidate layer (the layer-wise analysis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layerwise = target_result["layerwise_val"]
    layers = sorted(layerwise, key=lambda s: int(s))
    scores = [layerwise[layer]["selection_score"] for layer in layers]

    metric = "R²" if target_result["target"] == "value_of_compute" else "AUROC"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(layers)), scores, color="tab:blue")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Hidden-state layer")
    ax.set_ylabel(f"Val selection score ({metric})")
    ax.set_title(title or f"Layer-wise decodability: {target_result['target']}")
    ax.axhline(0.5 if metric == "AUROC" else 0.0, color="tab:gray", linestyle="--",
               label="chance / no signal")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_probe_vs_baselines(
    target_result: dict[str, Any],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Test-set bar chart: hidden-state probe vs input-only baseline vs prior."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_reg = target_result["target"] == "value_of_compute"
    key = "r2" if is_reg else "auroc"
    metric = "R²" if is_reg else "AUROC"
    names = ["hidden-state\nprobe", "input-only\nbaseline", "prior"]
    values = [
        target_result["hidden_state_probe"]["test"].get(key, float("nan")),
        target_result["input_only_baseline"]["test"].get(key, float("nan")),
        0.5 if not is_reg else 0.0,  # prior: AUROC = 0.5, R² = 0 by construction
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:gray"])
    ax.set_ylabel(f"Test {metric}")
    ax.set_title(title or f"Probe vs baselines (test): {target_result['target']}")
    ax.grid(True, axis="y", alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
