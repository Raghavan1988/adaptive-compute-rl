"""Calibration analysis for probabilistic predictions (M5).

Applies to any predictor that outputs a probability — the M3 probe's P(continuing fixes
an incorrect answer) is the primary target. Reports the reliability curve (binned mean
prediction vs observed frequency), the Expected Calibration Error (ECE), and the Brier
score. A probe can be *decodable* (good AUROC) yet *miscalibrated* (probabilities not
matching frequencies); M5 reports both, and describes the probability as a decoded
signal, not the model's own belief (CLAUDE.md).

All numbers derive from the prediction arrays, so they are reproducible from the result
files (§18). Empty bins are dropped rather than counted as perfectly calibrated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from when_to_think.probes.metrics import brier_score


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Equal-width reliability bins over [0, 1]; empty bins are omitted."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Last bin is closed on the right so prob == 1.0 is included.
        in_bin = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bins.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": count,
            "mean_prediction": float(prob[in_bin].mean()),
            "fraction_positive": float(y_true[in_bin].mean()),
        })
    return bins


def expected_calibration_error(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> float:
    """ECE: count-weighted mean gap between predicted probability and observed frequency."""
    bins = reliability_curve(y_true, prob, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    return sum(
        b["count"] / total * abs(b["mean_prediction"] - b["fraction_positive"])
        for b in bins
    )


def calibration_summary(
    y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    """ECE + Brier + base rate + the reliability curve, as a JSON-friendly dict."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return {
        "n": int(len(y_true)),
        "n_bins": n_bins,
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "ece": expected_calibration_error(y_true, prob, n_bins=n_bins),
        "brier": brier_score(y_true, prob) if len(y_true) else float("nan"),
        "reliability_curve": reliability_curve(y_true, prob, n_bins=n_bins),
    }
