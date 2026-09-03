"""Pure-numpy metrics for probe evaluation (no sklearn dependency, AGENTS.md §23).

Split by target type. Binary classification (the "continuing fixes an incorrect
answer" target) is scored with AUROC / Brier / accuracy; the value-of-compute
regression target (Δ = P(correct|continue) − P(correct|stop)) is scored with
MSE / MAE / R² / correlation. All are computed from arrays only, so every reported
number is reproducible from the prediction files (AGENTS.md §18).

Degenerate inputs (a single class present, zero-variance targets) return ``nan``
rather than a misleading number — a probe cannot be credited with discrimination on
data that has none to find.
"""

from __future__ import annotations

import math

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann–Whitney rank statistic (ties averaged).

    Returns ``nan`` if only one class is present (AUROC is undefined — there is no
    positive/negative pair to order).
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average ranks within tied score groups so ties contribute 0.5, not a coin flip.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    group_rank_sum = np.zeros(len(counts))
    np.add.at(group_rank_sum, inv, ranks)
    ranks = (group_rank_sum / counts)[inv]
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 label."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction correct when thresholding predicted probability at ``threshold``."""
    y_true = np.asarray(y_true, dtype=float)
    pred = (np.asarray(prob, dtype=float) >= threshold).astype(float)
    return float(np.mean(pred == y_true))


def base_rate(y_true: np.ndarray) -> float:
    """Fraction of positive labels — the prior a constant classifier would predict."""
    return float(np.mean(np.asarray(y_true, dtype=float)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; ``nan`` when the target has zero variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return math.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation; ``nan`` if either input has zero variance."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.std() <= 0 or y_pred.std() <= 0:
        return math.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks)."""

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        return ranks

    return pearson(_rank(np.asarray(y_true)), _rank(np.asarray(y_pred)))


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """All classification metrics for the binary target, as a JSON-friendly dict."""
    return {
        "auroc": auroc(y_true, prob),
        "brier": brier_score(y_true, prob),
        "accuracy": accuracy(y_true, prob, threshold),
        "base_rate": base_rate(y_true),
        "n": int(len(y_true)),
        "n_positive": int(np.asarray(y_true).sum()),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """All regression metrics for the value-of-compute target, as a dict."""
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "n": int(len(y_true)),
    }
