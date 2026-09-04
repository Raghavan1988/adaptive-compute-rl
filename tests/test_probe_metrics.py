"""Probe metrics: AUROC (incl. ties + degenerate), Brier, R², correlation."""

import math

import numpy as np

from when_to_think.probes import metrics as M


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # Perfectly tied scores -> 0.5 (ties averaged, no spurious discrimination).
    assert M.auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auroc_single_class_is_nan():
    assert math.isnan(M.auroc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))


def test_auroc_matches_probability_interpretation():
    # AUROC = P(score_pos > score_neg) over all pairs; here 3/4 pairs correctly ordered.
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.5, 0.9])  # one misordered pair (0.6 neg > 0.5 pos)
    assert M.auroc(y, scores) == 0.75


def test_brier_and_accuracy():
    y = np.array([1.0, 0.0])
    assert M.brier_score(y, np.array([1.0, 0.0])) == 0.0
    assert M.accuracy(y, np.array([0.9, 0.1])) == 1.0
    assert M.accuracy(y, np.array([0.1, 0.9])) == 0.0


def test_r2_perfect_and_zero_variance():
    y = np.array([1.0, 2.0, 3.0])
    assert M.r2(y, y) == 1.0
    assert math.isnan(M.r2(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0])))


def test_regression_and_binary_bundles():
    reg = M.regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 2.9]))
    assert set(reg) == {"mse", "mae", "r2", "pearson", "spearman", "n"}
    binm = M.binary_metrics(np.array([0, 1, 1]), np.array([0.2, 0.7, 0.6]))
    assert binm["n"] == 3 and binm["n_positive"] == 2
