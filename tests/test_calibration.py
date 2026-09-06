"""Calibration metrics: ECE, reliability curve, Brier (M5)."""

import numpy as np

from when_to_think.evaluation.calibration import (
    calibration_summary,
    expected_calibration_error,
    reliability_curve,
)


def test_perfect_calibration_has_zero_ece():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.array([0.0] * 50 + [1.0] * 50)
    assert expected_calibration_error(y, prob) == 0.0
    summary = calibration_summary(y, prob)
    assert summary["ece"] == 0.0 and summary["brier"] == 0.0


def test_fully_miscalibrated():
    # Predict 0.5 everywhere but every label is 1 -> ECE = 0.5, Brier = 0.25.
    y = np.ones(20)
    prob = np.full(20, 0.5)
    assert expected_calibration_error(y, prob) == 0.5
    summary = calibration_summary(y, prob)
    assert summary["brier"] == 0.25


def test_reliability_curve_bins_and_positive_fraction():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    bins = reliability_curve(y, prob, n_bins=2)  # halves: [0,0.5) negatives, [0.5,1] positives
    assert len(bins) == 2 and all(b["count"] == 2 for b in bins)
    assert bins[0]["fraction_positive"] == 0.0
    assert bins[-1]["fraction_positive"] == 1.0


def test_prob_one_is_included_in_last_bin():
    # A prediction of exactly 1.0 must land in the closed final bin, not be dropped.
    bins = reliability_curve(np.array([1, 1]), np.array([1.0, 1.0]), n_bins=5)
    assert sum(b["count"] for b in bins) == 2
