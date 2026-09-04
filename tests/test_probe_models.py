"""Probe models: ridge recovers a linear signal, logistic separates, scaler is train-only."""

import numpy as np

from when_to_think.probes.models import LogisticProbe, RidgeProbe, StandardScaler


def test_scaler_uses_fit_statistics_only():
    train = np.array([[0.0], [2.0], [4.0]])  # mean 2, std ~1.633
    scaler = StandardScaler().fit(train)
    assert scaler.mean_[0] == 2.0
    # Transform of a *different* array uses the TRAIN mean/std (no re-fit): 2.0 -> 0.
    out = scaler.transform(np.array([[2.0]]))
    assert out[0, 0] == 0.0


def test_scaler_handles_constant_feature():
    scaler = StandardScaler().fit(np.array([[5.0], [5.0]]))
    # No divide-by-zero; a constant column maps to 0 after centering.
    np.testing.assert_allclose(scaler.transform(np.array([[5.0]])), [[0.0]])


def test_ridge_recovers_linear_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.3
    model = RidgeProbe(alpha=1e-6).fit(X, y)
    pred = model.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_logistic_separates_linearly_separable_data():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-2, 0.5, size=(100, 2)), rng.normal(2, 0.5, size=(100, 2))])
    y = np.array([0] * 100 + [1] * 100, dtype=float)
    model = LogisticProbe(alpha=0.01, lr=0.5, max_iter=800).fit(X, y)
    prob = model.predict_proba(X)
    acc = np.mean((prob >= 0.5) == (y == 1))
    assert acc > 0.95


def test_logistic_is_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] > 0).astype(float)
    a = LogisticProbe().fit(X, y).predict_proba(X)
    b = LogisticProbe().fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
