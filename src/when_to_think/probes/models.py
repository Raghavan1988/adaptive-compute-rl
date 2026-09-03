"""Small, deterministic linear probes on frozen hidden states (M3).

A probe measures *decodability*: whether the value of continuing to reason is
linearly readable off a frozen hidden state. It is deliberately simple (logistic /
ridge, not a deep net) so that a positive result is attributable to the signal in
the representation rather than to a powerful predictor — AGENTS.md §14, and the
CLAUDE.md rule to describe results as decodability, never mechanism.

Everything here is pure-numpy and deterministic given the data (closed-form ridge;
full-batch gradient descent from a zero init for logistic regression), so a probe
refit reproduces bit-for-bit. Features are standardized with statistics fit on the
TRAIN split only — the scaler must never see val/test, or the "no test-set training"
invariant (AGENTS.md §4.2) is silently violated through the normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Per-feature zero-mean/unit-variance standardizer, fit on train features only."""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # Guard zero-variance features (a constant column) against divide-by-zero.
        std[std < 1e-12] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler must be fit before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass
class RidgeProbe:
    """L2-regularized linear regression (closed form) for the value-of-compute target.

    The intercept is fit by centering and is never penalized; only the weight vector
    is shrunk by ``alpha``. Closed form keeps the fit deterministic and free of any
    optimizer state.
    """

    alpha: float = 1.0
    weights_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeProbe:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_mean = float(y.mean())
        yc = y - y_mean
        n_features = X.shape[1]
        gram = X.T @ X + self.alpha * np.eye(n_features)
        self.weights_ = np.linalg.solve(gram, X.T @ yc)
        self.intercept_ = y_mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("RidgeProbe must be fit before predict")
        return np.asarray(X, dtype=np.float64) @ self.weights_ + self.intercept_


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class LogisticProbe:
    """L2-regularized logistic regression for the binary "continuing fixes it" target.

    Full-batch gradient descent from a zero init: deterministic (no shuffling, no
    random init) so a refit is reproducible. The bias term is not regularized. This
    is a probe, not a production classifier — a few hundred steps on standardized
    features is enough to read out a linear signal if one is present.
    """

    alpha: float = 1.0
    lr: float = 0.1
    max_iter: int = 500
    weights_: np.ndarray | None = None
    intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticProbe:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        for _ in range(self.max_iter):
            p = _sigmoid(X @ w + b)
            grad_w = X.T @ (p - y) / n_samples + self.alpha * w / n_samples
            grad_b = float(np.mean(p - y))  # intercept unregularized
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.weights_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("LogisticProbe must be fit before predict_proba")
        return _sigmoid(np.asarray(X, dtype=np.float64) @ self.weights_ + self.intercept_)
