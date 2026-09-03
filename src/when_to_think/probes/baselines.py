"""Baselines the hidden-state probe must beat (M3, Question 2).

For the probe to be evidence that *internal state* carries the value of compute, it
has to beat predictors that never look inside the model:

- **base rate / prior** — a constant predictor (mean Δ, or the positive base rate).
  The floor: any probe below this has found nothing.
- **input-only difficulty** — features of the *question alone* (length, digit/number
  counts). Tests whether the value of compute is decodable from the problem itself,
  with no model internals. This is also the internal-state-vs-input-only ablation
  (PLAN.md M5).

Two baselines named in PLAN.md M3 — **entropy-based** and **verbalized confidence** —
are intentionally NOT here yet: they need signals the M1 sweep does not log (token-level
logits for predictive entropy; a "how confident are you?" elicitation for verbalized
confidence). Adding them is a generation-side pass (see docs/tasks); the probe/baseline
comparison in ``train.py`` is written to accept any extra feature matrix so they drop in
without reshaping the pipeline. Until then, Question 2 is answered against the
input-only baseline, and that scope limit is reported honestly in the results.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_DIGIT_RE = re.compile(r"\d")


def input_difficulty_features(meta: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Question-only difficulty features (no model internals), one row per instance.

    Deliberately cheap and interpretable: prompt length in tokens, character length,
    word count, digit count, and count of numeric spans. These proxy "how hard / how
    long is the problem" from the input alone.
    """
    names = ["prompt_tokens", "char_len", "n_words", "n_digits", "n_numbers"]
    feats: list[list[float]] = []
    for m in meta:
        q = m.get("question", "") or ""
        prompt_tokens = m.get("prompt_tokens")
        feats.append(
            [
                float(prompt_tokens) if prompt_tokens is not None else float(len(q.split())),
                float(len(q)),
                float(len(q.split())),
                float(len(_DIGIT_RE.findall(q))),
                float(len(_NUMBER_RE.findall(q))),
            ]
        )
    return np.asarray(feats, dtype=np.float64), names


def prior_regression_prediction(y_train: np.ndarray, n: int) -> np.ndarray:
    """Constant predictor for the regression target: the train-set mean Δ, repeated."""
    return np.full(n, float(np.mean(y_train)), dtype=np.float64)


def prior_binary_prediction(y_train: np.ndarray, n: int) -> np.ndarray:
    """Constant predictor for the binary target: the train-set positive base rate."""
    return np.full(n, float(np.mean(y_train)), dtype=np.float64)
