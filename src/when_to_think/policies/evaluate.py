"""Evaluate the STOP/CONTINUE policy against fixed budgets and the oracle (M4, Q3/Q4).

Everything is a matched comparison (AGENTS.md §4.1): the adaptive policy, the fixed
budgets, and the omniscient oracle are all scored on the SAME test trajectories, and
the policy is credited only against the accuracy a fixed budget reaches at the *same*
mean compute. Uncertainty is reported as bootstrap CIs over examples, so the headline
accuracy-vs-compute numbers carry error bars (§20). Non-monotonicity is respected — a
later checkpoint may be wrong where an earlier was right; the oracle argmax and the
fixed-step accuracies handle that directly (§4.5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from when_to_think.config import RewardConfig
from when_to_think.policies.data import Trajectory
from when_to_think.policies.env import stop_reward


def fixed_step_points(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """Fixed-budget baselines: always STOP at checkpoint step k, for each k.

    A trajectory shorter than k (finished naturally early) stops at its last checkpoint —
    a shorter budget simply cannot spend more. Accuracy and mean tokens are averaged over
    the same examples (matched comparison).
    """
    max_steps = max(len(t.checkpoints) for t in trajectories)
    points = []
    for k in range(max_steps):
        correct, tokens = [], []
        for t in trajectories:
            cp = t.checkpoints[min(k, len(t.checkpoints) - 1)]
            correct.append(1.0 if cp.correct else 0.0)
            tokens.append(float(cp.cumulative_reasoning_tokens))
        points.append({
            "step": k,
            "accuracy": float(np.mean(correct)),
            "mean_reasoning_tokens": float(np.mean(tokens)),
            "n": len(trajectories),
        })
    return points


def oracle_point(
    trajectories: list[Trajectory], lambda_compute: float, reward_config: RewardConfig,
) -> dict[str, Any]:
    """Omniscient per-trajectory best stop at this ``lambda`` (the M4 upper bound)."""
    correct, tokens = [], []
    for t in trajectories:
        rewards = [
            stop_reward(t, k, lambda_compute, reward_config).reward_total
            for k in range(len(t.checkpoints))
        ]
        best_k = int(np.argmax(rewards))  # ties: earliest (cheapest) via argmax
        correct.append(1.0 if t.checkpoints[best_k].correct else 0.0)
        tokens.append(float(t.checkpoints[best_k].cumulative_reasoning_tokens))
    return {
        "accuracy": float(np.mean(correct)),
        "mean_reasoning_tokens": float(np.mean(tokens)),
        "n": len(trajectories),
    }


def _bootstrap_ci(
    per_example: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of a per-example quantity."""
    rng = np.random.default_rng(seed)
    n = len(per_example)
    means = np.array([
        per_example[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)
    ])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(per_example.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def policy_point(episodes: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Adaptive policy's (accuracy, compute, reward) with bootstrap CIs over examples."""
    correct = np.array([1.0 if ep["correct"] else 0.0 for ep in episodes])
    tokens = np.array([float(ep["stop_tokens"]) for ep in episodes])
    reward = np.array([ep["reward_total"] for ep in episodes])
    return {
        "n": len(episodes),
        "accuracy": _bootstrap_ci(correct, seed=seed),
        "mean_reasoning_tokens": _bootstrap_ci(tokens, seed=seed + 1),
        "mean_reward": _bootstrap_ci(reward, seed=seed + 2),
    }


def matched_compute_comparison(
    policy_compute: float,
    policy_accuracy: float,
    fixed_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Accuracy gain of the policy over the fixed-budget frontier at matched compute.

    The fixed-budget accuracy is linearly interpolated at the policy's mean compute, so
    the comparison is at equal compute rather than crediting the policy for spending more
    or less than a baseline (§4.1).
    """
    xs = np.array([p["mean_reasoning_tokens"] for p in fixed_points])
    ys = np.array([p["accuracy"] for p in fixed_points])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # np.interp clamps outside the range to the endpoint accuracies.
    fixed_acc_at_match = float(np.interp(policy_compute, xs, ys))
    return {
        "policy_compute": policy_compute,
        "policy_accuracy": policy_accuracy,
        "fixed_accuracy_at_policy_compute": fixed_acc_at_match,
        "accuracy_gain_at_matched_compute": policy_accuracy - fixed_acc_at_match,
    }
