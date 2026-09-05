"""Orchestrate the M4 policy sweep: train per lambda, evaluate on test (Q3/Q4).

For each compute penalty in the sweep (lambda is never universal, §7) a fresh policy is
trained on TRAIN trajectories and rolled out greedily on TEST, producing one adaptive
frontier point. The fixed-budget and oracle frontiers are computed once from the same
test trajectories, so the headline curve — adaptive vs fixed, with the oracle ceiling —
is a matched comparison from a single set of result files (§4.1, §18).

Discipline: TRAIN is used for learning, TEST only for the final rollout; no quantity is
tuned on test. VAL trajectories are carried through untouched here (available for future
model selection) — the sweep itself does not select on them.
"""

from __future__ import annotations

from typing import Any

from when_to_think.config import ExperimentConfig
from when_to_think.policies.data import Trajectory
from when_to_think.policies.diagnostics import summarize_rollouts
from when_to_think.policies.evaluate import (
    fixed_step_points,
    matched_compute_comparison,
    oracle_point,
    policy_point,
)
from when_to_think.policies.reinforce import train_policy
from when_to_think.policies.rollouts import greedy_rollouts


def _by_split(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
    out: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        out.setdefault(t.source_split, []).append(t)
    return out


def _downsample(log: list[dict[str, Any]], k: int = 20) -> list[dict[str, Any]]:
    """Keep at most k+1 evenly-spaced training-curve points (plus the last)."""
    if len(log) <= k + 1:
        return log
    step = max(1, len(log) // k)
    kept = log[::step]
    if kept[-1] is not log[-1]:
        kept.append(log[-1])
    return kept


def run_policy_sweep(
    trajectories: list[Trajectory], cfg: ExperimentConfig,
) -> dict[str, Any]:
    """Train + evaluate a STOP/CONTINUE policy across the lambda sweep."""
    splits = _by_split(trajectories)
    train = splits.get("train", [])
    test = splits.get("test", [])
    if not train:
        raise ValueError(
            "No TRAIN trajectories. Generate them with --splits train,val,test "
            "before training a policy (no test-set training, §4.2)."
        )
    if not test:
        raise ValueError("No TEST trajectories to evaluate on.")

    fixed = fixed_step_points(test)
    per_lambda: dict[str, Any] = {}
    adaptive_frontier: list[dict[str, Any]] = []
    oracle_frontier: list[dict[str, Any]] = []

    for lam in cfg.reward.lambda_compute_sweep:
        model, log = train_policy(
            train, cfg.policy, cfg.reward, lam, seed=cfg.seed
        )
        episodes = greedy_rollouts(model, test, cfg.reward, lam)
        diagnostics = summarize_rollouts(episodes)
        pt = policy_point(episodes, seed=cfg.seed)
        orc = oracle_point(test, lam, cfg.reward)
        matched = matched_compute_comparison(
            pt["mean_reasoning_tokens"]["mean"], pt["accuracy"]["mean"], fixed
        )
        per_lambda[str(lam)] = {
            "lambda_compute": lam,
            "training_curve": _downsample(log),
            "diagnostics": diagnostics,
            "policy_point": pt,
            "oracle_point": orc,
            "matched_compute": matched,
            "episodes": episodes,
        }
        adaptive_frontier.append({
            "lambda_compute": lam,
            "accuracy": pt["accuracy"]["mean"],
            "accuracy_ci": [pt["accuracy"]["ci_low"], pt["accuracy"]["ci_high"]],
            "mean_reasoning_tokens": pt["mean_reasoning_tokens"]["mean"],
            "collapsed": diagnostics["collapse"]["collapsed"],
        })
        oracle_frontier.append({"lambda_compute": lam, **orc})

    # Q4 verdict: does the adaptive policy improve on the fixed-budget frontier at
    # matched compute anywhere on the sweep? (Reported honestly; CIs are available.)
    gains = [
        per_lambda[str(lam)]["matched_compute"]["accuracy_gain_at_matched_compute"]
        for lam in cfg.reward.lambda_compute_sweep
    ]
    best_gain = max(gains) if gains else float("nan")

    return {
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "lambda_sweep": list(cfg.reward.lambda_compute_sweep),
        "fixed_frontier": fixed,
        "oracle_frontier": oracle_frontier,
        "adaptive_frontier": adaptive_frontier,
        "per_lambda": per_lambda,
        "best_accuracy_gain_at_matched_compute": best_gain,
        "adaptive_beats_fixed": bool(best_gain > 0),
        "any_collapsed": any(p["collapsed"] for p in adaptive_frontier),
    }
