"""Evaluation math: fixed/oracle points, bootstrap CIs, matched-compute (M4)."""

import pytest

from when_to_think.config import RewardConfig
from when_to_think.policies.evaluate import (
    fixed_step_points,
    matched_compute_comparison,
    oracle_point,
    policy_point,
)


def _reward_cfg():
    return RewardConfig(task_reward_correct=1.0, task_reward_incorrect=0.0,
                        lambda_compute_sweep=[0.0], compute_proxy="reasoning_tokens")


def test_fixed_step_points(make_trajectory):
    # Two examples; step 0 half correct, step 1 both correct.
    trajs = [
        make_trajectory("test-0", "test", [(0, False, 0), (64, True, 0)]),
        make_trajectory("test-1", "test", [(0, True, 0), (64, True, 0)]),
    ]
    pts = fixed_step_points(trajs)
    assert pts[0]["accuracy"] == 0.5 and pts[0]["mean_reasoning_tokens"] == 0.0
    assert pts[1]["accuracy"] == 1.0 and pts[1]["mean_reasoning_tokens"] == 64.0


def test_oracle_picks_cheapest_correct(make_trajectory):
    # Example correct at both 0 and 64 -> oracle stops at 0. Example only correct at 64.
    trajs = [
        make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)]),
        make_trajectory("test-1", "test", [(0, False, 0), (64, True, 0)]),
    ]
    orc = oracle_point(trajs, lambda_compute=0.001, reward_config=_reward_cfg())
    assert orc["accuracy"] == 1.0
    assert orc["mean_reasoning_tokens"] == pytest.approx(32.0)  # (0 + 64) / 2


def test_policy_point_bootstrap_ci_brackets_mean():
    episodes = [
        {"correct": True, "stop_tokens": 0, "reward_total": 1.0},
        {"correct": False, "stop_tokens": 64, "reward_total": -0.064},
        {"correct": True, "stop_tokens": 128, "reward_total": 0.872},
        {"correct": True, "stop_tokens": 64, "reward_total": 0.936},
    ]
    pt = policy_point(episodes, seed=0)
    acc = pt["accuracy"]
    assert acc["ci_low"] <= acc["mean"] <= acc["ci_high"]
    assert acc["mean"] == 0.75


def test_matched_compute_interpolates_fixed_frontier():
    fixed = [
        {"mean_reasoning_tokens": 0.0, "accuracy": 0.4},
        {"mean_reasoning_tokens": 100.0, "accuracy": 0.8},
    ]
    # Policy at 50 tokens, 0.75 accuracy: fixed interpolates to 0.6 -> gain +0.15.
    m = matched_compute_comparison(50.0, 0.75, fixed)
    assert m["fixed_accuracy_at_policy_compute"] == pytest.approx(0.6)
    assert m["accuracy_gain_at_matched_compute"] == pytest.approx(0.15)
