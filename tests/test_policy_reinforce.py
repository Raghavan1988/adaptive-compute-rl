"""REINFORCE learns an adaptive STOP/CONTINUE policy, and the sweep runs end to end (M4).

Synthetic, model-free: dim 0 of the hidden state says whether to keep going. 'Easy'
examples are already correct (signal = STOP), 'hard' examples become correct only later
(signal = CONTINUE until the fixing step). A policy that follows the signal stops easy
examples early and continues hard ones — strictly better than any fixed budget.
"""

import numpy as np
import pytest

from when_to_think.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PolicyConfig,
    RewardConfig,
)
from when_to_think.policies.experiment import run_policy_sweep
from when_to_think.policies.reinforce import train_policy
from when_to_think.policies.rollouts import greedy_rollouts

EASY = [(0, True, -5.0), (64, True, -5.0), (128, True, -5.0)]
HARD = [(0, False, 5.0), (64, False, 5.0), (128, True, -5.0)]


def _dataset(make_trajectory, split, n_each, rng):
    trajs = []
    for i in range(n_each):
        trajs.append(make_trajectory(f"{split}-e{i}", split, EASY, rng=rng))
        trajs.append(make_trajectory(f"{split}-h{i}", split, HARD, rng=rng))
    return trajs


def _reward_cfg():
    return RewardConfig(task_reward_correct=1.0, task_reward_incorrect=0.0,
                        lambda_compute_sweep=[0.001], compute_proxy="reasoning_tokens")


def test_reinforce_learns_adaptive_policy(make_trajectory):
    rng = np.random.default_rng(0)
    train = _dataset(make_trajectory, "train", 30, rng)
    test = _dataset(make_trajectory, "test", 30, rng)
    cfg = PolicyConfig(layer=-1, hidden_sizes=[], include_progress_feature=True,
                       lr=0.02, iterations=250, episodes_per_batch=64, entropy_coef=0.01)

    model, log = train_policy(train, cfg, _reward_cfg(), lambda_compute=0.001, seed=0)
    assert log[-1]["mean_return"] > log[0]["mean_return"]  # learning improved return

    eps = greedy_rollouts(model, test, _reward_cfg(), 0.001)
    accuracy = np.mean([ep["correct"] for ep in eps])
    mean_reward = np.mean([ep["reward_total"] for ep in eps])

    def _is_easy(ep):
        return ep["example_id"].split("-")[1].startswith("e")

    easy_tokens = np.mean([ep["stop_tokens"] for ep in eps if _is_easy(ep)])
    hard_tokens = np.mean([ep["stop_tokens"] for ep in eps if not _is_easy(ep)])

    assert accuracy > 0.9                       # solves hard by continuing, keeps easy right
    assert easy_tokens < hard_tokens            # adaptive: easy stops earlier than hard
    # Beats always-STOP (reward 0.5) and always-run-to-cap (reward ~0.872) baselines.
    assert mean_reward > 0.85


def _experiment_cfg(iterations, lambdas):
    return ExperimentConfig(
        name="m4_test",
        model=ModelConfig(name="dummy"),
        data=DataConfig(dataset_name="gsm8k"),
        reward=RewardConfig(lambda_compute_sweep=lambdas),
        policy=PolicyConfig(iterations=iterations, episodes_per_batch=32),
        seed=0,
    )


def test_policy_sweep_runs_and_reports(make_trajectory):
    rng = np.random.default_rng(1)
    trajs = _dataset(make_trajectory, "train", 15, rng) + _dataset(make_trajectory, "test", 15, rng)
    results = run_policy_sweep(trajs, _experiment_cfg(iterations=30, lambdas=[0.0, 0.001]))

    assert len(results["adaptive_frontier"]) == 2
    assert results["split_sizes"]["train"] == 30 and results["split_sizes"]["test"] == 30
    assert "episodes" in results["per_lambda"]["0.0"]
    assert "diagnostics" in results["per_lambda"]["0.0"]
    assert isinstance(results["adaptive_beats_fixed"], bool)


def test_sweep_refuses_without_train_split(make_trajectory):
    rng = np.random.default_rng(2)
    test_only = _dataset(make_trajectory, "test", 5, rng)
    with pytest.raises(ValueError, match="TRAIN"):
        run_policy_sweep(test_only, _experiment_cfg(iterations=5, lambdas=[0.0]))
