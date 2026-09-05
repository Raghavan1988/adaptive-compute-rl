"""Deterministic rollouts of a trained policy, for evaluation and diagnostics (M4).

A greedy rollout walks a trajectory's checkpoints, taking the policy's argmax action at
each, until STOP (or a forced STOP at the budget cap). It returns a flat episode record
with the stop point, correctness, and the separated reward components (§17) so the
collapse / reward-hacking diagnostics can be computed from result files.
"""

from __future__ import annotations

from typing import Any

from when_to_think.config import RewardConfig
from when_to_think.policies.data import Trajectory
from when_to_think.policies.env import StopContinueEnv
from when_to_think.policies.policy import PolicyModel


def greedy_rollout(
    policy: PolicyModel,
    traj: Trajectory,
    reward_config: RewardConfig,
    lambda_compute: float,
    *,
    max_reasoning_budget: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic episode; return a flat, JSON-friendly episode record."""
    env = StopContinueEnv(
        traj, reward_config, lambda_compute, max_reasoning_budget=max_reasoning_budget
    )
    cp = env.reset()
    actions: list[int] = []
    info: dict[str, Any] = {}
    while True:
        action = policy.act_greedy(cp, traj)
        actions.append(action)
        cp, _reward, done, info = env.step(action)
        if done:
            break

    return {
        "example_id": traj.example_id,
        "source_split": traj.source_split,
        "sample_index": traj.sample_index,
        "n_steps": env.n_steps,
        "actions": actions,
        "stop_step": info["stop_step"],
        "stop_tokens": info["stop_tokens"],
        "correct": bool(info["correct"]),
        "forced_stop": bool(info["forced_stop"]),
        "reward_task": info["reward_task"],
        "reward_compute": info["reward_compute"],
        "reward_total": info["reward_total"],
    }


def greedy_rollouts(
    policy: PolicyModel,
    trajectories: list[Trajectory],
    reward_config: RewardConfig,
    lambda_compute: float,
    *,
    max_reasoning_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Greedy rollout over many trajectories (one episode each)."""
    return [
        greedy_rollout(
            policy, traj, reward_config, lambda_compute,
            max_reasoning_budget=max_reasoning_budget,
        )
        for traj in trajectories
    ]
