"""Task reward + compute penalty: R = R_task - lambda * C (AGENTS.md §7, §17).

`reward_task` and `reward_compute` are kept as SEPARATE fields so collapse and
reward hacking stay diagnosable (§17): a policy that games one component is visible
only if the components are logged apart. The compute penalty is applied exactly
once, over the named compute proxy (default: reasoning tokens — never "FLOPs"
unless FLOPs are measured, §7). lambda is never universal, so a sweep helper is the
primary entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

from when_to_think.config import RewardConfig


@dataclass
class RewardBreakdown:
    """One reward evaluation at a single lambda, with components kept separate."""

    correct: bool
    lambda_compute: float
    compute_units: float
    compute_proxy: str
    reward_task: float
    reward_compute: float
    reward_total: float


def compute_reward(
    *,
    correct: bool,
    compute_units: float,
    lambda_compute: float,
    reward_config: RewardConfig,
) -> RewardBreakdown:
    """R = R_task - lambda * C, with the penalty applied exactly once."""
    if compute_units < 0:
        raise ValueError("compute_units must be non-negative")
    if lambda_compute < 0:
        raise ValueError("lambda_compute must be non-negative")

    reward_task = (
        reward_config.task_reward_correct if correct else reward_config.task_reward_incorrect
    )
    # Single compute penalty; negative so more compute lowers total reward.
    reward_compute = -lambda_compute * compute_units
    return RewardBreakdown(
        correct=correct,
        lambda_compute=lambda_compute,
        compute_units=float(compute_units),
        compute_proxy=reward_config.compute_proxy,
        reward_task=float(reward_task),
        reward_compute=float(reward_compute),
        reward_total=float(reward_task + reward_compute),
    )


def compute_reward_sweep(
    *,
    correct: bool,
    compute_units: float,
    reward_config: RewardConfig,
) -> list[RewardBreakdown]:
    """Reward across the full lambda sweep (AGENTS.md §7: lambda is never universal)."""
    return [
        compute_reward(
            correct=correct,
            compute_units=compute_units,
            lambda_compute=lam,
            reward_config=reward_config,
        )
        for lam in reward_config.lambda_compute_sweep
    ]
