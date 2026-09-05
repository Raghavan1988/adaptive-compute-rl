"""RL STOP/CONTINUE policy over frozen hidden states (M4).

The policy decides *when to stop reasoning*, conditioned on the decision-point hidden
state, and is trained to beat fixed budgets on the accuracy-compute frontier. See
``env.py`` for the offline trajectory environment, ``reinforce.py`` for training, and
``experiment.py`` for the lambda-swept headline comparison.
"""

from when_to_think.policies.data import (
    Checkpoint,
    Trajectory,
    load_trajectories,
    write_trajectories,
)
from when_to_think.policies.diagnostics import summarize_rollouts
from when_to_think.policies.env import CONTINUE, STOP, StopContinueEnv, stop_reward
from when_to_think.policies.evaluate import (
    fixed_step_points,
    matched_compute_comparison,
    oracle_point,
    policy_point,
)
from when_to_think.policies.experiment import run_policy_sweep
from when_to_think.policies.policy import PolicyModel, StopContinuePolicy, build_features
from when_to_think.policies.reinforce import train_policy
from when_to_think.policies.rollouts import greedy_rollout, greedy_rollouts

__all__ = [
    "CONTINUE",
    "STOP",
    "Checkpoint",
    "PolicyModel",
    "StopContinueEnv",
    "StopContinuePolicy",
    "Trajectory",
    "build_features",
    "fixed_step_points",
    "greedy_rollout",
    "greedy_rollouts",
    "load_trajectories",
    "matched_compute_comparison",
    "oracle_point",
    "policy_point",
    "run_policy_sweep",
    "stop_reward",
    "summarize_rollouts",
    "train_policy",
    "write_trajectories",
]
