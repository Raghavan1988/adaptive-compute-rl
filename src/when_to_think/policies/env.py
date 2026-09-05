"""The STOP/CONTINUE environment over one trajectory's checkpoints (M4, AGENTS.md §15).

Action semantics (research-significant, §25):

- ``STOP`` (0): terminate, score the provisional answer at the current checkpoint, and
  apply the accumulated compute cost exactly once (R = R_task − λ·C, §7). Episode ends.
- ``CONTINUE`` (1): reveal the next checkpoint of the same trajectory (a fixed reasoning
  increment already granted at generation time), accruing no reward yet. The cost is
  charged in full only when the episode finally STOPs, so the penalty is applied once.

The maximum compute budget is enforced structurally: a trajectory's last checkpoint is
at most ``max_reasoning_budget`` tokens, and CONTINUE at the last checkpoint is a
**forced STOP** — the environment can never spend beyond the cap (§15). An assertion
guards against a trajectory that somehow exceeds it.
"""

from __future__ import annotations

from typing import Any

from when_to_think.config import RewardConfig
from when_to_think.policies.data import Checkpoint, Trajectory
from when_to_think.rewards.reward import compute_reward

STOP = 0
CONTINUE = 1


def stop_reward(
    traj: Trajectory,
    step_index: int,
    lambda_compute: float,
    reward_config: RewardConfig,
) -> Any:
    """Reward breakdown for STOPping ``traj`` at checkpoint ``step_index``.

    Shared by the environment and by the fixed/oracle baselines so every method scores
    a stop identically (matched comparison, §4.1).
    """
    cp = traj.checkpoints[step_index]
    return compute_reward(
        correct=cp.correct,
        compute_units=cp.cumulative_reasoning_tokens,
        lambda_compute=lambda_compute,
        reward_config=reward_config,
    )


class StopContinueEnv:
    """Single-trajectory STOP/CONTINUE episode. Observation is the current Checkpoint."""

    def __init__(
        self,
        traj: Trajectory,
        reward_config: RewardConfig,
        lambda_compute: float,
        *,
        max_reasoning_budget: int | None = None,
    ) -> None:
        if not traj.checkpoints:
            raise ValueError("trajectory has no checkpoints")
        self.traj = traj
        self.reward_config = reward_config
        self.lambda_compute = lambda_compute
        self.max_reasoning_budget = (
            max_reasoning_budget if max_reasoning_budget is not None else traj.max_tokens
        )
        self._i = 0
        self._done = False

    @property
    def n_steps(self) -> int:
        return len(self.traj.checkpoints)

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> Checkpoint:
        self._i = 0
        self._done = False
        return self.current()

    def current(self) -> Checkpoint:
        return self.traj.checkpoints[self._i]

    def step(self, action: int) -> tuple[Checkpoint, float, bool, dict]:
        """Advance one decision. Returns (next_checkpoint, reward, done, info)."""
        if self._done:
            raise RuntimeError("step() called on a finished episode; reset() first")
        if action not in (STOP, CONTINUE):
            raise ValueError(f"action must be STOP(0) or CONTINUE(1), got {action!r}")

        at_last = self._i >= self.n_steps - 1
        if action == CONTINUE and not at_last:
            self._i += 1
            return self.current(), 0.0, False, {"action": CONTINUE, "forced_stop": False}

        # STOP, or CONTINUE at the last checkpoint (budget cap) => forced STOP.
        forced = action == CONTINUE and at_last
        cp = self.current()
        if cp.cumulative_reasoning_tokens > self.max_reasoning_budget:
            raise AssertionError(
                f"checkpoint tokens {cp.cumulative_reasoning_tokens} exceed max budget "
                f"{self.max_reasoning_budget} — budget enforcement violated"
            )
        rb = stop_reward(self.traj, self._i, self.lambda_compute, self.reward_config)
        self._done = True
        info = {
            "action": action,
            "forced_stop": forced,
            "stop_step": self._i,
            "stop_tokens": cp.cumulative_reasoning_tokens,
            "correct": cp.correct,
            "reward_task": rb.reward_task,
            "reward_compute": rb.reward_compute,
            "reward_total": rb.reward_total,
        }
        return cp, rb.reward_total, True, info
