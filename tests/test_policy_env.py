"""StopContinueEnv transition semantics, reward, and budget enforcement (M4)."""

import pytest

from when_to_think.config import RewardConfig
from when_to_think.policies.env import CONTINUE, STOP, StopContinueEnv, stop_reward


def _reward_cfg():
    return RewardConfig(task_reward_correct=1.0, task_reward_incorrect=0.0,
                        lambda_compute_sweep=[0.0], compute_proxy="reasoning_tokens")


def test_stop_ends_episode_with_task_and_compute_reward(make_trajectory):
    # Stop at step 1: correct, 64 tokens, lambda 0.001 -> reward 1 - 0.064.
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, True, 0), (128, True, 0)])
    env = StopContinueEnv(traj, _reward_cfg(), lambda_compute=0.001)
    env.reset()
    _cp, reward, done, info = env.step(CONTINUE)  # advance to step 1
    assert not done and info["action"] == CONTINUE
    _cp, reward, done, info = env.step(STOP)
    assert done and info["stop_step"] == 1 and info["stop_tokens"] == 64
    assert info["reward_task"] == 1.0
    assert info["reward_compute"] == pytest.approx(-0.064)
    assert reward == pytest.approx(1.0 - 0.064)


def test_continue_at_last_checkpoint_is_forced_stop(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, True, 0)])
    env = StopContinueEnv(traj, _reward_cfg(), lambda_compute=0.0)
    env.reset()
    env.step(CONTINUE)  # to last (step 1)
    _cp, _reward, done, info = env.step(CONTINUE)  # cannot go further -> forced STOP
    assert done and info["forced_stop"] is True and info["stop_step"] == 1


def test_step_after_done_raises(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)])
    env = StopContinueEnv(traj, _reward_cfg(), lambda_compute=0.0)
    env.reset()
    env.step(STOP)
    with pytest.raises(RuntimeError):
        env.step(STOP)


def test_budget_never_exceeded(make_trajectory):
    # Max budget below the last checkpoint's tokens must trip the enforcement guard when
    # the episode would bill compute beyond the cap (§15: never silently exceed).
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, False, 0), (128, True, 0)])
    env = StopContinueEnv(traj, _reward_cfg(), lambda_compute=0.0, max_reasoning_budget=100)
    env.reset()
    env.step(CONTINUE)  # -> step 1 (64 tokens, within cap)
    env.step(CONTINUE)  # -> step 2 (128 tokens), not yet a stop
    with pytest.raises(AssertionError, match="budget enforcement"):
        env.step(STOP)  # stopping here would bill 128 > 100


def test_compute_penalty_applied_once(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0), (128, True, 0)])
    # Continue through all steps, forced stop at 128: penalty uses cumulative 128 exactly once.
    env = StopContinueEnv(traj, _reward_cfg(), lambda_compute=0.01)
    env.reset()
    env.step(CONTINUE)
    env.step(CONTINUE)
    _cp, reward, done, info = env.step(CONTINUE)  # forced stop at 128
    assert done and info["reward_compute"] == pytest.approx(-1.28)
    assert reward == pytest.approx(1.0 - 1.28)


def test_stop_reward_helper_matches(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, True, 0)])
    rb = stop_reward(traj, 1, 0.001, _reward_cfg())
    assert rb.reward_task == 1.0 and rb.reward_compute == pytest.approx(-0.064)
