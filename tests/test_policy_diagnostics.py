"""Collapse + action-distribution diagnostics for the STOP/CONTINUE policy (M4)."""

from when_to_think.policies.diagnostics import (
    action_distribution_by_step,
    summarize_rollouts,
)
from when_to_think.policies.env import CONTINUE, STOP


def _ep(actions, stop_step, stop_tokens, correct, forced_stop, n_steps=3):
    reward_task = 1.0 if correct else 0.0
    return {
        "actions": actions, "stop_step": stop_step, "stop_tokens": stop_tokens,
        "correct": correct, "forced_stop": forced_stop, "n_steps": n_steps,
        "reward_task": reward_task, "reward_compute": -0.001 * stop_tokens,
        "reward_total": reward_task - 0.001 * stop_tokens,
    }


def test_detects_always_stop_collapse():
    eps = [_ep([STOP], 0, 0, True, False) for _ in range(50)]
    diag = summarize_rollouts(eps)
    assert diag["collapse"]["always_stop"] is True
    assert diag["collapse"]["collapsed"] is True
    assert diag["fraction_stop_immediately"] == 1.0
    assert diag["mean_reasoning_tokens"] == 0.0


def test_detects_always_continue_collapse():
    # Every episode runs to the cap (forced stop at the last step).
    eps = [_ep([CONTINUE, CONTINUE, CONTINUE], 2, 128, True, True) for _ in range(50)]
    diag = summarize_rollouts(eps)
    assert diag["collapse"]["always_continue"] is True
    assert diag["fraction_forced_stop_at_cap"] == 1.0


def test_healthy_mix_is_not_collapsed():
    eps = [_ep([STOP], 0, 0, True, False) for _ in range(25)]
    eps += [_ep([CONTINUE, STOP], 1, 64, True, False) for _ in range(25)]
    diag = summarize_rollouts(eps)
    assert diag["collapse"]["collapsed"] is False
    assert 0.0 < diag["mean_stop_step"] < 1.0
    # Components logged separately and task reward tracks accuracy (§17).
    assert diag["accuracy_equals_mean_reward_task"] is True


def test_action_distribution_by_step():
    eps = [_ep([CONTINUE, STOP], 1, 64, True, False),
           _ep([STOP], 0, 0, True, False)]
    dist = action_distribution_by_step(eps)
    # Step 0: one CONTINUE, one STOP -> 0.5 continue. Step 1: only the first reached it.
    assert dist[0]["fraction_continue"] == 0.5
    assert dist[1]["n_reached"] == 1 and dist[1]["fraction_stop"] == 1.0
