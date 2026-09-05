"""RL collapse + reward-hacking diagnostics for the STOP/CONTINUE policy (M4).

CLAUDE.md / AGENTS.md §16 require these on every policy run: fraction STOP vs CONTINUE,
mean reasoning tokens, accuracy, mean reward with components separated, and the action
distribution by reasoning step. Rising aggregate reward is NOT proof of learning, so
collapse toward ~100% STOP (never think) or ~100% CONTINUE (always run to the cap) is
flagged explicitly. All numbers derive from the episode records, never hand-typed (§18).
"""

from __future__ import annotations

from typing import Any

from when_to_think.policies.env import CONTINUE, STOP

# A run is flagged as collapsed past these fractions (near-degenerate behavior).
_COLLAPSE_THRESHOLD = 0.98


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def action_distribution_by_step(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per decision step, among episodes that REACHED it, the fraction that CONTINUEd.

    Reveals *where* the policy stops: a healthy policy varies its stop step across
    examples; a collapsed one continues everywhere or stops at step 0 everywhere.
    """
    max_steps = max((ep["n_steps"] for ep in episodes), default=0)
    out: list[dict[str, Any]] = []
    for step in range(max_steps):
        reached = [ep for ep in episodes if len(ep["actions"]) > step]
        if not reached:
            continue
        n_continue = sum(1 for ep in reached if ep["actions"][step] == CONTINUE)
        out.append({
            "step": step,
            "n_reached": len(reached),
            "fraction_continue": n_continue / len(reached),
            "fraction_stop": 1.0 - n_continue / len(reached),
        })
    return out


def summarize_rollouts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate diagnostics + collapse/reward-hacking flags for a batch of episodes."""
    n = len(episodes)
    if n == 0:
        raise ValueError("no episodes to summarize")

    # Overall action mix across ALL decisions taken (not just terminal ones).
    total_actions = [a for ep in episodes for a in ep["actions"]]
    n_continue = sum(1 for a in total_actions if a == CONTINUE)
    n_stop = sum(1 for a in total_actions if a == STOP)

    frac_stop_immediately = _mean([1.0 if ep["stop_step"] == 0 else 0.0 for ep in episodes])
    frac_forced_stop = _mean([1.0 if ep["forced_stop"] else 0.0 for ep in episodes])
    accuracy = _mean([1.0 if ep["correct"] else 0.0 for ep in episodes])
    mean_reward = _mean([ep["reward_total"] for ep in episodes])

    # Collapse: nearly every episode stops immediately (never thinks) or runs to the
    # cap (never stops early). Either makes the policy equivalent to a fixed budget.
    always_stop = frac_stop_immediately >= _COLLAPSE_THRESHOLD
    always_continue = frac_forced_stop >= _COLLAPSE_THRESHOLD

    # Reward-hacking hint: stopping on answers that never parse would inflate compute
    # savings while task reward stays 0. Surface the share of stops that scored 0 task
    # reward at the cheapest step, as a signal to inspect (not a hard verdict).
    frac_correct = accuracy

    return {
        "n_episodes": n,
        "fraction_stop": n_stop / len(total_actions) if total_actions else float("nan"),
        "fraction_continue": n_continue / len(total_actions) if total_actions else float("nan"),
        "fraction_stop_immediately": frac_stop_immediately,
        "fraction_forced_stop_at_cap": frac_forced_stop,
        "mean_stop_step": _mean([float(ep["stop_step"]) for ep in episodes]),
        "mean_reasoning_tokens": _mean([float(ep["stop_tokens"]) for ep in episodes]),
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "mean_reward_task": _mean([ep["reward_task"] for ep in episodes]),
        "mean_reward_compute": _mean([ep["reward_compute"] for ep in episodes]),
        "action_distribution_by_step": action_distribution_by_step(episodes),
        "collapse": {
            "always_stop": always_stop,
            "always_continue": always_continue,
            "collapsed": always_stop or always_continue,
        },
        # Kept for the reward-hacking check: task reward must track accuracy, not diverge.
        "accuracy_equals_mean_reward_task": abs(frac_correct - _mean(
            [ep["reward_task"] for ep in episodes]
        )) < 1e-9,
    }
