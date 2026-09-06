"""Error taxonomy for the STOP/CONTINUE policy: where it makes expensive mistakes (M5).

Given each trajectory's full checkpoint outcomes and the policy's chosen stop step, every
episode is placed in exactly one bucket. The point is to separate the policy's *own*
mistakes from the model's limitations, and to quantify what each costs:

- ``unavoidable_wrong``  : no checkpoint is ever correct — a model limitation, not the
                           policy's fault (kept, not hidden — §4.4).
- ``optimal_stop``       : stopped correct at the earliest correct checkpoint (ideal).
- ``correct_but_late``   : stopped correct, but an earlier checkpoint was already correct —
                           right answer, wasted compute.
- ``premature_stop``     : stopped wrong, but a LATER checkpoint would have been correct —
                           accuracy left on the table (the classic under-thinking error).
- ``overthought_to_wrong``: an earlier checkpoint was correct, but the policy continued and
                           stopped wrong — a correct→wrong flip caused by over-thinking
                           (no monotonicity assumed, §4.5). Wrong AND wasteful.

``premature_stop`` and ``overthought_to_wrong`` are the policy's own accuracy failures
(the model *could* have been right); ``correct_but_late`` and ``overthought_to_wrong``
waste compute. All numbers derive from the result files (§18).
"""

from __future__ import annotations

from typing import Any

from when_to_think.policies.data import Trajectory

# Categories where the model could have been right but the policy got it wrong.
_POLICY_FAULT = {"premature_stop", "overthought_to_wrong"}
# Categories that spent more compute than the cheapest correct checkpoint needed.
_WASTEFUL = {"correct_but_late", "overthought_to_wrong"}


def categorize_episode(traj: Trajectory, stop_step: int) -> dict[str, Any]:
    """Classify one episode's stop against its trajectory's checkpoint outcomes."""
    corrects = [cp.correct for cp in traj.checkpoints]
    tokens = [cp.cumulative_reasoning_tokens for cp in traj.checkpoints]
    any_correct = any(corrects)
    earliest = next((i for i, c in enumerate(corrects) if c), None)
    stopped_correct = corrects[stop_step]

    if not any_correct:
        category = "unavoidable_wrong"
    elif stopped_correct:
        category = "optimal_stop" if stop_step == earliest else "correct_but_late"
    elif earliest > stop_step:
        category = "premature_stop"
    else:  # earliest < stop_step: had it, continued, lost it
        category = "overthought_to_wrong"

    # Compute wasted beyond the cheapest correct checkpoint (only meaningful when a
    # correct checkpoint exists and the policy paid past it).
    wasted = 0
    if earliest is not None and stop_step > earliest and category in _WASTEFUL:
        wasted = tokens[stop_step] - tokens[earliest]

    return {
        "example_id": traj.example_id,
        "sample_index": traj.sample_index,
        "category": category,
        "stop_step": stop_step,
        "stop_tokens": tokens[stop_step],
        "stopped_correct": stopped_correct,
        "any_correct": any_correct,
        "earliest_correct_step": earliest,
        "wasted_tokens": wasted,
        "policy_fault": category in _POLICY_FAULT,
    }


def summarize_error_taxonomy(
    trajectories: list[Trajectory], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate the taxonomy over matched (trajectory, episode) pairs."""
    stops = {(ep["example_id"], ep["sample_index"]): ep["stop_step"] for ep in episodes}

    per_episode: list[dict[str, Any]] = []
    unmatched = 0
    for traj in trajectories:
        key = (traj.example_id, traj.sample_index)
        if key not in stops:
            unmatched += 1
            continue
        per_episode.append(categorize_episode(traj, stops[key]))

    n = len(per_episode)
    if n == 0:
        raise ValueError("no matched (trajectory, episode) pairs to analyze")

    categories = [
        "optimal_stop", "correct_but_late", "premature_stop",
        "overthought_to_wrong", "unavoidable_wrong",
    ]
    counts = {c: sum(1 for e in per_episode if e["category"] == c) for c in categories}
    by_cat_tokens = {
        c: (sum(e["stop_tokens"] for e in per_episode if e["category"] == c)
            / counts[c] if counts[c] else 0.0)
        for c in categories
    }

    def _frac(pred) -> float:
        return sum(1 for e in per_episode if pred(e)) / n

    return {
        "n_episodes": n,
        "unmatched_trajectories": unmatched,
        "category_counts": counts,
        "category_fractions": {c: counts[c] / n for c in categories},
        "mean_stop_tokens_by_category": by_cat_tokens,
        "policy_accuracy": _frac(lambda e: e["stopped_correct"]),
        # Upper bound: fraction the model gets right at SOME checkpoint (the oracle ceiling).
        "ceiling_accuracy": _frac(lambda e: e["any_correct"]),
        # The policy's own accuracy failures (correctable, but it got them wrong).
        "recoverable_accuracy_lost": _frac(lambda e: e["policy_fault"]),
        "mean_wasted_tokens": sum(e["wasted_tokens"] for e in per_episode) / n,
        "per_episode": per_episode,
    }
