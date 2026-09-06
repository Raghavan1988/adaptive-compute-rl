"""Policy error taxonomy: category assignment and aggregate accounting (M5)."""

import pytest

from when_to_think.policies.error_taxonomy import (
    categorize_episode,
    summarize_error_taxonomy,
)


def test_optimal_stop(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)])
    out = categorize_episode(traj, stop_step=0)
    assert out["category"] == "optimal_stop" and out["policy_fault"] is False
    assert out["wasted_tokens"] == 0


def test_correct_but_late_counts_wasted_tokens(make_trajectory):
    # Correct already at step 0, but the policy stopped at step 1 (64 wasted tokens).
    traj = make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)])
    out = categorize_episode(traj, stop_step=1)
    assert out["category"] == "correct_but_late"
    assert out["wasted_tokens"] == 64 and out["policy_fault"] is False


def test_premature_stop_is_policy_fault(make_trajectory):
    # Correct only at step 2, but the policy stopped at step 0.
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, False, 0), (128, True, 0)])
    out = categorize_episode(traj, stop_step=0)
    assert out["category"] == "premature_stop"
    assert out["stopped_correct"] is False and out["policy_fault"] is True


def test_overthought_to_wrong(make_trajectory):
    # Correct at step 0, then a later checkpoint is wrong; the policy continued to it.
    traj = make_trajectory("test-0", "test", [(0, True, 0), (64, False, 0)])
    out = categorize_episode(traj, stop_step=1)
    assert out["category"] == "overthought_to_wrong"
    assert out["policy_fault"] is True and out["wasted_tokens"] == 64


def test_unavoidable_wrong_is_not_a_fault(make_trajectory):
    traj = make_trajectory("test-0", "test", [(0, False, 0), (64, False, 0)])
    out = categorize_episode(traj, stop_step=1)
    assert out["category"] == "unavoidable_wrong" and out["policy_fault"] is False


def test_summary_aggregates(make_trajectory):
    trajs = [
        make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)]),   # optimal @0
        make_trajectory("test-1", "test", [(0, False, 0), (64, True, 0)]),  # premature if stop@0
        make_trajectory("test-2", "test", [(0, False, 0), (64, False, 0)]), # unavoidable
    ]
    episodes = [
        {"example_id": "test-0", "sample_index": 0, "stop_step": 0},
        {"example_id": "test-1", "sample_index": 0, "stop_step": 0},
        {"example_id": "test-2", "sample_index": 0, "stop_step": 1},
    ]
    s = summarize_error_taxonomy(trajs, episodes)
    assert s["n_episodes"] == 3
    assert s["category_counts"]["optimal_stop"] == 1
    assert s["category_counts"]["premature_stop"] == 1
    assert s["category_counts"]["unavoidable_wrong"] == 1
    assert s["policy_accuracy"] == pytest.approx(1 / 3)   # only test-0 stopped correct
    assert s["ceiling_accuracy"] == pytest.approx(2 / 3)  # test-0 & test-1 are answerable
    assert s["recoverable_accuracy_lost"] == pytest.approx(1 / 3)  # test-1 premature


def test_summary_requires_a_match(make_trajectory):
    trajs = [make_trajectory("test-0", "test", [(0, True, 0), (64, True, 0)])]
    with pytest.raises(ValueError, match="no matched"):
        summarize_error_taxonomy(trajs, [{"example_id": "other", "sample_index": 0, "stop_step": 0}])
