"""Trajectory write/load round-trip through the sharded store (M4)."""

import numpy as np

from when_to_think.policies.data import load_trajectories, write_trajectories
from when_to_think.representations import RepresentationDescriptor


def _desc():
    return RepresentationDescriptor(layers=[-1], token_position="last", pooling=None,
                                    model_name="synthetic", model_revision=None)


def test_trajectory_round_trip(tmp_path, make_trajectory):
    trajs = [
        make_trajectory("train-0", "train", [(0, False, 1.0), (64, True, -1.0)]),
        make_trajectory("test-3", "test", [(0, True, -1.0), (64, True, -1.0), (128, False, 1.0)],
                        sample_index=2),
    ]
    write_trajectories(tmp_path, trajs, _desc())
    loaded = load_trajectories(tmp_path)

    by_id = {(t.example_id, t.sample_index): t for t in loaded}
    assert set(by_id) == {("train-0", 0), ("test-3", 2)}

    t = by_id[("test-3", 2)]
    assert t.source_split == "test" and t.gold_answer == "42"
    assert [c.cumulative_reasoning_tokens for c in t.checkpoints] == [0, 64, 128]
    assert [c.correct for c in t.checkpoints] == [True, True, False]
    # Hidden vectors survive the round-trip.
    np.testing.assert_allclose(t.checkpoints[2].hidden[-1][0], 1.0)


def test_checkpoints_sorted_by_step(make_trajectory):
    # __post_init__ keeps checkpoints ordered regardless of construction order.
    t = make_trajectory("train-0", "train", [(0, True, 0), (64, True, 0)])
    t.checkpoints.reverse()
    t.checkpoints.sort(key=lambda c: c.step_index)
    assert [c.step_index for c in t.checkpoints] == [0, 1]
