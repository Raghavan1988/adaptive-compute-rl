"""HiddenStateReader round-trips vectors written by ShardedRepresentationWriter."""

import numpy as np

from when_to_think.representations import (
    HiddenStateReader,
    RepresentationDescriptor,
    ShardedRepresentationWriter,
)


def _descriptor():
    return RepresentationDescriptor(
        layers=[-1, 0], token_position="last", pooling=None,
        model_name="tiny", model_revision="abc123",
    )


def test_reader_joins_by_example_budget_sample(tmp_path):
    out = tmp_path / "hidden_states"
    vectors = {}
    with ShardedRepresentationWriter(out, _descriptor(), shard_size=2) as w:
        for eid in ("train-0", "test-1"):
            for budget in (0, 128):
                for s in range(2):
                    v_last = np.array([hash((eid, budget, s)) % 7, budget, s], dtype=np.float32)
                    v_zero = v_last + 100.0
                    vectors[(eid, budget, s)] = (v_last, v_zero)
                    w.add(eid, reasoning_step=0, layer_vectors={-1: v_last, 0: v_zero},
                          budget=budget, sample_index=s)

    reader = HiddenStateReader(out)
    assert reader.layers == [-1, 0]
    assert len(reader) == 8
    for (eid, budget, s), (v_last, v_zero) in vectors.items():
        got_last = reader.get(eid, budget=budget, sample_index=s, layer=-1)
        got_zero = reader.get(eid, budget=budget, sample_index=s, layer=0)
        np.testing.assert_allclose(got_last, v_last)
        np.testing.assert_allclose(got_zero, v_zero)


def test_reader_missing_key_returns_none(tmp_path):
    out = tmp_path / "hidden_states"
    with ShardedRepresentationWriter(out, _descriptor()) as w:
        w.add("train-0", reasoning_step=0,
              layer_vectors={-1: np.zeros(3, np.float32), 0: np.zeros(3, np.float32)},
              budget=0, sample_index=0)
    reader = HiddenStateReader(out)
    assert reader.get("train-0", budget=999, sample_index=0, layer=-1) is None
