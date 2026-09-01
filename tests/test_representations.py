"""Tests for hidden-state extraction and sharded storage (AGENTS.md §13)."""

import json

import numpy as np
import pytest
import torch

from when_to_think.representations import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
    extract_hidden_states,
    pool_hidden,
)


def _descriptor(layers):
    return RepresentationDescriptor(
        layers=layers, token_position="last", pooling=None,
        model_name="test-model", model_revision="rev0",
    )


# --------------------------------------------------------------------------- #
# Pooling / token selection
# --------------------------------------------------------------------------- #

def test_last_token_left_padded():
    # Left-padded row: real token is at the END. Values encode position along seq.
    layer = torch.tensor([[[0.0], [0.0], [9.0]]])  # (batch=1, seq=3, hidden=1)
    mask = torch.tensor([[0, 0, 1]])  # only last position is real
    out = pool_hidden(layer, mask, "last", None)
    assert out.item() == 9.0


def test_last_token_right_padded():
    # Right-padded row: real tokens first, then pad. Last real token is index 1.
    layer = torch.tensor([[[1.0], [7.0], [0.0]]])
    mask = torch.tensor([[1, 1, 0]])
    out = pool_hidden(layer, mask, "last", None)
    assert out.item() == 7.0


def test_mean_pool_ignores_padding():
    layer = torch.tensor([[[2.0], [4.0], [100.0]]])  # last token is padding
    mask = torch.tensor([[1, 1, 0]])
    out = pool_hidden(layer, mask, "all", "mean")
    assert out.item() == 3.0  # mean of 2 and 4, not 100


def test_last_token_rejects_pooling():
    layer = torch.zeros((1, 2, 1))
    mask = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError):
        pool_hidden(layer, mask, "last", "mean")


# --------------------------------------------------------------------------- #
# Layer selection
# --------------------------------------------------------------------------- #

def test_extract_selects_layers_and_negative_index():
    # 3 hidden states (embedding + 2 layers), each (batch=2, seq=2, hidden=4).
    hs = [torch.randn(2, 2, 4) for _ in range(3)]
    mask = torch.ones((2, 2), dtype=torch.long)
    spec = _descriptor(layers=[0, -1])
    out = extract_hidden_states(hs, mask, spec)
    assert set(out) == {0, -1}
    assert out[0].shape == (2, 4)
    assert out[0].dtype == np.float32
    # layer -1 must equal the last-token of the last hidden state.
    expected_last = hs[-1][:, -1, :].numpy()
    np.testing.assert_allclose(out[-1], expected_last, rtol=1e-6)


def test_extract_out_of_range_layer_raises():
    hs = [torch.randn(1, 2, 4) for _ in range(3)]
    mask = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(IndexError):
        extract_hidden_states(hs, mask, _descriptor(layers=[99]))


def test_extract_handles_bfloat16_input():
    hs = [torch.randn(1, 2, 4).to(torch.bfloat16) for _ in range(2)]
    mask = torch.ones((1, 2), dtype=torch.long)
    out = extract_hidden_states(hs, mask, _descriptor(layers=[-1]))
    assert out[-1].dtype == np.float32  # bf16 promoted so numpy can hold it


# --------------------------------------------------------------------------- #
# Sharded writer
# --------------------------------------------------------------------------- #

def test_sharded_writer_roundtrip(tmp_path):
    descriptor = _descriptor(layers=[-1])
    # 5 records with shard_size 2 => shards of 2, 2, 1.
    vectors = {i: {-1: np.full((4,), float(i), dtype=np.float32)} for i in range(5)}
    with ShardedRepresentationWriter(tmp_path, descriptor, shard_size=2) as writer:
        for i in range(5):
            writer.add(example_id=f"ex-{i}", reasoning_step=0, layer_vectors=vectors[i])

    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 5
    assert {row["shard"] for row in manifest} == {"shard_00000.npz", "shard_00001.npz", "shard_00002.npz"}

    # descriptor persisted with provenance.
    desc = json.loads((tmp_path / "descriptor.json").read_text())
    assert desc["model_revision"] == "rev0"
    assert desc["token_position"] == "last"

    # Values round-trip: reconstruct ex-3's vector from its manifest entry.
    entry = next(row for row in manifest if row["example_id"] == "ex-3")
    shard = np.load(tmp_path / entry["shard"])
    np.testing.assert_array_equal(shard["layer_-1"][entry["row"]], np.full((4,), 3.0, dtype=np.float32))
