"""Incremental trajectory generation on a tiny model (skips when offline) (M4)."""

import pytest

from when_to_think.config import GenerationConfig, ModelConfig
from when_to_think.generation.incremental import (
    _checkpoint_prefix_lengths,
    generate_trajectory,
)
from when_to_think.models import load_model_and_tokenizer
from when_to_think.representations import RepresentationDescriptor


def test_checkpoint_prefix_lengths():
    # 0, then every interval, then the natural end.
    assert _checkpoint_prefix_lengths(200, 64) == [0, 64, 128, 192, 200]
    assert _checkpoint_prefix_lengths(128, 64) == [0, 64, 128]  # exact multiple, no dup
    assert _checkpoint_prefix_lengths(0, 64) == [0]             # no reasoning at all


@pytest.fixture(scope="module")
def tiny_loaded():
    cfg = ModelConfig(name="hf-internal-testing/tiny-random-gpt2", dtype="float32", device="cpu")
    try:
        return load_model_and_tokenizer(cfg)
    except Exception as exc:  # noqa: BLE001 - any load failure (offline) => skip
        pytest.skip(f"tiny model unavailable: {exc}")


def _spec():
    return RepresentationDescriptor(layers=[-1], token_position="last", pooling=None,
                                    model_name="tiny", model_revision=None)


def test_generate_trajectory_is_coherent_chain(tiny_loaded):
    gen_cfg = GenerationConfig(max_reasoning_budget=24, decision_interval=8,
                               do_sample=False, answer_max_tokens=5)
    traj = generate_trajectory(
        tiny_loaded, "test-0", "What is 2+2?", "4", gen_cfg, _spec(),
        source_split="test", sample_index=0,
    )
    assert traj.example_id == "test-0" and traj.source_split == "test"
    steps = [c.cumulative_reasoning_tokens for c in traj.checkpoints]
    assert steps[0] == 0                       # a budget-0 (direct answer) checkpoint
    assert steps == sorted(steps)              # ordered, coherent chain
    assert traj.max_tokens <= 24               # budget cap respected (§15)
    for cp in traj.checkpoints:
        assert cp.hidden[-1].ndim == 1         # decision-point hidden state present
        assert isinstance(cp.correct, bool)
