"""Tests for budget-forced generation (M1). Model-dependent tests skip when offline."""

import numpy as np
import pytest

from when_to_think.config import GenerationConfig, ModelConfig
from when_to_think.generation.fixed_budgets import generate_at_budget
from when_to_think.models import load_model_and_tokenizer
from when_to_think.representations import RepresentationDescriptor


@pytest.fixture(scope="module")
def tiny_loaded():
    cfg = ModelConfig(name="hf-internal-testing/tiny-random-gpt2", dtype="float32", device="cpu")
    try:
        return load_model_and_tokenizer(cfg)
    except Exception as exc:  # noqa: BLE001 - any load failure (offline) => skip
        pytest.skip(f"tiny model unavailable: {exc}")


def _spec():
    return RepresentationDescriptor(
        layers=[-1], token_position="last", pooling=None,
        model_name="tiny", model_revision=None,
    )


def test_budget_zero_is_direct_answer(tiny_loaded):
    gen_cfg = GenerationConfig(do_sample=False, answer_max_tokens=5)
    out = generate_at_budget(tiny_loaded, "ex-0", "What is 2+2?", 0, gen_cfg, _spec())
    # No reasoning tokens, but the answer-forcing phase still runs to elicit an answer.
    assert out.reasoning_tokens == 0
    assert out.forced_answer is True
    assert out.answer_tokens > 0
    assert out.total_generated_tokens == out.answer_tokens
    # Decision-point hidden state captured (from the last prompt token).
    assert out.last_hidden_states[-1].ndim == 1


def test_budget_respected_and_hidden_state_present(tiny_loaded):
    gen_cfg = GenerationConfig(do_sample=False, answer_max_tokens=5)
    out = generate_at_budget(tiny_loaded, "ex-1", "Compute 13 times 4.", 8, gen_cfg, _spec())
    assert out.reasoning_tokens <= 8
    assert out.total_generated_tokens == out.reasoning_tokens + out.answer_tokens
    assert isinstance(out.last_hidden_states[-1], np.ndarray)


def test_negative_budget_rejected(tiny_loaded):
    with pytest.raises(ValueError):
        generate_at_budget(tiny_loaded, "ex", "q", -1, GenerationConfig(), _spec())
