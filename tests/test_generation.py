"""Tests for generation budget enforcement (AGENTS.md §12, §15).

The pure enforcement check runs always. The end-to-end generation check needs a
(tiny) model and is skipped when it cannot be loaded (e.g. offline).
"""

import pytest

from when_to_think.config import GenerationConfig, ModelConfig
from when_to_think.generation import enforce_budget, generate_single
from when_to_think.models import load_model_and_tokenizer
from when_to_think.representations import RepresentationDescriptor


def test_enforce_budget_raises_on_overrun():
    enforce_budget(64, 64)  # exactly at budget is allowed
    enforce_budget(10, 64)  # under budget is allowed
    with pytest.raises(RuntimeError, match="exceeding budget"):
        enforce_budget(65, 64)


@pytest.fixture(scope="module")
def tiny_loaded():
    cfg = ModelConfig(name="hf-internal-testing/tiny-random-gpt2", dtype="float32", device="cpu")
    try:
        return load_model_and_tokenizer(cfg)
    except Exception as exc:  # noqa: BLE001 - any load failure (offline, hub) => skip
        pytest.skip(f"tiny model unavailable: {exc}")


def test_generation_respects_budget(tiny_loaded):
    spec = RepresentationDescriptor(
        layers=[-1], token_position="last", pooling=None,
        model_name="tiny", model_revision=None,
    )
    gen_cfg = GenerationConfig(max_reasoning_budget=8, do_sample=False)
    out = generate_single(tiny_loaded, "ex-0", "What is 2+2?", gen_cfg, spec, budget=8)
    # Budget is a hard cap; never silently exceeded.
    assert out.reasoning_tokens <= 8
    assert out.budget == 8
    # A decision-point hidden vector was captured for the requested layer.
    assert out.last_hidden_states[-1].ndim == 1
    assert out.last_hidden_states[-1].shape[0] > 0
