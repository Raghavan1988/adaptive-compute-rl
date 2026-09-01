"""Single-pass fixed-budget generation with budget enforcement (AGENTS.md §15, §22).

For M0 this generates one reasoning pass per question up to a fixed token budget,
then captures the hidden state at the final token as the decision-point
representation. It deliberately does NOT implement STOP/CONTINUE — that is the RL
environment in M4. The compute proxy is the number of generated reasoning tokens
(never called FLOPs, §7).

Budget enforcement is explicit: `max_new_tokens` caps generation, and we assert the
generated length never exceeds the configured budget so a silent overrun is caught
rather than mislabeled as a compute measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from when_to_think.config import GenerationConfig
from when_to_think.models.loader import LoadedModel
from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    extract_hidden_states,
)

# Kept explicit so the answer-extraction contract (a '####' marker) is visible and
# versioned; changing this prompt is a research-significant change (AGENTS.md §25).
_INSTRUCTION = (
    "Solve the problem step by step. "
    "Then give the final numeric answer on its own line after '#### '."
)


@dataclass
class GenerationOutput:
    example_id: str
    prompt: str
    completion_text: str
    prompt_tokens: int
    reasoning_tokens: int  # generated tokens = the compute proxy for this pass
    budget: int
    hit_budget: bool  # True if generation stopped because it reached the budget
    latency_s: float
    last_hidden_states: dict[int, np.ndarray]  # per requested layer, shape (hidden,)


def build_prompt(tokenizer, question: str) -> str:
    """Render the question into a prompt, using the chat template when available."""
    content = f"{question}\n\n{_INSTRUCTION}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return content + "\n"


def enforce_budget(reasoning_tokens: int, budget: int) -> None:
    """Fail loudly if generation exceeded its configured budget (§15)."""
    if reasoning_tokens > budget:
        raise RuntimeError(
            f"Generation produced {reasoning_tokens} tokens, exceeding budget {budget}"
        )


@torch.no_grad()
def generate_single(
    loaded: LoadedModel,
    example_id: str,
    question: str,
    gen_cfg: GenerationConfig,
    rep_spec: RepresentationDescriptor,
    *,
    budget: int | None = None,
) -> GenerationOutput:
    """Generate one reasoning pass and capture the final-token hidden state.

    `budget` defaults to `gen_cfg.max_reasoning_budget`; M1 will call this per fixed
    budget in the sweep. A budget of 0 means "answer directly" (no reasoning tokens).
    """
    budget = gen_cfg.max_reasoning_budget if budget is None else budget
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device

    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    generated = model.generate(
        **inputs,
        max_new_tokens=max(budget, 1),  # generate>=1 token; budget=0 handled below
        do_sample=gen_cfg.do_sample and budget > 0,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency = time.perf_counter() - start

    generated_ids = generated[:, prompt_len:]
    reasoning_tokens = 0 if budget == 0 else int(generated_ids.shape[1])
    enforce_budget(reasoning_tokens, budget)
    completion_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Decision-point representation: hidden state at the final token of the full
    # (prompt + completion) sequence. Single sequence => no padding, mask all ones.
    full_ids = generated
    attention_mask = torch.ones_like(full_ids)
    forward = model(full_ids, attention_mask=attention_mask, output_hidden_states=True)
    per_layer = extract_hidden_states(forward.hidden_states, attention_mask, rep_spec)
    last_hidden_states = {layer: vec[0] for layer, vec in per_layer.items()}

    return GenerationOutput(
        example_id=example_id,
        prompt=prompt,
        completion_text=completion_text,
        prompt_tokens=prompt_len,
        reasoning_tokens=reasoning_tokens,
        budget=budget,
        hit_budget=(reasoning_tokens >= budget and budget > 0),
        latency_s=latency,
        last_hidden_states=last_hidden_states,
    )
