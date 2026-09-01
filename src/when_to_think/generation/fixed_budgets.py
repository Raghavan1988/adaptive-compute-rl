"""Budget-forced fixed-budget generation for the counterfactual dataset (M1).

Protocol (research-significant — AGENTS.md §25):

1. Reasoning phase: generate up to `budget` tokens continuing the reasoning prompt.
   Stops early on EOS ("finished naturally"); otherwise truncates at the budget.
2. Answer-elicitation phase: if the model did NOT finish naturally with a clean
   parseable answer (or budget == 0), append `answer_cue` and greedily generate up
   to `answer_max_tokens` to force a final answer.

Why forcing: the README lists "0 / direct answer" as a budget and compares budgets
at matched compute. A truncated reasoning trace has no clean final answer, so
taking "the last number" would score noise. Forcing gives every budget a fair,
comparable chance to state an answer.

Compute accounting keeps the varied quantity clean: `reasoning_tokens` (phase 1,
the x-axis of the accuracy-compute curve) is separate from `answer_tokens` (phase 2
overhead). Both, and their sum, are logged. The decision-point hidden state is taken
at the last reasoning-content token (EOS stripped) — the state from which a
STOP/CONTINUE choice would be made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from when_to_think.config import GenerationConfig
from when_to_think.generation.generate import build_prompt, enforce_budget
from when_to_think.models.loader import LoadedModel
from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    extract_hidden_states,
)
from when_to_think.rewards.answer_extraction import extract_numeric_answer


@dataclass
class BudgetRunOutput:
    example_id: str
    budget: int
    sample_index: int
    reasoning_text: str
    answer_text: str
    prediction: str | None
    prompt_tokens: int
    reasoning_tokens: int  # phase-1 generated (<= budget); the compute proxy varied
    answer_tokens: int  # phase-2 forced-answer tokens (overhead)
    total_generated_tokens: int
    finished_naturally: bool  # emitted EOS during phase 1
    forced_answer: bool  # phase 2 ran
    latency_s: float
    last_hidden_states: dict[int, np.ndarray]


def _strip_trailing_eos(ids: torch.Tensor, eos_token_id: int | None) -> torch.Tensor:
    """Drop a single trailing EOS so the forcing cue / hidden state use real content."""
    if eos_token_id is not None and ids.shape[1] > 0 and ids[0, -1].item() == eos_token_id:
        return ids[:, :-1]
    return ids


@torch.no_grad()
def generate_at_budget(
    loaded: LoadedModel,
    example_id: str,
    question: str,
    budget: int,
    gen_cfg: GenerationConfig,
    rep_spec: RepresentationDescriptor,
    *,
    sample_index: int = 0,
) -> BudgetRunOutput:
    """Run one (example, budget, sample) with the budget-forced protocol."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    eos_id = tokenizer.eos_token_id

    prompt = build_prompt(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    start = time.perf_counter()

    # --- Phase 1: reasoning up to `budget` tokens -------------------------------
    if budget > 0:
        reasoned = model.generate(
            **enc,
            max_new_tokens=budget,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        reasoning_ids = reasoned[:, prompt_len:]
        reasoning_tokens = int(reasoning_ids.shape[1])
        # generate() stops early only on EOS, so fewer tokens than the budget means
        # it finished naturally; hitting the budget means it was truncated.
        finished_naturally = reasoning_tokens < budget
        reasoning_text = tokenizer.decode(reasoning_ids[0], skip_special_tokens=True)
    else:
        reasoned = enc["input_ids"]
        reasoning_tokens = 0
        finished_naturally = False
        reasoning_text = ""

    enforce_budget(reasoning_tokens, budget)

    # Base = prompt + reasoning content, trailing EOS stripped. Used for both the
    # decision-point hidden state and the answer-forcing continuation.
    base_ids = _strip_trailing_eos(reasoned, eos_id)
    base_mask = torch.ones_like(base_ids)

    # --- Decision-point hidden state (last reasoning-content token) --------------
    forward = model(base_ids, attention_mask=base_mask, output_hidden_states=True)
    per_layer = extract_hidden_states(forward.hidden_states, base_mask, rep_spec)
    last_hidden_states = {layer: vec[0] for layer, vec in per_layer.items()}

    # --- Phase 2: force an answer if needed -------------------------------------
    prelim_prediction = extract_numeric_answer(reasoning_text) if reasoning_text else None
    need_force = budget == 0 or not finished_naturally or prelim_prediction is None

    answer_tokens = 0
    answer_text = ""
    prediction = prelim_prediction
    if need_force:
        cue_ids = tokenizer(
            gen_cfg.answer_cue, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        forced_input = torch.cat([base_ids, cue_ids], dim=1)
        forced = model.generate(
            input_ids=forced_input,
            attention_mask=torch.ones_like(forced_input),
            max_new_tokens=gen_cfg.answer_max_tokens,
            do_sample=False,  # deterministic answer given the (sampled) reasoning
            pad_token_id=tokenizer.pad_token_id,
        )
        answer_ids = forced[:, forced_input.shape[1]:]
        answer_tokens = int(answer_ids.shape[1])
        answer_text = gen_cfg.answer_cue + tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        forced_prediction = extract_numeric_answer(answer_text)
        # Prefer the forced answer; fall back to any answer seen during reasoning.
        prediction = forced_prediction if forced_prediction is not None else prelim_prediction

    latency = time.perf_counter() - start

    return BudgetRunOutput(
        example_id=example_id,
        budget=budget,
        sample_index=sample_index,
        reasoning_text=reasoning_text,
        answer_text=answer_text,
        prediction=prediction,
        prompt_tokens=prompt_len,
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
        total_generated_tokens=reasoning_tokens + answer_tokens,
        finished_naturally=finished_naturally,
        forced_answer=need_force,
        latency_s=latency,
        last_hidden_states=last_hidden_states,
    )
