"""Incremental trajectory generation with decision-point checkpoints (M4).

Generates ONE coherent reasoning rollout per example up to ``max_reasoning_budget``,
then records a checkpoint every ``decision_interval`` reasoning tokens (plus one at the
natural end). Because the policy only decides *when to stop* — never what is generated —
a single trajectory checkpointed along the way fully specifies the STOP/CONTINUE
environment: continuing just reveals the next checkpoint of the same rollout.

At each checkpoint we record (a) the decision-point hidden state and (b) the provisional
answer that STOPping there would produce, scored by rule-based exact match. The answer
elicitation mirrors the M1 budget-forced protocol (append ``answer_cue``, greedily
generate) so "STOP now" is scored the same way a fixed budget is (§4.1). The elicitation
is a side branch — it never extends the reasoning, so the trajectory stays coherent.

All decision-point hidden states come from a SINGLE forward pass over the full
prompt+reasoning sequence (causal attention makes the state at prefix length L the
state after L reasoning tokens), so only the K answer elicitations cost extra generation.
``token_position`` must be ``last`` (the decision-point convention).
"""

from __future__ import annotations

import numpy as np
import torch

from when_to_think.config import GenerationConfig
from when_to_think.generation.fixed_budgets import _strip_trailing_eos
from when_to_think.generation.generate import build_prompt
from when_to_think.models.loader import LoadedModel
from when_to_think.policies.data import Checkpoint, Trajectory
from when_to_think.representations.extraction import RepresentationDescriptor
from when_to_think.rewards.answer_extraction import answers_match, extract_numeric_answer


def _checkpoint_prefix_lengths(reasoning_len: int, decision_interval: int) -> list[int]:
    """Reasoning-token prefix lengths at each decision point: 0, I, 2I, …, end."""
    lengths = [0]
    k = 1
    while k * decision_interval < reasoning_len:
        lengths.append(k * decision_interval)
        k += 1
    lengths.append(reasoning_len)  # always checkpoint the natural end
    # Deduplicate while preserving ascending order (handles reasoning_len == 0 / multiples).
    seen: set[int] = set()
    return [x for x in lengths if not (x in seen or seen.add(x))]


@torch.no_grad()
def generate_trajectory(
    loaded: LoadedModel,
    example_id: str,
    question: str,
    gold_answer: str,
    gen_cfg: GenerationConfig,
    rep_spec: RepresentationDescriptor,
    *,
    source_split: str,
    sample_index: int = 0,
) -> Trajectory:
    """Generate one checkpointed reasoning trajectory for an example."""
    if rep_spec.token_position != "last":
        raise ValueError("trajectory generation requires token_position='last'")
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    eos_id = tokenizer.eos_token_id

    prompt = build_prompt(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # --- One reasoning rollout to the budget cap -------------------------------
    if gen_cfg.max_reasoning_budget > 0:
        reasoned = model.generate(
            **enc,
            max_new_tokens=gen_cfg.max_reasoning_budget,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        reasoned = enc["input_ids"]
    reasoning_content = _strip_trailing_eos(reasoned, eos_id)[:, prompt_len:]
    reasoning_len = int(reasoning_content.shape[1])
    finished_naturally = reasoning_len < gen_cfg.max_reasoning_budget

    full_ids = torch.cat([enc["input_ids"], reasoning_content], dim=1)
    full_mask = torch.ones_like(full_ids)

    # --- All decision-point hidden states in one forward -----------------------
    forward = model(full_ids, attention_mask=full_mask, output_hidden_states=True)
    num_layers = len(forward.hidden_states)

    def hidden_at(prefix_len: int) -> dict[int, np.ndarray]:
        pos = prompt_len + prefix_len - 1  # last token of prompt+prefix (>=0)
        out = {}
        for layer in rep_spec.layers:
            idx = layer if layer >= 0 else num_layers + layer
            out[layer] = forward.hidden_states[idx][0, pos].to(torch.float32).cpu().numpy()
        return out

    checkpoints: list[Checkpoint] = []
    for step_index, prefix_len in enumerate(_checkpoint_prefix_lengths(
        reasoning_len, gen_cfg.decision_interval
    )):
        base_ids = full_ids[:, : prompt_len + prefix_len]
        prediction = _elicit_answer(loaded, base_ids, gen_cfg)
        checkpoints.append(Checkpoint(
            step_index=step_index,
            cumulative_reasoning_tokens=prefix_len,
            correct=answers_match(prediction, gold_answer),
            prediction=prediction,
            finished_naturally=(prefix_len == reasoning_len and finished_naturally),
            hidden=hidden_at(prefix_len),
        ))

    return Trajectory(
        example_id=example_id,
        source_split=source_split,
        sample_index=sample_index,
        prompt_tokens=prompt_len,
        gold_answer=gold_answer,
        checkpoints=checkpoints,
    )


@torch.no_grad()
def _elicit_answer(
    loaded: LoadedModel, base_ids: torch.Tensor, gen_cfg: GenerationConfig
) -> str | None:
    """Force a final answer from a reasoning prefix (greedy), return the parsed answer.

    A side branch off the reasoning prefix (mirrors the M1 forcing protocol): append the
    answer cue and greedily decode, so STOPping here is scored like a fixed budget.
    """
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
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
    answer_text = gen_cfg.answer_cue + tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return extract_numeric_answer(answer_text)
