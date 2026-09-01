"""Single-pass fixed-budget evaluation → machine-readable per-example JSONL (§18).

This is the M0 end-to-end pipeline: seed everything, load the frozen SLM and the
GSM8K splits, generate one reasoning pass per example, score it with rule-based
exact match, compute the reward across the full lambda sweep, and stream both the
per-example results (JSONL) and the decision-point hidden states (sharded .npz) to
a per-run output directory alongside a reproducibility run record.

It evaluates the TEST split — evaluating on test is allowed; only TRAINING on test
is forbidden (AGENTS.md §4.2). Failures (malformed answers, wrong answers) are
recorded, never dropped (§4.4).
"""

from __future__ import annotations

import json
from pathlib import Path

from when_to_think.config import ExperimentConfig
from when_to_think.data.gsm8k import load_gsm8k
from when_to_think.generation.generate import generate_single
from when_to_think.models.loader import load_model_and_tokenizer
from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
)
from when_to_think.rewards.answer_extraction import answers_match, extract_numeric_answer
from when_to_think.rewards.reward import compute_reward_sweep
from when_to_think.utils.run_record import create_run_record, write_run_record
from when_to_think.utils.seeding import seed_everything

METHOD = "single_pass_fixed_budget"


def run_evaluation(cfg: ExperimentConfig, *, repo_dir: str | Path | None = None) -> Path:
    """Run the M0 pipeline end-to-end; return the run directory."""
    seed_everything(cfg.seed)

    loaded = load_model_and_tokenizer(cfg.model)
    splits = load_gsm8k(cfg.data)

    rep_spec = RepresentationDescriptor(
        layers=cfg.representation.layers,
        token_position=cfg.representation.token_position,
        pooling=cfg.representation.pooling,
        model_name=loaded.model_name,
        model_revision=loaded.revision,
    )

    # Record resolved runtime facts config alone can't capture (§9).
    record = create_run_record(
        cfg,
        repo_dir=repo_dir,
        runtime={
            "model_revision": loaded.revision,
            "resolved_dtype": loaded.resolved_dtype,
            "device": loaded.device,
            "split_sizes": splits.sizes(),
            "method": METHOD,
            "eval_budget": cfg.generation.max_reasoning_budget,
        },
    )
    run_dir = write_run_record(record, cfg.output_dir)

    eval_path = run_dir / "eval.jsonl"
    hidden_dir = run_dir / "hidden_states"

    with (
        open(eval_path, "w") as out,
        ShardedRepresentationWriter(hidden_dir, rep_spec) as hidden_writer,
    ):
        for example in splits.test:
            result = generate_single(
                loaded, example.example_id, example.question, cfg.generation, rep_spec
            )
            prediction = extract_numeric_answer(result.completion_text)
            correct = answers_match(prediction, example.gold_answer)
            rewards = compute_reward_sweep(
                correct=correct,
                compute_units=result.reasoning_tokens,
                reward_config=cfg.reward,
            )

            row = {
                "example_id": example.example_id,
                "question": example.question,
                "ground_truth": example.gold_answer,
                "method": METHOD,
                "seed": cfg.seed,
                "budget": result.budget,
                "prompt_tokens": result.prompt_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "hit_budget": result.hit_budget,
                "latency_s": result.latency_s,
                "prediction": prediction,
                "correct": correct,
                "compute_proxy": cfg.reward.compute_proxy,
                # reward_task is lambda-independent; compute/total vary with lambda,
                # so the sweep is stored as a list rather than flat fields (§7).
                "reward_task": rewards[0].reward_task,
                "rewards_by_lambda": [
                    {
                        "lambda_compute": r.lambda_compute,
                        "reward_compute": r.reward_compute,
                        "reward_total": r.reward_total,
                    }
                    for r in rewards
                ],
                # Single-pass baseline has no STOP/CONTINUE trajectory yet (M4).
                "actions": [],
            }
            out.write(json.dumps(row) + "\n")
            hidden_writer.add(
                example.example_id,
                reasoning_step=0,
                layer_vectors=result.last_hidden_states,
            )

    return run_dir
