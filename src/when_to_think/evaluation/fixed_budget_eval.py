"""M1 orchestrator: run the counterfactual fixed-budget sweep → JSONL + hidden states.

For every test example, every fixed budget, and every counterfactual sample, run the
budget-forced protocol, score with rule-based exact match, and record a per-run row
(AGENTS.md §18) plus the decision-point hidden state. The SAME examples are run at
every budget (§4.1), and every trajectory — right, wrong, malformed — is kept (§4.4).
"""

from __future__ import annotations

import json
from pathlib import Path

from when_to_think.config import ExperimentConfig
from when_to_think.data.gsm8k import load_gsm8k
from when_to_think.generation.fixed_budgets import generate_at_budget
from when_to_think.models.loader import load_model_and_tokenizer
from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
)
from when_to_think.rewards.answer_extraction import answers_match
from when_to_think.rewards.reward import compute_reward_sweep
from when_to_think.utils.run_record import create_run_record, write_run_record
from when_to_think.utils.seeding import seed_everything

METHOD = "fixed_budget_sweep"
RUNS_FILENAME = "fixed_budget_runs.jsonl"


def run_fixed_budget_sweep(cfg: ExperimentConfig, *, repo_dir: str | Path | None = None) -> Path:
    """Run the M1 sweep end-to-end; return the run directory."""
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

    record = create_run_record(
        cfg,
        repo_dir=repo_dir,
        runtime={
            "model_revision": loaded.revision,
            "resolved_dtype": loaded.resolved_dtype,
            "device": loaded.device,
            "split_sizes": splits.sizes(),
            "method": METHOD,
            "fixed_budgets": cfg.generation.fixed_budgets,
            "num_samples": cfg.generation.num_samples,
        },
    )
    run_dir = write_run_record(record, cfg.output_dir)

    runs_path = run_dir / RUNS_FILENAME
    with (
        open(runs_path, "w") as out,
        ShardedRepresentationWriter(run_dir / "hidden_states", rep_spec) as hidden_writer,
    ):
        for example in splits.test:
            for budget in cfg.generation.fixed_budgets:
                for sample_index in range(cfg.generation.num_samples):
                    result = generate_at_budget(
                        loaded,
                        example.example_id,
                        example.question,
                        budget,
                        cfg.generation,
                        rep_spec,
                        sample_index=sample_index,
                    )
                    correct = answers_match(result.prediction, example.gold_answer)
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
                        "budget": budget,
                        "sample_index": sample_index,
                        "prompt_tokens": result.prompt_tokens,
                        "reasoning_tokens": result.reasoning_tokens,
                        "answer_tokens": result.answer_tokens,
                        "total_generated_tokens": result.total_generated_tokens,
                        "finished_naturally": result.finished_naturally,
                        "forced_answer": result.forced_answer,
                        "latency_s": result.latency_s,
                        "prediction": result.prediction,
                        "correct": correct,
                        "compute_proxy": cfg.reward.compute_proxy,
                        "reward_task": rewards[0].reward_task,
                        "rewards_by_lambda": [
                            {
                                "lambda_compute": r.lambda_compute,
                                "reward_compute": r.reward_compute,
                                "reward_total": r.reward_total,
                            }
                            for r in rewards
                        ],
                    }
                    out.write(json.dumps(row) + "\n")
                    hidden_writer.add(
                        example.example_id,
                        reasoning_step=0,
                        layer_vectors=result.last_hidden_states,
                        budget=budget,
                        sample_index=sample_index,
                    )

    return run_dir
