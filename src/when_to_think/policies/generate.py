"""Orchestrate M4 trajectory generation: run the SLM, write checkpointed trajectories.

Parallels the M1 fixed-budget orchestrator, but produces coherent STOP/CONTINUE
trajectories (one rollout per example, checkpointed every ``decision_interval`` tokens)
instead of independent per-budget samples. Writes them under ``run_dir/trajectories/``
plus a ``run_record.json``. Defaults to all three splits — the policy needs TRAIN data,
not just TEST (§4.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from when_to_think.config import ExperimentConfig
from when_to_think.data.gsm8k import load_gsm8k
from when_to_think.generation.incremental import generate_trajectory
from when_to_think.models.loader import load_model_and_tokenizer
from when_to_think.policies.data import write_trajectories
from when_to_think.representations.extraction import RepresentationDescriptor
from when_to_think.utils.run_record import create_run_record, write_run_record
from when_to_think.utils.seeding import seed_everything

METHOD = "stop_continue_trajectories"


def run_trajectory_generation(
    cfg: ExperimentConfig,
    *,
    repo_dir: str | Path | None = None,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Path:
    """Generate checkpointed trajectories for the requested splits; return the run dir."""
    seed_everything(cfg.seed)
    loaded = load_model_and_tokenizer(cfg.model)
    dataset_splits = load_gsm8k(cfg.data)

    valid = {"train": dataset_splits.train, "val": dataset_splits.val, "test": dataset_splits.test}
    unknown = set(splits) - set(valid)
    if unknown:
        raise ValueError(f"Unknown split(s) {sorted(unknown)}; choose from {sorted(valid)}")
    selected = list(dict.fromkeys(splits))

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
            "split_sizes": dataset_splits.sizes(),
            "method": METHOD,
            "eval_splits": selected,
            "decision_interval": cfg.generation.decision_interval,
            "max_reasoning_budget": cfg.generation.max_reasoning_budget,
            "num_samples": cfg.generation.num_samples,
        },
    )
    run_dir = write_run_record(record, cfg.output_dir)

    trajectories = []
    for split_name in selected:
        for example in valid[split_name]:
            for sample_index in range(cfg.generation.num_samples):
                trajectories.append(generate_trajectory(
                    loaded, example.example_id, example.question, example.gold_answer,
                    cfg.generation, rep_spec,
                    source_split=split_name, sample_index=sample_index,
                ))
    write_trajectories(run_dir, trajectories, rep_spec)
    return run_dir
