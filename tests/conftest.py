"""Shared fixtures for probe tests: synthesize a fixed-budget run on disk.

Builds a ``fixed_budget_runs.jsonl`` + ``hidden_states/`` directory equivalent to what
``run_fixed_budget_sweep`` writes, but without a model — so the M3 probe pipeline can be
tested deterministically and offline.
"""

import json

import numpy as np
import pytest

from when_to_think.policies.data import Checkpoint, Trajectory
from when_to_think.representations import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
)


@pytest.fixture
def make_probe_run(tmp_path):
    """Return a factory writing synthetic records to a fresh run dir; returns its path.

    ``records`` is a list of dicts with keys: example_id, budget, sample_index, correct
    (bool), and optionally hidden (np.ndarray for layer -1), question, prompt_tokens,
    reasoning_tokens. Only budget/sample_index/correct/hidden feed the probe; the rest
    default to sane placeholders.
    """
    counter = {"n": 0}

    def _factory(records, *, hidden_dim=6):
        counter["n"] += 1
        run_dir = tmp_path / f"run_{counter['n']}"
        run_dir.mkdir()
        desc = RepresentationDescriptor(
            layers=[-1], token_position="last", pooling=None,
            model_name="synthetic", model_revision=None,
        )
        with (
            open(run_dir / "fixed_budget_runs.jsonl", "w") as out,
            ShardedRepresentationWriter(run_dir / "hidden_states", desc, shard_size=8) as hw,
        ):
            for r in records:
                eid = r["example_id"]
                out.write(json.dumps({
                    "example_id": eid,
                    "source_split": eid.rpartition("-")[0],
                    "budget": r["budget"],
                    "sample_index": r["sample_index"],
                    "correct": bool(r["correct"]),
                    "question": r.get("question", "q " * (r["budget"] % 5 + 1)),
                    "prompt_tokens": r.get("prompt_tokens", 10),
                    "reasoning_tokens": r.get("reasoning_tokens", r["budget"]),
                }) + "\n")
                hidden = r.get("hidden")
                if hidden is None:
                    hidden = np.zeros(hidden_dim, dtype=np.float32)
                hw.add(eid, reasoning_step=0, layer_vectors={-1: np.asarray(hidden, np.float32)},
                       budget=r["budget"], sample_index=r["sample_index"])
        return run_dir

    return _factory


@pytest.fixture
def make_trajectory():
    """Factory: build a Trajectory from a list of (cum_tokens, correct, signal) steps.

    ``signal`` is written into dim 0 of the (layer -1) hidden state so tests can control
    what the policy should learn; remaining dims are optional light noise.
    """

    def _factory(example_id, source_split, steps, *, sample_index=0, hidden_dim=4, rng=None):
        cps = []
        for k, (tokens, correct, signal) in enumerate(steps):
            vec = np.zeros(hidden_dim, dtype=np.float32)
            vec[0] = signal
            if rng is not None and hidden_dim > 1:
                vec[1:] = rng.normal(0, 0.1, size=hidden_dim - 1)
            cps.append(Checkpoint(
                step_index=k, cumulative_reasoning_tokens=tokens, correct=bool(correct),
                prediction=("42" if correct else "0"), finished_naturally=False,
                hidden={-1: vec},
            ))
        return Trajectory(
            example_id=example_id, source_split=source_split, sample_index=sample_index,
            prompt_tokens=5, gold_answer="42", checkpoints=cps,
        )

    return _factory
