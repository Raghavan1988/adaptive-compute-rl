"""Trajectory checkpoint data for the STOP/CONTINUE environment (M4).

A *trajectory* is one coherent reasoning rollout of the frozen SLM for a single
example, sampled once. It is recorded as a chain of *checkpoints*, one per decision
point (every ``decision_interval`` reasoning tokens). Each checkpoint stores what a
STOP at that point would yield — the provisional answer's correctness and the
cumulative reasoning tokens spent — plus the decision-point hidden state the policy
conditions on.

The key property that makes the environment faithful yet offline (AGENTS.md §4.1,
§4.5): the policy never changes *what* the SLM generates, only *when to stop*. So one
trajectory generated to the max budget, checkpointed along the way, fully determines
the reward of *any* stop decision on it — CONTINUE simply reveals the next checkpoint
of the same rollout. No branching, no monotonicity assumed (a later checkpoint may be
wrong where an earlier one was right — that is kept, §4.4).

Storage reuses the sharded hidden-state writer for provenance; all scalar checkpoint
fields ride along in the manifest, so a run directory is self-describing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
)

TRAJECTORY_DIRNAME = "trajectories"


@dataclass
class Checkpoint:
    """One decision point on a trajectory: the state, and the outcome of STOPping here."""

    step_index: int
    cumulative_reasoning_tokens: int
    correct: bool
    prediction: str | None
    finished_naturally: bool
    hidden: dict[int, np.ndarray]  # per stored layer, shape (hidden,)


@dataclass
class Trajectory:
    """A coherent reasoning rollout for one example, as an ordered checkpoint chain."""

    example_id: str
    source_split: str
    sample_index: int
    prompt_tokens: int
    gold_answer: str
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.checkpoints.sort(key=lambda c: c.step_index)

    @property
    def max_tokens(self) -> int:
        return self.checkpoints[-1].cumulative_reasoning_tokens if self.checkpoints else 0


def write_trajectories(
    run_dir: str | Path,
    trajectories: list[Trajectory],
    rep_spec: RepresentationDescriptor,
) -> Path:
    """Persist trajectories under ``run_dir/trajectories/`` (shards + manifest)."""
    out = Path(run_dir) / TRAJECTORY_DIRNAME
    with ShardedRepresentationWriter(out, rep_spec) as writer:
        for traj in trajectories:
            for cp in traj.checkpoints:
                writer.add(
                    traj.example_id,
                    reasoning_step=cp.step_index,
                    layer_vectors=cp.hidden,
                    sample_index=traj.sample_index,
                    source_split=traj.source_split,
                    prompt_tokens=traj.prompt_tokens,
                    gold_answer=traj.gold_answer,
                    cumulative_reasoning_tokens=cp.cumulative_reasoning_tokens,
                    correct=cp.correct,
                    prediction=cp.prediction,
                    finished_naturally=cp.finished_naturally,
                )
    return out


def load_trajectories(run_dir: str | Path) -> list[Trajectory]:
    """Read trajectories back from ``run_dir/trajectories/`` into Trajectory objects."""
    traj_dir = Path(run_dir) / TRAJECTORY_DIRNAME
    manifest_path = traj_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No trajectory manifest at {manifest_path}")
    descriptor = json.loads((traj_dir / "descriptor.json").read_text())
    layers = descriptor["layers"]

    shard_cache: dict[str, dict[str, np.ndarray]] = {}

    def _vec(shard: str, row: int, layer: int) -> np.ndarray:
        arrays = shard_cache.get(shard)
        if arrays is None:
            with np.load(traj_dir / shard) as npz:
                arrays = {name: npz[name] for name in npz.files}
            shard_cache[shard] = arrays
        return arrays[f"layer_{layer}"][row]

    grouped: dict[tuple[str, int], Trajectory] = {}
    for r in (json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()):
        key = (r["example_id"], r["sample_index"])
        traj = grouped.get(key)
        if traj is None:
            traj = Trajectory(
                example_id=r["example_id"],
                source_split=r["source_split"],
                sample_index=r["sample_index"],
                prompt_tokens=r["prompt_tokens"],
                gold_answer=r["gold_answer"],
            )
            grouped[key] = traj
        traj.checkpoints.append(
            Checkpoint(
                step_index=r["reasoning_step"],
                cumulative_reasoning_tokens=r["cumulative_reasoning_tokens"],
                correct=bool(r["correct"]),
                prediction=r["prediction"],
                finished_naturally=bool(r["finished_naturally"]),
                hidden={layer: _vec(r["shard"], r["row"], layer) for layer in layers},
            )
        )
    trajectories = list(grouped.values())
    for traj in trajectories:
        traj.checkpoints.sort(key=lambda c: c.step_index)
    return trajectories
