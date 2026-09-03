"""Build the supervised value-of-compute probe dataset from an M1 fixed-budget run (M3).

Each *decision-point instance* asks the M3 question at one point in one trajectory:
given the hidden state after ``stop_budget`` reasoning tokens, is it worth continuing?
Features are the frozen hidden state; the label is the *value of compute*, defined two
explicit, distinct ways (PLAN.md M3):

- ``value_of_compute`` (regression): Δ = P(correct | continue) − P(correct | stop),
  the marginal change in accuracy from granting one more budget increment.
- ``fixes_incorrect`` (binary): 1 iff continuing turns a likely-wrong answer into a
  likely-right one — P(correct | stop) < τ and P(correct | continue) ≥ τ.

Both are **population** quantities estimated from the counterfactual samples, and both
are deliberately kept distinct from "P(the current answer is correct)" (``p_stop``,
also recorded): a probe that predicts current correctness is solving a different task
than one that predicts the *value of continuing* (AGENTS.md §4.5, PLAN.md M3).

Split membership is derived from ``example_id`` (``"test-42"`` → ``"test"``), which the
GSM8K loader stamps as ``"{source_split}-{index}"``. Grouping by example keeps every
instance of an example in a single split, so the delta target — which spans an example's
budgets — never leaks across the train/val/test boundary (AGENTS.md §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from when_to_think.representations.reader import HiddenStateReader


def split_of(example_id: str) -> str:
    """Recover the source split from an example id (``"{split}-{index}"``)."""
    head, sep, _tail = example_id.rpartition("-")
    return head if sep else "unknown"


def _per_example_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """P(correct | example, budget) estimated as mean correctness over samples."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(row["example_id"], {}).setdefault(row["budget"], []).append(
            1.0 if row["correct"] else 0.0
        )
    return {
        eid: {b: float(np.mean(v)) for b, v in by_b.items()}
        for eid, by_b in grouped.items()
    }


@dataclass
class ProbeDataset:
    """Parallel arrays over decision-point instances plus per-instance metadata.

    ``X`` is (n, hidden). ``y_value`` and ``y_binary`` are the two targets. ``splits``
    holds the train/val/test label per instance; ``groups`` the example id (for
    leakage checks and grouped analysis). ``meta`` carries everything the baselines and
    per-example prediction files need (question, token counts, budgets, probabilities).
    """

    X: np.ndarray
    y_value: np.ndarray
    y_binary: np.ndarray
    splits: np.ndarray
    groups: np.ndarray
    budgets: np.ndarray
    layer: int
    continue_mode: str
    correct_threshold: float
    meta: list[dict[str, Any]] = field(default_factory=list)

    def split_mask(self, split: str) -> np.ndarray:
        return self.splits == split

    def present_splits(self) -> list[str]:
        return sorted(set(self.splits.tolist()))

    def assert_disjoint_groups(self) -> None:
        """Fail loudly if any example appears in more than one split (leakage guard)."""
        by_group: dict[str, set[str]] = {}
        for g, s in zip(self.groups.tolist(), self.splits.tolist(), strict=True):
            by_group.setdefault(g, set()).add(s)
        leaked = {g: sorted(s) for g, s in by_group.items() if len(s) > 1}
        if leaked:
            raise ValueError(f"Examples span multiple splits (leakage): {leaked}")


def build_probe_dataset(
    rows: list[dict[str, Any]],
    reader: HiddenStateReader,
    *,
    layer: int = -1,
    continue_mode: str = "next",
    correct_threshold: float = 0.5,
) -> ProbeDataset:
    """Join outcome rows to hidden states and emit decision-point instances.

    One instance per (example, stop_budget, sample) whose hidden state is present and
    for which a valid "continue" budget exists. ``continue_mode`` picks that budget:

    - ``"next"``: the next larger budget the example has — the marginal value of one
      increment (the quantity a STOP/CONTINUE policy actually faces).
    - ``"max"``: the largest budget available — the total value of continuing to the cap.
    """
    if continue_mode not in ("next", "max"):
        raise ValueError(f"continue_mode must be 'next' or 'max', got {continue_mode!r}")

    acc = _per_example_accuracy(rows)

    feats: list[np.ndarray] = []
    y_value: list[float] = []
    y_binary: list[int] = []
    splits: list[str] = []
    groups: list[str] = []
    budgets: list[int] = []
    meta: list[dict[str, Any]] = []

    for row in rows:
        eid = row["example_id"]
        stop_b = row["budget"]
        by_budget = acc[eid]
        larger = sorted(b for b in by_budget if b > stop_b)
        if not larger:
            continue  # no decision to make past the largest budget
        cont_b = larger[0] if continue_mode == "next" else larger[-1]

        vec = reader.get(
            eid, budget=stop_b, sample_index=row.get("sample_index"), layer=layer
        )
        if vec is None:
            continue  # hidden state not stored for this instance; skip (kept as a gap)

        p_stop = by_budget[stop_b]
        p_cont = by_budget[cont_b]
        delta = p_cont - p_stop
        fixes = int(p_stop < correct_threshold <= p_cont)

        feats.append(np.asarray(vec, dtype=np.float64).ravel())
        y_value.append(delta)
        y_binary.append(fixes)
        splits.append(split_of(eid))
        groups.append(eid)
        budgets.append(stop_b)
        meta.append(
            {
                "example_id": eid,
                "sample_index": row.get("sample_index"),
                "stop_budget": stop_b,
                "continue_budget": cont_b,
                "p_stop": p_stop,
                "p_continue": p_cont,
                "delta_value": delta,
                "fixes_incorrect": fixes,
                # Distinct from the target: this sample's own correctness at the stop
                # budget (the "current answer correct" signal, not the value of compute).
                "current_correct": bool(row["correct"]),
                "question": row.get("question", ""),
                "prompt_tokens": row.get("prompt_tokens"),
                "reasoning_tokens": row.get("reasoning_tokens"),
            }
        )

    if not feats:
        raise ValueError(
            "No probe instances built — need at least two budgets and stored hidden "
            "states. Run the fixed-budget sweep with multiple fixed_budgets first."
        )

    dataset = ProbeDataset(
        X=np.vstack(feats),
        y_value=np.asarray(y_value, dtype=np.float64),
        y_binary=np.asarray(y_binary, dtype=np.int64),
        splits=np.asarray(splits, dtype=object),
        groups=np.asarray(groups, dtype=object),
        budgets=np.asarray(budgets, dtype=np.int64),
        layer=layer,
        continue_mode=continue_mode,
        correct_threshold=correct_threshold,
        meta=meta,
    )
    dataset.assert_disjoint_groups()
    return dataset
