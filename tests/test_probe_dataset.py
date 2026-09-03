"""Probe dataset construction: targets, split derivation, leakage guard, continue mode."""

import json

import numpy as np
import pytest

from when_to_think.evaluation import load_runs
from when_to_think.probes.dataset import build_probe_dataset, split_of
from when_to_think.representations import HiddenStateReader


def _records_two_budgets(example_id, p_stop_correct, p_cont_correct, n=2, hidden0=None):
    """n samples at budget 0 (each correct=p_stop_correct[i]) and budget 128 (=p_cont)."""
    recs = []
    for s in range(n):
        recs.append({"example_id": example_id, "budget": 0, "sample_index": s,
                     "correct": p_stop_correct[s],
                     "hidden": hidden0[s] if hidden0 is not None else np.full(4, float(s))})
        recs.append({"example_id": example_id, "budget": 128, "sample_index": s,
                     "correct": p_cont_correct[s]})
    return recs


def test_split_of():
    assert split_of("test-42") == "test"
    assert split_of("train-0") == "train"
    assert split_of("val-7") == "val"
    assert split_of("noseparator") == "unknown"


def test_targets_delta_and_fixes_incorrect(make_probe_run):
    # Example that improves: wrong at budget 0 (p_stop=0), right at budget 128 (p_cont=1).
    recs = _records_two_budgets("test-0", [False, False], [True, True])
    # Example already correct at both (p_stop=1, p_cont=1): no fix, delta 0.
    recs += _records_two_budgets("test-1", [True, True], [True, True])
    run_dir = make_probe_run(recs)

    rows = load_runs(run_dir / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(run_dir / "hidden_states")
    ds = build_probe_dataset(rows, reader, layer=-1)

    # Only budget-0 rows become instances (budget 128 has no larger budget).
    assert set(ds.budgets.tolist()) == {0}
    by_ex = {eid: [] for eid in ("test-0", "test-1")}
    for m in ds.meta:
        by_ex[m["example_id"]].append(m)

    for m in by_ex["test-0"]:
        assert m["delta_value"] == pytest.approx(1.0)  # 1.0 - 0.0
        assert m["fixes_incorrect"] == 1               # 0 < 0.5 <= 1
    for m in by_ex["test-1"]:
        assert m["delta_value"] == pytest.approx(0.0)
        assert m["fixes_incorrect"] == 0               # already correct, not "fixed"


def test_continue_mode_next_vs_max(make_probe_run):
    # Three budgets: acc 0.0 @0, 0.0 @128, 1.0 @256. "next" from budget 0 sees no gain
    # (128); "max" sees the full gain (256). Distinguishes the two definitions.
    recs = []
    for s in range(1):
        recs.append({"example_id": "train-0", "budget": 0, "sample_index": s, "correct": False,
                     "hidden": np.ones(4)})
        recs.append({"example_id": "train-0", "budget": 128, "sample_index": s, "correct": False})
        recs.append({"example_id": "train-0", "budget": 256, "sample_index": s, "correct": True})
    run_dir = make_probe_run(recs)
    rows = load_runs(run_dir / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(run_dir / "hidden_states")

    ds_next = build_probe_dataset(rows, reader, layer=-1, continue_mode="next")
    ds_max = build_probe_dataset(rows, reader, layer=-1, continue_mode="max")
    m_next = next(m for m in ds_next.meta if m["stop_budget"] == 0)
    m_max = next(m for m in ds_max.meta if m["stop_budget"] == 0)
    assert m_next["continue_budget"] == 128 and m_next["delta_value"] == pytest.approx(0.0)
    assert m_max["continue_budget"] == 256 and m_max["delta_value"] == pytest.approx(1.0)


def test_leakage_guard_rejects_split_crossing_example(make_probe_run, tmp_path):
    run_dir = make_probe_run(_records_two_budgets("test-0", [False], [True], n=1))
    # Corrupt the JSONL: duplicate the example under a second split (a leak).
    runs_path = run_dir / "fixed_budget_runs.jsonl"
    rows = [json.loads(line) for line in runs_path.read_text().splitlines()]
    # Same example_id can't appear in two splits; forge an id whose split differs from
    # another sharing hidden states is hard — instead assert the guard directly.
    from when_to_think.probes.dataset import ProbeDataset

    ds = ProbeDataset(
        X=np.zeros((2, 3)), y_value=np.zeros(2), y_binary=np.zeros(2, int),
        splits=np.array(["train", "test"], dtype=object),
        groups=np.array(["e-1", "e-1"], dtype=object),
        budgets=np.zeros(2, int), layer=-1, continue_mode="next", correct_threshold=0.5,
    )
    with pytest.raises(ValueError, match="leakage"):
        ds.assert_disjoint_groups()


def test_empty_dataset_raises(make_probe_run):
    # A single budget → no decision points → no instances.
    recs = [{"example_id": "test-0", "budget": 0, "sample_index": 0, "correct": True,
             "hidden": np.ones(4)}]
    run_dir = make_probe_run(recs)
    rows = load_runs(run_dir / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(run_dir / "hidden_states")
    with pytest.raises(ValueError, match="No probe instances"):
        build_probe_dataset(rows, reader, layer=-1)
