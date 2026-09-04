"""End-to-end M3 probe: hidden state that encodes value-of-compute beats baselines,
with strict train/val/test discipline. Uses synthetic data (no model)."""

import numpy as np
import pytest

from when_to_think.config import ProbeConfig
from when_to_think.evaluation import load_runs
from when_to_think.probes.train import train_probe
from when_to_think.representations import HiddenStateReader

HIDDEN_DIM = 6


def _build_records(rng, n_per_split=20):
    """Half the examples 'fix' (wrong->right on continue); the hidden state encodes it.

    Budgets [0, 128], 4 counterfactual samples. Fixing examples are wrong at budget 0
    and right at budget 128 (delta=1, fixes=1); the rest are already right (delta=0,
    fixes=0). Dim 0 of the budget-0 hidden state carries the label + noise, so it is
    decodable; the question features are constant, so the input baseline cannot.
    """
    records = []
    for split in ("train", "val", "test"):
        for i in range(n_per_split):
            fixes = i % 2 == 0
            eid = f"{split}-{i}"
            for s in range(4):
                signal = (2.0 if fixes else -2.0) + rng.normal(0, 0.3)
                hidden = np.concatenate([[signal], rng.normal(0, 0.3, size=HIDDEN_DIM - 1)])
                records.append({"example_id": eid, "budget": 0, "sample_index": s,
                                "correct": (not fixes), "hidden": hidden})
                records.append({"example_id": eid, "budget": 128, "sample_index": s,
                                "correct": True})
    return records


@pytest.fixture
def probe_run(make_probe_run):
    rng = np.random.default_rng(0)
    return make_probe_run(_build_records(rng), hidden_dim=HIDDEN_DIM)


def test_probe_beats_baselines_on_both_targets(probe_run):
    rows = load_runs(probe_run / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(probe_run / "hidden_states")
    results = train_probe(rows, reader, ProbeConfig(layers=[-1]))

    binm = results["targets"]["fixes_incorrect"]
    assert binm["hidden_state_probe"]["test"]["auroc"] > 0.9
    assert binm["hidden_state_beats_input_baseline"] is True
    # Baselines are near chance because the input features carry no signal here.
    assert binm["input_only_baseline"]["test"]["auroc"] < 0.75

    reg = results["targets"]["value_of_compute"]
    assert reg["hidden_state_probe"]["test"]["r2"] > 0.5
    assert reg["hidden_state_beats_input_baseline"] is True


def test_split_sizes_and_no_test_leakage(probe_run):
    rows = load_runs(probe_run / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(probe_run / "hidden_states")
    results = train_probe(rows, reader, ProbeConfig(layers=[-1]))

    tgt = results["targets"]["fixes_incorrect"]
    n = tgt["n_instances"]
    # 20 examples/split * 4 samples, only budget-0 rows become instances.
    assert n["train"] == 80 and n["val"] == 80 and n["test"] == 80
    # Exactly the test instances are scored (one prediction row each).
    assert len(tgt["predictions"]) == 80
    # Every prediction is a test example (selection/fitting never saw these ids).
    assert all(p["example_id"].startswith("test-") for p in tgt["predictions"])


def test_missing_split_is_rejected(make_probe_run):
    # Only test examples: no train/val -> the probe must refuse rather than leak.
    recs = []
    for i in range(4):
        for s in range(2):
            recs.append({"example_id": f"test-{i}", "budget": 0, "sample_index": s,
                         "correct": False, "hidden": np.ones(HIDDEN_DIM)})
            recs.append({"example_id": f"test-{i}", "budget": 128, "sample_index": s,
                         "correct": True})
    run_dir = make_probe_run(recs, hidden_dim=HIDDEN_DIM)
    rows = load_runs(run_dir / "fixed_budget_runs.jsonl")
    reader = HiddenStateReader(run_dir / "hidden_states")
    with pytest.raises(ValueError, match="split"):
        train_probe(rows, reader, ProbeConfig(layers=[-1]))
