"""Tests for run records (AGENTS.md §9)."""

import json
import re
from pathlib import Path

from when_to_think.config import load_config
from when_to_think.utils import (
    create_run_record,
    generate_run_id,
    get_git_commit,
    write_run_record,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_CONFIG = REPO_ROOT / "configs" / "experiment" / "gsm8k_smoke.yaml"


def test_generate_run_id_contains_name_and_is_unique():
    a = generate_run_id("exp")
    b = generate_run_id("exp")
    assert a.startswith("exp-")
    assert a != b


def test_get_git_commit_in_repo():
    commit = get_git_commit(Path(__file__).resolve().parent.parent)
    # This repo is under git; commit should be a 40-char hex sha.
    assert commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_create_run_record_captures_config_and_repro_fields():
    cfg = load_config(SMOKE_CONFIG)
    record = create_run_record(cfg, repo_dir=Path(__file__).resolve().parent.parent)
    # §9 fields sourced from the config snapshot.
    assert record.config["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert record.config["data"]["dataset_name"] == "openai/gsm8k"
    assert record.config["reward"]["lambda_compute_sweep"]
    assert record.config["seed"] == 0
    # Runtime repro fields.
    assert record.run_id.startswith("gsm8k_smoke-")
    assert record.git_commit is not None
    assert record.timestamp


def test_write_run_record_roundtrip(tmp_path):
    cfg = load_config(SMOKE_CONFIG)
    record = create_run_record(cfg, run_id="fixed-id", runtime={"model_revision": "abc123"})
    out_dir = write_run_record(record, tmp_path)
    assert out_dir == tmp_path / "fixed-id"
    loaded = json.loads((out_dir / "run_record.json").read_text())
    assert loaded["run_id"] == "fixed-id"
    assert loaded["runtime"]["model_revision"] == "abc123"
    assert loaded["config"]["name"] == "gsm8k_smoke"
