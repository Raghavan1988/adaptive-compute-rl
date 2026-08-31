"""Tests for experiment config loading (AGENTS.md §12: research-critical logic is tested).

Config is research-critical: a silently-mishandled key can change an experiment's
budgets, splits, or reward without anyone noticing. These tests lock in that
unknown keys are rejected, overrides are typed correctly, refs compose, and the
invariants in `validate_config` actually fire.
"""

from pathlib import Path

import pytest
import yaml

from when_to_think.config import (
    ExperimentConfig,
    apply_overrides,
    load_config,
    load_config_from_args,
    save_config,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ROOT = REPO_ROOT / "configs"
SMOKE_CONFIG = CONFIGS_ROOT / "experiment" / "gsm8k_smoke.yaml"


def _write(dir_path: Path, name: str, data: dict) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _make_config_tree(tmp_path: Path) -> Path:
    """Build a minimal configs/{model,data,experiment} tree; return the exp file."""
    _write(tmp_path / "model", "tiny", {"name": "hf/tiny-model"})
    _write(
        tmp_path / "data",
        "toy",
        {"dataset_name": "toy", "train_split": "train", "test_split": "test"},
    )
    exp = _write(
        tmp_path / "experiment",
        "exp",
        {"name": "exp", "model": "tiny", "data": "toy"},
    )
    return exp


# --------------------------------------------------------------------------- #
# Real shipped config
# --------------------------------------------------------------------------- #

def test_smoke_config_loads_and_validates():
    cfg = load_config(SMOKE_CONFIG)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "gsm8k_smoke"
    # Ref resolution pulled in the model/data sub-configs.
    assert cfg.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.data.dataset_name == "openai/gsm8k"
    # lambda is a sweep, not a scalar.
    assert len(cfg.reward.lambda_compute_sweep) >= 1


# --------------------------------------------------------------------------- #
# Composition and defaults
# --------------------------------------------------------------------------- #

def test_string_refs_resolve(tmp_path):
    exp = _make_config_tree(tmp_path)
    cfg = load_config(exp, configs_root=tmp_path)
    assert cfg.model.name == "hf/tiny-model"
    assert cfg.data.dataset_name == "toy"


def test_tokenizer_name_defaults_to_model_name(tmp_path):
    exp = _make_config_tree(tmp_path)
    cfg = load_config(exp, configs_root=tmp_path)
    assert cfg.model.tokenizer_name == cfg.model.name


def test_missing_ref_file_raises(tmp_path):
    _write(tmp_path / "data", "toy", {"dataset_name": "toy"})
    exp = _write(
        tmp_path / "experiment",
        "exp",
        {"name": "exp", "model": "does-not-exist", "data": "toy"},
    )
    with pytest.raises(FileNotFoundError):
        load_config(exp, configs_root=tmp_path)


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #

def test_override_is_typed_via_yaml(tmp_path):
    exp = _make_config_tree(tmp_path)
    cfg = load_config(
        exp,
        overrides=["generation.max_reasoning_budget=128", "generation.fixed_budgets=[0, 64, 128]"],
        configs_root=tmp_path,
    )
    assert cfg.generation.max_reasoning_budget == 128  # int, not "128"
    assert cfg.generation.fixed_budgets == [0, 64, 128]  # list, not a string


def test_override_can_set_nested_model_field(tmp_path):
    exp = _make_config_tree(tmp_path)
    cfg = load_config(exp, overrides=["model.dtype=float16"], configs_root=tmp_path)
    assert cfg.model.dtype == "float16"


def test_malformed_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


# --------------------------------------------------------------------------- #
# Unknown keys rejected (typos fail loudly)
# --------------------------------------------------------------------------- #

def test_unknown_subconfig_key_rejected(tmp_path):
    _write(tmp_path / "model", "tiny", {"name": "hf/tiny-model", "dtpye": "bfloat16"})  # typo
    _write(tmp_path / "data", "toy", {"dataset_name": "toy"})
    exp = _write(tmp_path / "experiment", "exp", {"name": "exp", "model": "tiny", "data": "toy"})
    with pytest.raises(ValueError, match="Unknown ModelConfig keys"):
        load_config(exp, configs_root=tmp_path)


def test_unknown_top_level_key_rejected(tmp_path):
    _write(tmp_path / "model", "tiny", {"name": "hf/tiny-model"})
    _write(tmp_path / "data", "toy", {"dataset_name": "toy"})
    exp = _write(
        tmp_path / "experiment",
        "exp",
        {"name": "exp", "model": "tiny", "data": "toy", "lr": 0.1},  # lr is not top-level
    )
    with pytest.raises(ValueError, match="Unknown top-level config keys"):
        load_config(exp, configs_root=tmp_path)


# --------------------------------------------------------------------------- #
# Validation of research invariants
# --------------------------------------------------------------------------- #

def _minimal_cfg() -> ExperimentConfig:
    return load_config(SMOKE_CONFIG)


def test_validate_rejects_test_equals_train_split():
    cfg = _minimal_cfg()
    cfg.data.train_split = "test"
    cfg.data.test_split = "test"
    with pytest.raises(ValueError, match="must differ"):
        validate_config(cfg)


def test_validate_rejects_empty_lambda_sweep():
    cfg = _minimal_cfg()
    cfg.reward.lambda_compute_sweep = []
    with pytest.raises(ValueError, match="lambda is never universal"):
        validate_config(cfg)


def test_validate_rejects_budget_over_max():
    cfg = _minimal_cfg()
    cfg.generation.max_reasoning_budget = 100
    cfg.generation.fixed_budgets = [0, 128]
    with pytest.raises(ValueError, match="exceed max_reasoning_budget"):
        validate_config(cfg)


def test_validate_rejects_bad_val_fraction():
    cfg = _minimal_cfg()
    cfg.data.val_fraction = 1.0
    with pytest.raises(ValueError, match="val_fraction"):
        validate_config(cfg)


# --------------------------------------------------------------------------- #
# Round-trip and CLI helper
# --------------------------------------------------------------------------- #

def test_save_and_reload_roundtrip(tmp_path):
    cfg = _minimal_cfg()
    out = tmp_path / "snapshot.yaml"
    save_config(cfg, out)
    reloaded = load_config(out, configs_root=CONFIGS_ROOT)
    assert reloaded.model.name == cfg.model.name
    assert reloaded.generation.fixed_budgets == cfg.generation.fixed_budgets
    assert reloaded.reward.lambda_compute_sweep == cfg.reward.lambda_compute_sweep


def test_load_config_from_args(tmp_path):
    import argparse

    from when_to_think.config import add_config_args

    exp = _make_config_tree(tmp_path)
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(["--config", str(exp), "--set", "seed=7"])
    # configs_root defaults to grandparent of the config path, which is tmp_path here.
    cfg = load_config_from_args(args)
    assert cfg.seed == 7
