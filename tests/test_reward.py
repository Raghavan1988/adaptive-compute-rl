"""Tests for answer extraction and reward (AGENTS.md §12: required M0 tests).

Covers: answer extraction on valid and invalid outputs; correct/incorrect task
reward; and the compute penalty applied exactly once. STOP/CONTINUE transition
semantics are tested with the RL environment in M4, not here.
"""

import pytest

from when_to_think.config import RewardConfig
from when_to_think.rewards import (
    answers_match,
    compute_reward,
    compute_reward_sweep,
    extract_numeric_answer,
)

# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #

def test_extract_prefers_hash_marker():
    assert extract_numeric_answer("lots of reasoning 17\n#### 42") == "42"


def test_extract_boxed():
    assert extract_numeric_answer(r"so the total is \boxed{42}.") == "42"


def test_extract_answer_is_phrase():
    assert extract_numeric_answer("Therefore the answer is 42.") == "42"


def test_extract_falls_back_to_last_number():
    assert extract_numeric_answer("first 3, then 5, finally 42") == "42"


def test_extract_strips_commas():
    assert extract_numeric_answer("#### 1,234") == "1234"


def test_extract_malformed_returns_none():
    assert extract_numeric_answer("no numbers at all here") is None
    assert extract_numeric_answer("") is None
    assert extract_numeric_answer(None) is None


def test_answers_match_numeric_equivalence():
    assert answers_match("42", "42")
    assert answers_match("42.0", "42")
    assert answers_match("1,000", "1000")


def test_answers_match_rejects_wrong_and_none():
    assert not answers_match("7", "8")
    assert not answers_match(None, "5")  # malformed prediction never matches


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #

def _cfg(**kw) -> RewardConfig:
    return RewardConfig(**kw)


def test_correct_gets_task_reward():
    r = compute_reward(correct=True, compute_units=100, lambda_compute=0.0, reward_config=_cfg())
    assert r.reward_task == 1.0


def test_incorrect_gets_zero_task_reward():
    r = compute_reward(correct=False, compute_units=100, lambda_compute=0.0, reward_config=_cfg())
    assert r.reward_task == 0.0


def test_compute_penalty_applied_exactly_once():
    r = compute_reward(correct=True, compute_units=200, lambda_compute=0.001, reward_config=_cfg())
    # Exactly one penalty: total == task - lambda * C, components stay separate.
    assert r.reward_compute == pytest.approx(-0.2)
    assert r.reward_total == pytest.approx(1.0 - 0.2)
    assert r.reward_task == 1.0


def test_zero_lambda_means_no_penalty():
    r = compute_reward(correct=True, compute_units=999, lambda_compute=0.0, reward_config=_cfg())
    assert r.reward_compute == 0.0
    assert r.reward_total == r.reward_task


def test_compute_proxy_named_not_flops():
    r = compute_reward(correct=True, compute_units=1, lambda_compute=0.0, reward_config=_cfg())
    assert r.compute_proxy == "reasoning_tokens"


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        compute_reward(correct=True, compute_units=-1, lambda_compute=0.0, reward_config=_cfg())
    with pytest.raises(ValueError):
        compute_reward(correct=True, compute_units=1, lambda_compute=-0.1, reward_config=_cfg())


def test_reward_sweep_covers_all_lambdas():
    cfg = _cfg(lambda_compute_sweep=[0.0, 0.001, 0.01])
    sweep = compute_reward_sweep(correct=True, compute_units=100, reward_config=cfg)
    assert [r.lambda_compute for r in sweep] == [0.0, 0.001, 0.01]
    # Higher lambda => lower total reward at fixed compute.
    assert sweep[0].reward_total > sweep[1].reward_total > sweep[2].reward_total
