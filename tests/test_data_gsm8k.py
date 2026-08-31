"""Tests for GSM8K loading, split integrity, and gold-answer parsing (AGENTS.md §12).

Split integrity is a required M0 test: a val set carved from train, provable
disjointness, and deterministic subsampling. These run offline on synthetic
examples — no dataset download.
"""

import pytest

from when_to_think.data import (
    DatasetSplits,
    Example,
    assert_disjoint_splits,
    build_example,
    make_splits,
    parse_gsm8k_gold_answer,
)

# --------------------------------------------------------------------------- #
# Gold-answer parsing
# --------------------------------------------------------------------------- #

def test_parse_gold_basic():
    assert parse_gsm8k_gold_answer("Reason reason.\n#### 42") == "42"


def test_parse_gold_strips_commas_and_dollar():
    assert parse_gsm8k_gold_answer("...\n#### 1,000") == "1000"
    assert parse_gsm8k_gold_answer("...\n#### $2,500") == "2500"


def test_parse_gold_missing_marker_raises():
    with pytest.raises(ValueError, match="No '#### <answer>' marker"):
        parse_gsm8k_gold_answer("There is no marker here, just 42.")


def test_build_example_id_and_gold():
    ex = build_example("What is 2+2?", "Add them.\n#### 4", "train", 7)
    assert ex.example_id == "train-7"
    assert ex.gold_answer == "4"
    assert ex.answer_text == "Add them.\n#### 4"  # full solution retained, not dropped


# --------------------------------------------------------------------------- #
# Split construction
# --------------------------------------------------------------------------- #

def _synthetic(source_split: str, n: int, start: int = 0) -> list[Example]:
    return [
        build_example(f"Question {source_split} {i}?", f"soln\n#### {i}", source_split, i)
        for i in range(start, start + n)
    ]


def test_val_is_carved_from_train_only():
    train = _synthetic("train", 100)
    test = _synthetic("test", 20)
    splits = make_splits(train, test, val_fraction=0.1, sampling_seed=0)
    # Val must come from train provenance, never from the test split.
    assert all(e.source_split == "train" for e in splits.val)
    assert len(splits.val) == 10
    assert len(splits.train) == 90
    assert len(splits.test) == 20


def test_splits_are_disjoint():
    train = _synthetic("train", 50)
    test = _synthetic("test", 10)
    splits = make_splits(train, test, val_fraction=0.2, sampling_seed=1)
    train_ids = {e.example_id for e in splits.train}
    val_ids = {e.example_id for e in splits.val}
    test_ids = {e.example_id for e in splits.test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # No example is lost: train + val partition the original train set.
    assert len(train_ids) + len(val_ids) == 50


def test_subsampling_is_deterministic_for_same_seed():
    train = _synthetic("train", 200)
    test = _synthetic("test", 100)
    a = make_splits(train, test, val_fraction=0.1, sampling_seed=3,
                    max_train_examples=20, max_test_examples=15)
    b = make_splits(train, test, val_fraction=0.1, sampling_seed=3,
                    max_train_examples=20, max_test_examples=15)
    assert [e.example_id for e in a.train] == [e.example_id for e in b.train]
    assert [e.example_id for e in a.test] == [e.example_id for e in b.test]
    assert len(a.train) == 20
    assert len(a.test) == 15


def test_different_seed_changes_val_membership():
    train = _synthetic("train", 200)
    test = _synthetic("test", 20)
    a = make_splits(train, test, val_fraction=0.1, sampling_seed=0)
    b = make_splits(train, test, val_fraction=0.1, sampling_seed=999)
    assert {e.example_id for e in a.val} != {e.example_id for e in b.val}


def test_uncapped_test_preserves_full_set_in_order():
    train = _synthetic("train", 30)
    test = _synthetic("test", 12)
    splits = make_splits(train, test, val_fraction=0.1, sampling_seed=0)
    assert [e.example_id for e in splits.test] == [f"test-{i}" for i in range(12)]


# --------------------------------------------------------------------------- #
# Integrity check catches leakage
# --------------------------------------------------------------------------- #

def test_assert_disjoint_detects_duplicate_ids():
    dup = build_example("Q?", "a\n#### 1", "train", 0)
    splits = DatasetSplits(train=[dup], val=[dup], test=[])
    with pytest.raises(ValueError, match="Duplicate example_id"):
        assert_disjoint_splits(splits)


def test_assert_disjoint_detects_content_leak_across_splits():
    # Same question text, different ids: a content-level train/test leak.
    train_ex = build_example("Identical question?", "a\n#### 1", "train", 0)
    test_ex = build_example("Identical question?", "a\n#### 1", "test", 0)
    splits = DatasetSplits(train=[train_ex], val=[], test=[test_ex])
    with pytest.raises(ValueError, match="possible train/test leakage"):
        assert_disjoint_splits(splits)
