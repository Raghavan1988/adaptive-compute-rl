"""Dataset loading and split management (no train/test overlap)."""

from when_to_think.data.gsm8k import (
    DatasetSplits,
    Example,
    assert_disjoint_splits,
    build_example,
    load_gsm8k,
    make_splits,
    parse_gsm8k_gold_answer,
)

__all__ = [
    "DatasetSplits",
    "Example",
    "assert_disjoint_splits",
    "build_example",
    "load_gsm8k",
    "make_splits",
    "parse_gsm8k_gold_answer",
]
