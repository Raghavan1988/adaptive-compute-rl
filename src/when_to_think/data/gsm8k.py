"""GSM8K loading with explicit, non-overlapping train/val/test splits.

GSM8K is grade-school math word problems whose gold answer is an exact integer,
so the task reward can be deterministic rule-based exact match — no LLM judge
(AGENTS.md §4.6). This module owns three research-critical guarantees:

- Validation is carved from the TRAIN split only. The held-out test split is never
  touched for tuning probes, policies, thresholds, or hyperparameters (§4.2).
- Splits are provably disjoint, checked by both example id and question content
  every time they are built (§12).
- Any subsampling (max_train_examples / max_test_examples) is deterministic given
  the sampling seed, so every method compared sees the SAME evaluation data (§4.1).

The gold-answer parser here reads the DATASET's reference answer (the text after
`####`). Parsing the MODEL's generated answer is a separate concern handled in the
rewards module — the two must not be conflated.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from when_to_think.config import DataConfig

# GSM8K reference answers end with "#### <final answer>". Capture the final line.
_GSM8K_GOLD_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


@dataclass(frozen=True)
class Example:
    """One benchmark item. The full reference solution is kept, not dropped.

    `answer_text` (the gold chain-of-thought + final answer) is retained for error
    analysis; `gold_answer` is the parsed exact answer used for scoring.
    """

    example_id: str
    question: str
    gold_answer: str
    answer_text: str
    source_split: str
    source_index: int
    dataset_name: str = "gsm8k"


@dataclass
class DatasetSplits:
    """Disjoint train/val/test partitions plus the provenance needed for a run record."""

    train: list[Example]
    val: list[Example]
    test: list[Example]
    dataset_name: str = "gsm8k"
    sampling_seed: int = 0
    val_fraction: float = 0.1

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def parse_gsm8k_gold_answer(answer_text: str) -> str:
    """Extract the canonical exact answer (the text after `####`), stripped of formatting.

    Commas and a leading `$` are removed so "1,000" and "$1000" canonicalize to
    "1000". Returns the cleaned string; equality is decided in the rewards module.
    """
    match = _GSM8K_GOLD_RE.search(answer_text.strip())
    if match is None:
        raise ValueError(f"No '#### <answer>' marker in GSM8K answer: {answer_text!r}")
    cleaned = match.group(1).strip().replace(",", "").replace("$", "").strip()
    if not cleaned:
        raise ValueError(f"Empty gold answer after '####' in: {answer_text!r}")
    return cleaned


def build_example(question: str, answer_text: str, source_split: str, source_index: int) -> Example:
    """Construct an Example, parsing the gold answer. Id is stable per (split, index)."""
    return Example(
        example_id=f"{source_split}-{source_index}",
        question=question,
        gold_answer=parse_gsm8k_gold_answer(answer_text),
        answer_text=answer_text,
        source_split=source_split,
        source_index=source_index,
    )


def _normalize_question(question: str) -> str:
    """Whitespace/case-normalized question, used only for content-overlap detection."""
    return " ".join(question.split()).lower()


def assert_disjoint_splits(splits: DatasetSplits) -> None:
    """Fail loudly if any two splits share an example id or an identical question.

    Checking by content (not just id) guards against dataset-level duplicates that
    could leak test questions into training even when ids differ.
    """
    named = {"train": splits.train, "val": splits.val, "test": splits.test}

    # Ids must be globally unique (no duplicate rows within or across splits).
    all_ids = [e.example_id for split in named.values() for e in split]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate example_id across splits — splits are not disjoint")

    names = list(named)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            q_a = {_normalize_question(e.question) for e in named[a]}
            q_b = {_normalize_question(e.question) for e in named[b]}
            overlap = q_a & q_b
            if overlap:
                raise ValueError(
                    f"{len(overlap)} identical question(s) shared between "
                    f"'{a}' and '{b}' splits — possible train/test leakage"
                )


def make_splits(
    train_examples: list[Example],
    test_examples: list[Example],
    *,
    val_fraction: float,
    sampling_seed: int,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
    dataset_name: str = "gsm8k",
) -> DatasetSplits:
    """Carve val from train and optionally subsample, deterministically.

    Determinism matters twice over: the val carve must not depend on run order, and
    a capped test set must be identical across every method being compared (§4.1).
    Both use `sampling_seed`; the test stream is offset so it is independent of the
    val carve.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(sampling_seed)
    train_idx = list(range(len(train_examples)))
    rng.shuffle(train_idx)

    n_val = round(val_fraction * len(train_examples))
    val_idx = train_idx[:n_val]
    pool_idx = train_idx[n_val:]
    if max_train_examples is not None:
        pool_idx = pool_idx[:max_train_examples]

    # Present each split in stable (source-index) order for reproducible iteration,
    # even though membership was chosen randomly.
    train_split = [train_examples[i] for i in sorted(pool_idx)]
    val_split = [train_examples[i] for i in sorted(val_idx)]

    if max_test_examples is not None and max_test_examples < len(test_examples):
        test_rng = random.Random(sampling_seed + 1)  # independent of the val carve
        test_idx = list(range(len(test_examples)))
        test_rng.shuffle(test_idx)
        test_split = [test_examples[i] for i in sorted(test_idx[:max_test_examples])]
    else:
        # Uncapped: keep the full test set in its original order (fully deterministic).
        test_split = list(test_examples)

    splits = DatasetSplits(
        train=train_split,
        val=val_split,
        test=test_split,
        dataset_name=dataset_name,
        sampling_seed=sampling_seed,
        val_fraction=val_fraction,
    )
    assert_disjoint_splits(splits)
    return splits


def load_gsm8k(cfg: DataConfig) -> DatasetSplits:
    """Download GSM8K via 🤗 datasets and build disjoint splits from `cfg`.

    Kept thin: the split/parse logic lives in pure functions above so it can be
    tested without a network download.
    """
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset_name, cfg.dataset_config)
    raw_train = raw[cfg.train_split]
    raw_test = raw[cfg.test_split]

    train_examples = [
        build_example(row["question"], row["answer"], cfg.train_split, i)
        for i, row in enumerate(raw_train)
    ]
    test_examples = [
        build_example(row["question"], row["answer"], cfg.test_split, i)
        for i, row in enumerate(raw_test)
    ]

    return make_splits(
        train_examples,
        test_examples,
        val_fraction=cfg.val_fraction,
        sampling_seed=cfg.sampling_seed,
        max_train_examples=cfg.max_train_examples,
        max_test_examples=cfg.max_test_examples,
        dataset_name=cfg.dataset_name,
    )
