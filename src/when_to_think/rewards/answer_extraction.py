"""Extract and compare final numeric answers from model output (AGENTS.md §12, §17).

Extraction is layered and conservative to limit reward hacking: the model earns
credit for a clearly-marked final answer first, and only falls back to "last number
in the text" when no marker is present. A malformed output returns None, which is
PRESERVED as a failure downstream, never silently dropped (AGENTS.md §4.4).

This parses the MODEL's output. The DATASET's gold answer is parsed separately in
`when_to_think.data.gsm8k` — the two are intentionally not shared, because model
output is noisy free text while gold answers are structured.
"""

from __future__ import annotations

import re
from fractions import Fraction

# A signed integer or decimal, allowing thousands separators (e.g. -1,234.5).
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_MARKER_RE = re.compile(rf"####\s*({_NUMBER})")
_BOXED_RE = re.compile(rf"\\boxed\{{\s*({_NUMBER})\s*\}}")
_ANSWER_IS_RE = re.compile(rf"answer\s*(?:is|:)?\s*\$?({_NUMBER})", re.IGNORECASE)


def _clean_number(text: str) -> str:
    """Canonicalize a numeric string: drop commas/$ and a trailing period."""
    return text.strip().replace(",", "").replace("$", "").rstrip(".").strip()


def extract_numeric_answer(text: str | None) -> str | None:
    """Return the model's final numeric answer as a canonical string, or None.

    Precedence: explicit `#### x` marker, then `\\boxed{x}`, then "answer is x",
    then the last number anywhere in the text. The fallback mirrors the standard
    GSM8K convention but is a known reward-hacking surface, so markers win first.
    """
    if text is None:
        return None
    for pattern in (_MARKER_RE, _BOXED_RE, _ANSWER_IS_RE):
        match = pattern.search(text)
        if match:
            return _clean_number(match.group(1))
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _clean_number(numbers[-1])
    return None


def _to_number(text: str) -> Fraction | None:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(prediction: str | None, gold: str) -> bool:
    """Exact-match comparison, numeric when possible (so 42 == 42.0 == 1,000→1000).

    Falls back to canonical string equality only when a value is not parseable as a
    number. A None prediction (malformed output) never matches.
    """
    if prediction is None:
        return False
    pred_clean = _clean_number(prediction)
    gold_clean = _clean_number(gold)
    pred_num = _to_number(pred_clean)
    gold_num = _to_number(gold_clean)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return pred_clean == gold_clean
