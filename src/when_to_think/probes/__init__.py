"""Supervised probes on hidden states (decodability, not causality).

M3: can a frozen hidden state predict the *value of continuing to reason* better than
simple input-only / prior baselines? See ``train.py`` for the split-disciplined
pipeline and ``dataset.py`` for the two explicit targets.
"""

from when_to_think.probes.baselines import input_difficulty_features
from when_to_think.probes.dataset import (
    ProbeDataset,
    build_probe_dataset,
    split_of,
)
from when_to_think.probes.models import LogisticProbe, RidgeProbe, StandardScaler
from when_to_think.probes.train import TARGETS, train_probe, train_probe_for_target

__all__ = [
    "TARGETS",
    "LogisticProbe",
    "ProbeDataset",
    "RidgeProbe",
    "StandardScaler",
    "build_probe_dataset",
    "input_difficulty_features",
    "split_of",
    "train_probe",
    "train_probe_for_target",
]
