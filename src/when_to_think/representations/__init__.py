"""Selective hidden-state extraction (layer, position, step, pooling)."""

from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
    extract_hidden_states,
    pool_hidden,
)
from when_to_think.representations.reader import HiddenStateKey, HiddenStateReader

__all__ = [
    "HiddenStateKey",
    "HiddenStateReader",
    "RepresentationDescriptor",
    "ShardedRepresentationWriter",
    "extract_hidden_states",
    "pool_hidden",
]
