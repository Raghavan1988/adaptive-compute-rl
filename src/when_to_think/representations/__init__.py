"""Selective hidden-state extraction (layer, position, step, pooling)."""

from when_to_think.representations.extraction import (
    RepresentationDescriptor,
    ShardedRepresentationWriter,
    extract_hidden_states,
    pool_hidden,
)

__all__ = [
    "RepresentationDescriptor",
    "ShardedRepresentationWriter",
    "extract_hidden_states",
    "pool_hidden",
]
