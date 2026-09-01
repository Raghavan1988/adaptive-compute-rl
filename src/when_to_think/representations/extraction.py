"""Selective hidden-state extraction with full provenance (AGENTS.md §13).

Every stored vector records enough to reconstruct exactly what produced it: layer
index, token position, pooling, reasoning step, and model revision. Extraction is
SELECTIVE (only requested layers) to avoid gigantic dumps, and storage is sharded
so large sweeps stream to disk instead of living in memory.

Extraction is padding-agnostic: the "last token" is gathered via the attention
mask, so left- or right-padded batches both pick the true final token. (We
left-pad by default in the model loader, but relying on the mask is safer.)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RepresentationDescriptor:
    """Provenance for a set of extracted vectors (written alongside the shards)."""

    layers: list[int]
    token_position: str
    pooling: str | None
    model_name: str
    model_revision: str | None


def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real (non-pad) token in each row, for either padding side.

    Found as the last position where the mask is 1 (not `sum-1`, which is only
    correct for right padding — we left-pad by default, where real tokens sit at
    the end). `argmax` on the reversed mask returns the first 1 from the right.
    """
    seq_len = attention_mask.shape[1]
    from_right = torch.argmax(attention_mask.long().flip(dims=[1]), dim=1)
    return seq_len - 1 - from_right


def pool_hidden(
    layer_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    token_position: str,
    pooling: str | None,
) -> torch.Tensor:
    """Reduce a (batch, seq, hidden) layer tensor to (batch, hidden)."""
    if token_position == "last":
        if pooling not in (None, "none"):
            raise ValueError("token_position='last' takes no pooling")
        idx = _last_token_indices(attention_mask)
        return layer_tensor[torch.arange(layer_tensor.size(0), device=layer_tensor.device), idx]
    if token_position == "all":
        if pooling != "mean":
            raise ValueError("token_position='all' requires pooling='mean'")
        mask = attention_mask.unsqueeze(-1).to(layer_tensor.dtype)
        summed = (layer_tensor * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)  # avoid divide-by-zero on empty rows
        return summed / counts
    raise ValueError(f"Unknown token_position {token_position!r}")


def extract_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    spec: RepresentationDescriptor,
) -> dict[int, np.ndarray]:
    """Select + pool the requested layers into {layer: (batch, hidden)} float32 arrays.

    `hidden_states` is the tuple from a forward pass with output_hidden_states=True
    (length num_layers + 1; index 0 is the embedding output). Negative layer indices
    count from the last layer.
    """
    num = len(hidden_states)
    result: dict[int, np.ndarray] = {}
    for layer in spec.layers:
        if not -num <= layer < num:
            raise IndexError(f"layer {layer} out of range for {num} hidden states")
        pooled = pool_hidden(
            hidden_states[layer], attention_mask, spec.token_position, spec.pooling
        )
        # Store float32 on CPU: hidden states may be bf16, which numpy cannot hold.
        result[layer] = pooled.detach().to(torch.float32).cpu().numpy()
    return result


class ShardedRepresentationWriter:
    """Stream (example_id, reasoning_step, per-layer vectors) to sharded .npz files.

    Writes one compressed shard every `shard_size` records plus a `manifest.jsonl`
    mapping each record to its (shard, row), and a `descriptor.json` with provenance.
    Reading back: load the manifest, then index `shard[f"layer_{L}"][row]`.
    """

    def __init__(
        self,
        out_dir: str | Path,
        descriptor: RepresentationDescriptor,
        shard_size: int = 512,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.descriptor = descriptor
        self.shard_size = shard_size
        self._buffer: list[tuple[str, int, dict[int, np.ndarray]]] = []
        self._shard_idx = 0
        self._manifest: list[dict] = []
        (self.out_dir / "descriptor.json").write_text(json.dumps(asdict(descriptor), indent=2))

    def add(
        self, example_id: str, reasoning_step: int, layer_vectors: dict[int, np.ndarray]
    ) -> None:
        self._buffer.append((example_id, reasoning_step, layer_vectors))
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        arrays = {
            f"layer_{layer}": np.stack([vecs[layer] for (_, _, vecs) in self._buffer])
            for layer in self.descriptor.layers
        }
        shard_name = f"shard_{self._shard_idx:05d}.npz"
        np.savez_compressed(self.out_dir / shard_name, **arrays)
        for row, (example_id, step, _) in enumerate(self._buffer):
            self._manifest.append(
                {"example_id": example_id, "reasoning_step": step, "shard": shard_name, "row": row}
            )
        self._shard_idx += 1
        self._buffer = []

    def close(self) -> None:
        """Flush the final shard and write the manifest."""
        self._flush()
        with open(self.out_dir / "manifest.jsonl", "w") as handle:
            for row in self._manifest:
                handle.write(json.dumps(row) + "\n")

    def __enter__(self) -> ShardedRepresentationWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
