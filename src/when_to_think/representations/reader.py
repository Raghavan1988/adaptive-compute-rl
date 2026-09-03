"""Read back sharded hidden states written by ``ShardedRepresentationWriter``.

The writer streams (example_id, reasoning_step, per-layer vectors) plus arbitrary
metadata (budget, sample_index) to compressed ``.npz`` shards and a ``manifest.jsonl``.
This reader is the inverse: it loads the manifest and shards and exposes each stored
vector keyed by its provenance so a downstream probe can *join* a hidden state to the
outcome row that produced it (same example_id, budget, sample_index).

Kept separate from the writer so M3/M4 can consume representations without importing
generation machinery. Shards are loaded lazily and cached, so a large sweep is read a
shard at a time rather than all at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HiddenStateKey:
    """Provenance that uniquely identifies one stored vector within a run."""

    example_id: str
    budget: int | None
    sample_index: int | None
    reasoning_step: int


class HiddenStateReader:
    """Random-access reader over a run's ``hidden_states/`` directory.

    ``descriptor`` carries the extraction provenance (layers, token position, pooling,
    model revision). ``get(example_id, budget, sample_index, layer)`` returns the stored
    float32 vector; shards are memoized so repeated lookups touch disk once per shard.
    """

    def __init__(self, hidden_dir: str | Path) -> None:
        self.hidden_dir = Path(hidden_dir)
        manifest_path = self.hidden_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")
        self.manifest: list[dict] = [
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        descriptor_path = self.hidden_dir / "descriptor.json"
        self.descriptor: dict = (
            json.loads(descriptor_path.read_text()) if descriptor_path.exists() else {}
        )
        # (example_id, budget, sample_index) -> manifest row. reasoning_step is kept in
        # the row; the M1 sweep writes a single decision point (step 0) per key.
        self._index: dict[tuple[str, int | None, int | None], dict] = {}
        for row in self.manifest:
            key = (row["example_id"], row.get("budget"), row.get("sample_index"))
            self._index[key] = row
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}

    @property
    def layers(self) -> list[int]:
        return list(self.descriptor.get("layers", []))

    def _shard(self, shard_name: str) -> dict[str, np.ndarray]:
        cached = self._shard_cache.get(shard_name)
        if cached is None:
            with np.load(self.hidden_dir / shard_name) as npz:
                cached = {name: npz[name] for name in npz.files}
            self._shard_cache[shard_name] = cached
        return cached

    def get(
        self,
        example_id: str,
        *,
        budget: int | None = None,
        sample_index: int | None = None,
        layer: int = -1,
    ) -> np.ndarray | None:
        """Return the stored vector for one (example, budget, sample, layer), or None."""
        row = self._index.get((example_id, budget, sample_index))
        if row is None:
            return None
        arrays = self._shard(row["shard"])
        key = f"layer_{layer}"
        if key not in arrays:
            raise KeyError(
                f"layer {layer} not stored (available: {sorted(arrays)}); "
                f"descriptor layers={self.layers}"
            )
        return arrays[key][row["row"]]

    def __len__(self) -> int:
        return len(self.manifest)
