"""Global seeding for reproducibility (AGENTS.md §9).

Seeds Python, NumPy, and PyTorch (CPU + CUDA) from one seed. Data sampling is
seeded separately via `DataConfig.sampling_seed` so that changing a training seed
never silently reshuffles the evaluation set (AGENTS.md §4.1).

Remaining nondeterminism to document per run:
- Sampling-based generation is only reproducible if a seeded `torch.Generator` is
  passed to `model.generate` (see `make_generator`). The global seed alone does not
  pin multinomial sampling across all backends.
- Some CUDA kernels are nondeterministic unless `deterministic=True`, which can be
  slower and will raise (here: warn) if an op lacks a deterministic implementation.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch. Optionally force deterministic CUDA algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required for deterministic cuBLAS GEMMs; set before CUDA context use.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """A seeded torch.Generator for reproducible sampling in `model.generate`."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
