"""Tests for global seeding (AGENTS.md §9)."""

import random

import numpy as np
import torch

from when_to_think.utils import make_generator, seed_everything


def test_seed_everything_makes_python_numpy_torch_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    seed_everything(123)
    b = (random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist())
    assert a == b


def test_different_seed_gives_different_draw():
    seed_everything(0)
    a = torch.rand(5).tolist()
    seed_everything(1)
    b = torch.rand(5).tolist()
    assert a != b


def test_make_generator_is_reproducible():
    g1 = make_generator(7)
    g2 = make_generator(7)
    x1 = torch.rand(4, generator=g1)
    x2 = torch.rand(4, generator=g2)
    assert torch.equal(x1, x2)
