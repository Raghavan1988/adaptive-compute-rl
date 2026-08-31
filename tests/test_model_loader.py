"""Tests for the model loader helpers (AGENTS.md §12, §14).

These test the pure logic — dtype/device resolution and the frozen-SLM invariant —
without downloading a model, so they run fast and offline. An end-to-end load of a
real checkpoint is exercised separately (a manual smoke), not in the default suite.
"""

import pytest
import torch

from when_to_think.models import freeze_model, resolve_device, resolve_dtype


def test_resolve_dtype_known():
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("float32") is torch.float32
    assert resolve_dtype("auto") == "auto"


def test_resolve_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("int8")


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cuda:1") == "cuda:1"
    assert resolve_device("cpu") == "cpu"


def test_freeze_model_disables_grad_and_eval():
    # A tiny stand-in model: freezing must not depend on it being a real SLM.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    model.train()
    assert any(p.requires_grad for p in model.parameters())

    freeze_model(model)

    assert all(not p.requires_grad for p in model.parameters())
    assert model.training is False
