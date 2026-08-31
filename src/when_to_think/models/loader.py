"""Load the base SLM + tokenizer from a ModelConfig (AGENTS.md §14, §21).

The SLM is frozen by default: the initial experiments study a *fixed* model's
representations, so training gradients must never flow into it. `frozen=True`
(the default) sets `requires_grad=False` on every parameter and puts the model in
eval mode. Only flip it when an experiment explicitly studies fine-tuning.

The loader is thin and configurable — model name, revision, dtype, and device all
come from config, nothing hard-coded — so swapping SLMs or moving to a bigger GPU
is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from when_to_think.config import ModelConfig

# String dtype names -> torch dtypes. "auto" defers to the checkpoint's own dtype.
_DTYPE_MAP: dict[str, object] = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedModel:
    """A frozen (by default) SLM plus its tokenizer and the metadata needed for a run record."""

    model: torch.nn.Module
    tokenizer: object
    model_name: str
    revision: str | None
    resolved_dtype: str
    device: str
    frozen: bool


def resolve_dtype(name: str):
    """Map a config dtype string to a torch dtype (or the sentinel 'auto')."""
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def resolve_device(device: str) -> str:
    """Resolve 'auto' to a concrete device, else pass the requested device through.

    'auto' picks CUDA when available and falls back to CPU, so the same config runs
    on the 4090 or on a CPU-only box without editing source.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze every parameter and switch to eval mode (AGENTS.md §14).

    Freezing is a research invariant for the initial experiments: the policy/probe
    must learn from the SLM's representations, not by silently changing them.
    """
    model.requires_grad_(False)
    model.eval()


def load_model_and_tokenizer(cfg: ModelConfig) -> LoadedModel:
    """Load and (by default) freeze the base SLM and its tokenizer from config."""
    torch_dtype = resolve_dtype(cfg.dtype)
    device = resolve_device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
    )
    # Decoder-only tokenizers often lack a pad token. Reuse EOS so batched
    # generation has something to pad with; left-padding is required so that
    # generation continues from the true final token of each (variable-length)
    # prompt rather than from padding (AGENTS.md §22: no silent gen-semantics bugs).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        revision=cfg.revision,
        dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.to(device)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.frozen:
        freeze_model(model)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_name=cfg.name,
        # Prefer the exact resolved commit hash for the run record when transformers
        # exposes it; fall back to the requested revision (which may be None/a branch).
        revision=getattr(model.config, "_commit_hash", None) or cfg.revision,
        resolved_dtype=str(getattr(model, "dtype", torch_dtype)),
        device=device,
        frozen=cfg.frozen,
    )
