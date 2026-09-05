"""The STOP/CONTINUE policy: a small network over the decision-point hidden state (M4).

The policy is deliberately separable from the frozen base SLM (AGENTS.md §14): it sees
only the stored hidden state at the current checkpoint, optionally concatenated with a
normalized 'progress' feature (cumulative tokens / max budget) so it can condition on
how much budget it has already spent — not just the representation. Linear by default
(``hidden_sizes=[]``); a small MLP only if configured.

Feature standardization statistics are fit on TRAIN observations only (§4.2) — reusing
the probe's scaler so the discipline is identical across M3 and M4.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from when_to_think.config import PolicyConfig
from when_to_think.policies.data import Checkpoint, Trajectory
from when_to_think.probes.models import StandardScaler


def build_features(
    cp: Checkpoint,
    traj: Trajectory,
    *,
    layer: int,
    include_progress: bool,
    max_budget: int,
) -> np.ndarray:
    """Policy input for one checkpoint: the hidden state (+ optional progress scalar)."""
    hidden = np.asarray(cp.hidden[layer], dtype=np.float64).ravel()
    if not include_progress:
        return hidden
    progress = cp.cumulative_reasoning_tokens / max_budget if max_budget > 0 else 0.0
    return np.concatenate([hidden, [progress]])


class StopContinuePolicy(nn.Module):
    """Maps a feature vector to two logits: [STOP, CONTINUE]."""

    def __init__(self, input_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyModel:
    """A trained policy: the network, its input standardizer, and the feature spec.

    Bundles everything needed to turn a checkpoint into an action so the environment
    loop, evaluation, and serialization all agree on the feature contract.
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        max_budget: int,
        scaler: StandardScaler,
        net: StopContinuePolicy,
    ) -> None:
        self.cfg = cfg
        self.max_budget = max_budget
        self.scaler = scaler
        self.net = net

    def features(self, cp: Checkpoint, traj: Trajectory) -> np.ndarray:
        return build_features(
            cp, traj,
            layer=self.cfg.layer,
            include_progress=self.cfg.include_progress_feature,
            max_budget=self.max_budget,
        )

    def _tensor(self, feats: np.ndarray) -> torch.Tensor:
        x = np.atleast_2d(feats)
        if self.cfg.standardize:
            x = self.scaler.transform(x)
        return torch.as_tensor(x, dtype=torch.float32)

    def logits(self, feats: np.ndarray) -> torch.Tensor:
        return self.net(self._tensor(feats))

    @torch.no_grad()
    def act_greedy(self, cp: Checkpoint, traj: Trajectory) -> int:
        """Deterministic argmax action (used at evaluation)."""
        logits = self.logits(self.features(cp, traj))
        return int(torch.argmax(logits, dim=-1).item())
