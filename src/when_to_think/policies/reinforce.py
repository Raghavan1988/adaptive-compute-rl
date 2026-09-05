"""REINFORCE training for the STOP/CONTINUE policy (M4).

Episodic policy gradient on the offline trajectory environment. Reward arrives only at
the terminal STOP, so with no discounting every decision in an episode shares the return
G = R_task − λ·C; the update is the classic REINFORCE step with a batch-mean baseline
for variance reduction and an entropy bonus to resist the STOP/CONTINUE collapse that
AGENTS.md §16 warns about (rising reward is not proof of learning).

The policy is trained on TRAIN trajectories only; the feature standardizer is fit on
TRAIN observations only (§4.2). One policy is trained per ``lambda`` — the caller sweeps
the penalty to trace the frontier (§7). Sampling makes training nondeterministic beyond
the seed; that is documented, not hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from when_to_think.config import PolicyConfig, RewardConfig
from when_to_think.policies.data import Trajectory
from when_to_think.policies.env import StopContinueEnv
from when_to_think.policies.policy import PolicyModel, StopContinuePolicy, build_features
from when_to_think.probes.models import StandardScaler


def _all_features(trajs: list[Trajectory], cfg: PolicyConfig, max_budget: int) -> np.ndarray:
    """Every checkpoint's feature vector — used to fit the standardizer on train."""
    feats = [
        build_features(cp, traj, layer=cfg.layer,
                       include_progress=cfg.include_progress_feature, max_budget=max_budget)
        for traj in trajs for cp in traj.checkpoints
    ]
    return np.vstack(feats)


def _sample_episode(
    model: PolicyModel, traj: Trajectory, reward_config: RewardConfig, lam: float,
) -> tuple[float, list[torch.Tensor], list[torch.Tensor]]:
    """One on-policy sampled episode; return (return, per-step log-probs, entropies)."""
    env = StopContinueEnv(traj, reward_config, lam)
    cp = env.reset()
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    while True:
        logits = model.logits(model.features(cp, traj))  # (1, 2), differentiable
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        cp, _r, done, info = env.step(int(action.item()))
        if done:
            return info["reward_total"], log_probs, entropies


def train_policy(
    train_trajs: list[Trajectory],
    cfg: PolicyConfig,
    reward_config: RewardConfig,
    lambda_compute: float,
    *,
    seed: int = 0,
) -> tuple[PolicyModel, list[dict[str, Any]]]:
    """Train one policy at one ``lambda`` via REINFORCE; return (policy, training log)."""
    if not train_trajs:
        raise ValueError("no training trajectories")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    max_budget = max(traj.max_tokens for traj in train_trajs)
    scaler = StandardScaler().fit(_all_features(train_trajs, cfg, max_budget)) if cfg.standardize \
        else StandardScaler(mean_=np.zeros(1), std_=np.ones(1))
    input_dim = _all_features(train_trajs[:1], cfg, max_budget).shape[1]
    net = StopContinuePolicy(input_dim, cfg.hidden_sizes)
    model = PolicyModel(cfg, max_budget, scaler, net)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    log: list[dict[str, Any]] = []
    n = len(train_trajs)
    for iteration in range(cfg.iterations):
        idx = rng.integers(0, n, size=cfg.episodes_per_batch)  # sample with replacement
        returns: list[float] = []
        losses: list[torch.Tensor] = []
        ent_sum = 0.0
        for i in idx:
            G, log_probs, entropies = _sample_episode(
                model, train_trajs[int(i)], reward_config, lambda_compute
            )
            returns.append(G)
            traj_logp = torch.stack(log_probs).sum()
            traj_ent = torch.stack(entropies).sum()
            losses.append((traj_logp, traj_ent, G))
            ent_sum += float(traj_ent.item())

        baseline = float(np.mean(returns)) if cfg.baseline == "batch_mean" else 0.0
        loss = torch.stack([
            -(logp * (G - baseline)) - cfg.entropy_coef * ent
            for logp, ent, G in losses
        ]).mean()

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        optimizer.step()

        log.append({
            "iteration": iteration,
            "mean_return": float(np.mean(returns)),
            "loss": float(loss.item()),
            "mean_entropy": ent_sum / len(idx),
        })

    return model, log
