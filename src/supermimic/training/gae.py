"""GAE targets on an imagined rollout. Targets are detached before the critic is fit."""

from __future__ import annotations

import torch

from supermimic.mdp.observations import canonical, root_error


def gae_targets(reward: torch.Tensor, terminal: torch.Tensor, value: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """reward, terminal: [B, H]. value: [B, H+1] with value[:, -1] the bootstrap."""
    horizon = reward.shape[1]
    target = torch.zeros_like(reward)
    nxt = value[:, -1]
    for t in range(horizon - 1, -1, -1):
        mix = (1 - lam) * value[:, t + 1] + lam * nxt
        nxt = reward[:, t] + gamma * (1 - terminal[:, t]) * mix
        target[:, t] = nxt
    return target.detach()


def critic_values(critic, states, goals, xi, ref, n_dof: int) -> torch.Tensor:
    """states [B, T, S], goals [B, T, G], ref [B, T, nq]."""
    b, t, _ = states.shape
    flat_s = states.reshape(b * t, -1)
    feat = canonical(flat_s, n_dof)
    root = root_error(flat_s, ref.reshape(b * t, -1)[:, : 7 + n_dof], n_dof)
    goal = goals.reshape(b * t, -1)
    context = xi[:, None, :].expand(b, t, -1).reshape(b * t, -1) if xi.ndim == 2 else xi.reshape(b * t, -1)
    values = critic(feat, goal, context, root)
    return values.view(b, t)
