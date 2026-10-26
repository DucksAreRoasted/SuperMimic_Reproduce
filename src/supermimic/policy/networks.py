"""Motion encoder, optional velocity encoder, shared decoder, per-interface critic."""

from __future__ import annotations

import torch
from torch import nn


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, last_gain: float = 1.0, zero_last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.SiLU())
        d = h
    last = nn.Linear(d, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    else:
        nn.init.xavier_uniform_(last.weight, gain=last_gain)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, history: int, latent: int, hidden_enc, hidden_dec, hidden_vel=(256, 256)):
        super().__init__()
        self.history = history
        self.obs_dim = obs_dim
        self.motion = mlp(goal_dim, hidden_enc, latent)
        self.velocity = mlp(3 + history * obs_dim, hidden_vel, latent)
        self.decoder = mlp(latent + history * obs_dim, hidden_dec, act_dim, last_gain=0.01)

    def encode(self, hist: torch.Tensor, goal: torch.Tensor, kind: str) -> torch.Tensor:
        flat = hist.reshape(hist.shape[0], -1)
        if kind == "velocity":
            z = self.velocity(torch.cat((goal, flat), dim=-1))
        else:
            z = self.motion(goal)
        return torch.tanh(z)

    def act(self, hist: torch.Tensor, goal: torch.Tensor, kind: str = "motion") -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(hist, goal, kind)
        flat = hist.reshape(hist.shape[0], -1)
        return self.decoder(torch.cat((z, flat), dim=-1)), z


class Critic(nn.Module):
    def __init__(self, feat_dim: int, goal_dim: int, xi_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.net = mlp(feat_dim + goal_dim + xi_dim + 4, hidden, 1)

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi: torch.Tensor, root_err: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((feat, goal, xi, root_err), dim=-1)).squeeze(-1)
