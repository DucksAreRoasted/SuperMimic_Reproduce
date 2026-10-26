"""Deterministic world model. Dynamics are learned. Observation and reward stay analytic."""

from __future__ import annotations

import torch
from torch import nn

from supermimic.mdp.observations import canonical, integrate_delta
from supermimic.policy.networks import mlp


class WorldModel(nn.Module):
    def __init__(self, n_dof: int, xi_dim: int, goal_frame_dim: int, hidden_dyn: tuple[int, ...], hidden_term: tuple[int, ...], sigmas: dict[str, float]):
        super().__init__()
        self.n_dof = n_dof
        self.feat_dim = 13 + 2 * n_dof
        delta_dim = 12 + 2 * n_dof
        self.dyn = mlp(self.feat_dim + n_dof + xi_dim, hidden_dyn, delta_dim, zero_last=True)
        self.term = mlp(self.feat_dim + goal_frame_dim, hidden_term, 1)
        sigma = []
        sigma += [sigmas["pos"]] * 3
        sigma += [sigmas["rot"]] * 3
        sigma += [sigmas["lin"]] * 3
        sigma += [sigmas["ang"]] * 3
        sigma += [sigmas["q"]] * n_dof
        sigma += [sigmas["qd"]] * n_dof
        self.register_buffer("delta_scale", torch.tensor(sigma, dtype=torch.float32))

    def delta(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        feat = canonical(state, self.n_dof)
        raw = self.dyn(torch.cat((feat, action, xi), dim=-1))
        return raw * self.delta_scale

    def next_state(self, state: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        return integrate_delta(state, self.delta(state, action, xi), self.n_dof)

    def termination_prob(self, state: torch.Tensor, goal_frame: torch.Tensor) -> torch.Tensor:
        feat = canonical(state, self.n_dof)
        return torch.sigmoid(self.term(torch.cat((feat, goal_frame), dim=-1))).squeeze(-1)
