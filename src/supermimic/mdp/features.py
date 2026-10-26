"""Body kinematics for a batch of states, plus sensor noise."""

from __future__ import annotations

import torch

from supermimic.kinematics.fk import KinTree
from supermimic.mdp.observations import split_state


def body_kinematics(kin: KinTree, state: torch.Tensor, body_ids: torch.Tensor, n_dof: int):
    qpos, qvel = split_state(state, n_dof)
    xpos, xmat, lin, ang = kin.forward(qpos, qvel)
    idx = body_ids.to(state.device)
    return (
        xpos.index_select(1, idx),
        xmat.index_select(1, idx),
        lin.index_select(1, idx),
        ang.index_select(1, idx),
    )


def sensor_noise(obs: torch.Tensor, n_dof: int) -> torch.Tensor:
    noise = torch.zeros_like(obs)
    n = n_dof
    noise[..., :n] = torch.empty_like(noise[..., :n]).uniform_(-0.01, 0.01)
    noise[..., n : 2 * n] = torch.empty_like(noise[..., n : 2 * n]).uniform_(-0.5, 0.5)
    noise[..., 2 * n : 2 * n + 3] = torch.empty_like(noise[..., 2 * n : 2 * n + 3]).uniform_(-0.2, 0.2)
    noise[..., 2 * n + 3 : 2 * n + 6] = torch.empty_like(noise[..., 2 * n + 3 : 2 * n + 6]).uniform_(-0.05, 0.05)
    return noise
