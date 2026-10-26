"""H-step imagined return. Each dynamics step is checkpointed so the graph stays short."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from supermimic.mdp.observations import canonical, observation, root_error


def push_history(hist: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.cat((hist[:, 1:], obs.unsqueeze(1)), dim=1)


def imagined_return(
    world,
    policy,
    critic,
    s0: torch.Tensor,
    hist0: torch.Tensor,
    xi: torch.Tensor,
    goals: torch.Tensor,
    frames: torch.Tensor,
    ref_qpos: torch.Tensor,
    action_noise: torch.Tensor,
    sensor_noise: torch.Tensor,
    q0: torch.Tensor,
    bias: torch.Tensor,
    gamma: float,
    reward_fn,
    kind: str = "motion",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Policy goals are [B, H+1, goal]. ``frames`` is the single reference frame the termination head reads."""
    state = s0
    hist = hist0
    prev = hist0[:, -1, -world.n_dof :]
    survival = torch.ones(s0.shape[0], device=s0.device)
    objective = state.new_zeros(())
    saved_s, saved_r, saved_d = [], [], []
    horizon = goals.shape[1] - 1

    def step(s, a, x, _world=world):
        return _world.next_state(s, a, x)

    for t in range(horizon):
        obs = observation(state, prev, bias, q0, world.n_dof) + sensor_noise[t]
        hist = push_history(hist, obs)
        action, _ = policy.act(hist, goals[:, t], kind)
        executed = action + action_noise[t]
        state = checkpoint(step, state, executed, xi, use_reentrant=False)
        reward = reward_fn(state, action, prev, ref_qpos[:, t])
        fall = world.termination_prob(state, frames[:, t])
        objective = objective + (gamma**t) * (survival * reward).mean()
        survival = survival * (1.0 - fall)
        prev = action
        saved_s.append(state)
        saved_r.append(reward)
        saved_d.append(fall)
    feat = canonical(state, world.n_dof)
    # ref_qpos is the reference qpos, which is the leading slice of a full state if one was passed.
    ref_pose = ref_qpos[:, horizon, : 7 + world.n_dof]
    value = critic(feat, goals[:, horizon], xi, root_error(state, ref_pose, world.n_dof))
    objective = objective + (gamma**horizon) * (survival * value).mean()
    trace = {
        "state": torch.stack([s.detach() for s in saved_s + [state]], dim=1),
        "reward": torch.stack([r.detach() for r in saved_r], dim=1),
        "terminal": torch.stack([d.detach() for d in saved_d], dim=1),
        "goal": goals.detach(),
        "ref": ref_qpos.detach(),
        "xi": xi.detach(),
    }
    return objective, trace
