"""Appendix F style rollout: one pass from the first frame, friction 1, no noise."""

from __future__ import annotations

import numpy as np
import torch

from supermimic.data.motion import MotionLibrary
from supermimic.envs.sim import VectorSim
from supermimic.kinematics.fk import KinTree
from supermimic.mdp.features import body_kinematics
from supermimic.mdp.observations import goal_frame, observation
from supermimic.mdp.rewards import tracking_reward
from supermimic.mdp.terminations import tracking_failed
from supermimic.policy.networks import Policy
from supermimic.robots.profile import RobotProfile

MAX_STEP_REWARD = 0.1  # (1+1+1+1+0.5+0.5) * dt


@torch.no_grad()
def evaluate_library(policy: Policy, profile: RobotProfile, kin: KinTree, library: MotionLibrary, device: torch.device, future_frames: int = 10) -> dict[str, float]:
    returns = []
    fails = 0
    q0 = torch.tensor(profile.qpos0[7:], dtype=torch.float32, device=device)
    qmin = torch.tensor(profile.q_min, dtype=torch.float32, device=device)
    qmax = torch.tensor(profile.q_max, dtype=torch.float32, device=device)
    tracked = kin.body_ids(profile.tracked_bodies)
    anchor = profile.tracked_bodies.index(profile.anchor_body)
    n = profile.n_dof
    sim = VectorSim(kin_model(profile), profile, 1, profile.scene_xml)
    sim.set_friction(0, 1.0)
    for clip_id, clip in enumerate(library.clips):
        qpos, qvel = clip.qpos[0], clip.qvel[0]
        sim.set_state(0, qpos, qvel)
        prev = torch.zeros(1, n, device=device)
        hist = None
        total = 0.0
        failed = False
        for frame in range(clip.length):
            state_np = sim.states()
            state = torch.tensor(state_np, dtype=torch.float32, device=device)
            bias = torch.zeros(1, n, device=device)
            obs = observation(state, prev, bias, q0, n)
            hist = obs.unsqueeze(1).expand(1, policy.history, obs.shape[-1]).contiguous() if hist is None else torch.cat((hist[:, 1:], obs.unsqueeze(1)), dim=1)
            packed = []
            ref_qpos, ref_qvel = library.frame(clip_id, frame)
            for j in range(future_frames):
                qp, qv = library.frame(clip_id, frame + j)
                packed.append(goal_frame(torch.tensor(qp, dtype=torch.float32, device=device), torch.tensor(qv, dtype=torch.float32, device=device), q0, n))
            # future_frames is not stored on the policy. Use 10, matching the paper.
            goal = torch.cat(packed, dim=-1).unsqueeze(0)
            action, _ = policy.act(hist, goal, "motion")
            target = profile.qpos0[7:] + action[0].detach().cpu().numpy() * profile.action_scale
            sim.step(target[None, :])
            nxt = torch.tensor(sim.states(), dtype=torch.float32, device=device)
            ref = torch.tensor(np.concatenate([ref_qpos, ref_qvel])[None, :], dtype=torch.float32, device=device)
            pos, rot, lin, ang = body_kinematics(kin, nxt, tracked, n)
            rpos, rrot, rlin, rang = body_kinematics(kin, ref, tracked, n)
            q = nxt[:, 7 : 7 + n]
            reward = tracking_reward(pos, rot, lin, ang, rpos, rrot, rlin, rang, anchor, q, qmin, qmax, action, prev, 0.02, profile.pos_sigma)
            total += float(reward)
            quat = nxt[:, 3:7]
            if bool(tracking_failed(pos[:, 0, 2], rpos[:, 0, 2], quat, ref[:, 3:7], profile.z_terminate)):
                failed = True
                break
            prev = action
        returns.append(total / (MAX_STEP_REWARD * clip.length))
        fails += int(failed)
    return {"normalized_return": float(np.mean(returns)), "failures": fails, "clips": len(library.clips)}


def kin_model(profile: RobotProfile):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(profile.scene_xml))
