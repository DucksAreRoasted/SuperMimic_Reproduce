"""50 Hz tracker runtime. Maps simulator joint order onto the robot SDK vector.

Training does not clip actions. Deployment clips the setpoint into the joint range
and returns a damping request when the pelvis tilt is past the training threshold.
"""

from __future__ import annotations

import numpy as np
import torch

from supermimic.mathutil import projected_gravity, quat_to_rot
from supermimic.mdp.observations import observation
from supermimic.policy.networks import Policy
from supermimic.robots.profile import RobotProfile


class TrackerRuntime:
    def __init__(self, policy: Policy, profile: RobotProfile, device: str = "cpu"):
        self.policy = policy.eval()
        self.profile = profile
        self.device = torch.device(device)
        self.policy.to(self.device)
        n = profile.n_dof
        self.prev = torch.zeros(1, n, device=self.device)
        self.hist = torch.zeros(1, policy.history, profile.obs_dim, device=self.device)
        self.q0 = torch.tensor(profile.qpos0[7:], dtype=torch.float32, device=self.device)
        self.ready = False

    def reset(self) -> None:
        self.prev.zero_()
        self.hist.zero_()
        self.ready = False

    @torch.no_grad()
    def act(self, q_sdk: np.ndarray, qd_sdk: np.ndarray, gyro: np.ndarray, quat_wxyz: np.ndarray, goal: np.ndarray) -> dict[str, np.ndarray | bool]:
        q = self._from_sdk(q_sdk)
        qd = self._from_sdk(qd_sdk)
        n = self.profile.n_dof
        state = torch.zeros(1, 13 + 2 * n, device=self.device)
        state[0, 3:7] = torch.tensor(quat_wxyz, dtype=torch.float32)
        state[0, 7 : 7 + n] = torch.tensor(q, dtype=torch.float32)
        state[0, 7 + n + 3 : 7 + n + 6] = torch.tensor(gyro, dtype=torch.float32)
        state[0, 7 + n + 6 :] = torch.tensor(qd, dtype=torch.float32)
        bias = torch.zeros(1, n, device=self.device)
        obs = observation(state, self.prev, bias, self.q0, n)
        if not self.ready:
            self.hist = obs.unsqueeze(1).expand(1, self.policy.history, -1).contiguous()
            self.ready = True
        else:
            self.hist = torch.cat((self.hist[:, 1:], obs.unsqueeze(1)), dim=1)
        goal_t = torch.tensor(goal, dtype=torch.float32, device=self.device).view(1, -1)
        action, _ = self.policy.act(self.hist, goal_t, "motion")
        self.prev = action
        q_sp = self.profile.qpos0[7:] + action[0].cpu().numpy() * self.profile.action_scale
        q_sp = np.clip(q_sp, self.profile.q_min, self.profile.q_max)
        grav = projected_gravity(quat_to_rot(torch.tensor(quat_wxyz, dtype=torch.float32)))
        damping = bool(abs(float(grav[2]) - (-1.0)) > 0.8 and float(grav[2]) > -0.2)
        return {"q_sdk": self._to_sdk(q_sp), "damping": damping, "action": action[0].cpu().numpy()}

    def _from_sdk(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)[list(self.profile.sdk_indices)]

    def _to_sdk(self, sim_values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.profile.sdk_width, dtype=np.float64)
        for i, slot in enumerate(self.profile.sdk_indices):
            out[slot] = sim_values[i]
        return out
