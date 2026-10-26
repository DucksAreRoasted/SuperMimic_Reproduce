"""Flat-ground tracking environment. The policy lives outside; this only steps physics."""

from __future__ import annotations

import numpy as np

from supermimic.data.motion import MotionLibrary
from supermimic.envs.reset import ResetSampler
from supermimic.envs.sim import VectorSim
from supermimic.robots.profile import RobotProfile


class TrackingEnv:
    def __init__(self, sim: VectorSim, profile: RobotProfile, library: MotionLibrary, timeout_steps: int, seed: int = 0):
        self.sim = sim
        self.profile = profile
        self.library = library
        self.timeout_steps = timeout_steps
        self.rng = np.random.default_rng(seed)
        self.pose_noise = 0.1
        self.sampler = ResetSampler(library)
        n = sim.num_envs
        self.clip = np.zeros(n, dtype=np.int32)
        self.frame = np.zeros(n, dtype=np.int32)
        self.steps = np.zeros(n, dtype=np.int32)
        self.bias = np.zeros((n, profile.n_dof), dtype=np.float32)
        self.friction = np.ones(n, dtype=np.float32)
        self.com = np.zeros((n, 3), dtype=np.float32)
        self.episode_bins: list[list[int]] = [[] for _ in range(n)]
        self.episode_rewards: list[list[float]] = [[] for _ in range(n)]
        self._startup_com()
        for i in range(n):
            self._reset_env(i, record_return=False)

    def _startup_com(self) -> None:
        for i in range(self.sim.num_envs):
            offset = np.array(
                [
                    self.rng.uniform(-0.025, 0.025),
                    self.rng.uniform(-0.05, 0.05),
                    self.rng.uniform(-0.05, 0.05),
                ]
            )
            self.com[i] = offset
            self.sim.set_com_offset(i, offset)

    def xi(self, env: int) -> np.ndarray:
        return np.concatenate([self.friction[env : env + 1], self.bias[env], self.com[env]]).astype(np.float32)

    def _reference(self, env: int, frame: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        f = self.frame[env] if frame is None else frame
        return self.library.frame(int(self.clip[env]), int(f))

    def _reset_env(self, i: int, record_return: bool) -> None:
        if record_return and self.episode_rewards[i]:
            self._commit_return(i, bootstrap=0.0)
        clip, frame = self.sampler.sample(self.rng)
        self.clip[i] = clip
        self.frame[i] = frame
        self.steps[i] = 0
        self.bias[i] = self.rng.uniform(-0.01, 0.01, size=self.profile.n_dof).astype(np.float32)
        self.friction[i] = float(self.rng.uniform(0.3, 1.2))
        self.sim.set_friction(i, float(self.friction[i]))
        qpos, qvel = self.library.frame(clip, frame)
        qpos = qpos.copy()
        qvel = qvel.copy()
        n = self.pose_noise
        qpos[0] += self.rng.uniform(-0.5 * n, 0.5 * n)
        qpos[1] += self.rng.uniform(-0.5 * n, 0.5 * n)
        qpos[2] += self.rng.uniform(-0.1 * n, 0.1 * n)
        qpos[7:] += self.rng.uniform(-n, n, size=self.profile.n_dof)
        self.sim.set_state(i, qpos, qvel)
        self.episode_bins[i] = []
        self.episode_rewards[i] = []

    def _commit_return(self, i: int, bootstrap: float) -> None:
        rewards = self.episode_rewards[i]
        bins = self.episode_bins[i]
        R = bootstrap
        pairs_b, pairs_r = [], []
        for r, b in zip(reversed(rewards), reversed(bins)):
            R = r + 0.99 * R
            pairs_b.append(b)
            pairs_r.append(R)
        self.sampler.update(pairs_b, pairs_r)

    def step(self, executed: np.ndarray) -> dict[str, np.ndarray]:
        """executed is the noisy normalized action. Returns per-env bookkeeping of the transition."""
        scale = self.profile.action_scale
        q0 = self.profile.qpos0[7:]
        targets = q0[None, :] + executed * scale[None, :] - self.bias
        pre_frame = self.frame.copy()
        pre_clip = self.clip.copy()
        self.sim.step(targets.astype(np.float64))
        self.steps += 1
        self.frame += 1
        done = np.zeros(self.sim.num_envs, dtype=np.bool_)
        for i in range(self.sim.num_envs):
            if self.frame[i] >= self.library.clips[int(self.clip[i])].length or self.steps[i] >= self.timeout_steps:
                done[i] = True
        return {
            "state": self.sim.states(),
            "frame": pre_frame,
            "clip": pre_clip,
            "timeout": done,
        }

    def mark_reset(self, i: int, terminal: bool) -> None:
        self._reset_env(i, record_return=True)
        if terminal:
            # A fall has no bootstrap. The commit already used 0.
            pass

    def plant(self, i: int, qpos: np.ndarray, qvel: np.ndarray, friction: float = 1.0) -> None:
        """Deterministic state set used by evaluation."""
        self.bias[i] = 0
        self.friction[i] = friction
        self.com[i] = 0
        self.sim.set_friction(i, friction)
        self.sim.set_state(i, qpos, qvel)
