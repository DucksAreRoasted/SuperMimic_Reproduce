"""Fixed-length per-environment ring buffer. Observations are rebuilt from states."""

from __future__ import annotations

import numpy as np
import torch


class Replay:
    def __init__(self, num_envs: int, capacity: int, state_dim: int, act_dim: int, xi_dim: int):
        self.n = num_envs
        self.cap = capacity
        self.ptr = np.zeros(num_envs, dtype=np.int64)
        self.count = np.zeros(num_envs, dtype=np.int64)
        self.state = np.zeros((num_envs, capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((num_envs, capacity, act_dim), dtype=np.float32)
        self.executed = np.zeros((num_envs, capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros((num_envs, capacity), dtype=np.float32)
        self.terminal = np.zeros((num_envs, capacity), dtype=np.float32)
        self.ep_start = np.ones((num_envs, capacity), dtype=np.bool_)
        self.xi = np.zeros((num_envs, capacity, xi_dim), dtype=np.float32)
        self.clip = np.zeros((num_envs, capacity), dtype=np.int32)
        self.frame = np.zeros((num_envs, capacity), dtype=np.int32)

    def add(self, env: int, state, action, executed, reward, terminal, ep_start, xi, clip, frame) -> None:
        t = int(self.ptr[env] % self.cap)
        self.state[env, t] = state
        self.action[env, t] = action
        self.executed[env, t] = executed
        self.reward[env, t] = reward
        self.terminal[env, t] = terminal
        self.ep_start[env, t] = ep_start
        self.xi[env, t] = xi
        self.clip[env, t] = clip
        self.frame[env, t] = frame
        self.ptr[env] += 1
        self.count[env] = min(self.count[env] + 1, self.cap)

    def _index(self, env: int, age_from_oldest: int) -> int:
        start = int(self.ptr[env] - self.count[env])
        return int((start + age_from_oldest) % self.cap)

    def sample_windows(self, batch: int, k: int, rng: np.random.Generator):
        states, actions, rewards, terminals, xis, clips, frames = [], [], [], [], [], [], []
        found = 0
        tries = 0
        while found < batch and tries < batch * 40:
            tries += 1
            env = int(rng.integers(0, self.n))
            if self.count[env] < k + 1:
                continue
            start_age = int(rng.integers(0, self.count[env] - k))
            idxs = [self._index(env, start_age + j) for j in range(k + 1)]
            if self.ep_start[env, idxs[1:]].any():
                continue
            states.append(self.state[env, idxs])
            actions.append(self.executed[env, idxs[:-1]])
            rewards.append(self.reward[env, idxs[:-1]])
            terminals.append(self.terminal[env, idxs[:-1]])
            xis.append(self.xi[env, idxs[0]])
            clips.append(self.clip[env, idxs[:-1]])
            frames.append(self.frame[env, idxs[:-1]])
            found += 1
        if found == 0:
            return None
        return {
            "state": torch.tensor(np.stack(states)),
            "action": torch.tensor(np.stack(actions)),
            "reward": torch.tensor(np.stack(rewards)),
            "terminal": torch.tensor(np.stack(terminals)),
            "xi": torch.tensor(np.stack(xis)),
            "clip": torch.tensor(np.stack(clips)),
            "frame": torch.tensor(np.stack(frames)),
        }

    def sample_starts(self, batch: int, history: int, rng: np.random.Generator):
        out = []
        tries = 0
        while len(out) < batch and tries < batch * 40:
            tries += 1
            env = int(rng.integers(0, self.n))
            if self.count[env] < 2:
                continue
            age = int(rng.integers(0, self.count[env]))
            idx = self._index(env, age)
            hist_idx = []
            ok = True
            for back in range(history - 1, -1, -1):
                if age - back < 0 or self.ep_start[env, self._index(env, max(age - back, 0))] and back != 0 and age - back > 0:
                    # Pad with the oldest in-episode state.
                    pass
                h_age = max(0, age - back)
                h_idx = self._index(env, h_age)
                if h_age > 0 and self.ep_start[env, h_idx] and back != 0:
                    h_idx = idx
                hist_idx.append(h_idx)
            if not ok:
                continue
            out.append((env, idx, hist_idx))
        return out
