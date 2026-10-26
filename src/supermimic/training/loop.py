"""Algorithm A.1. The imagined batch is split so the update still matches Table A.5."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from supermimic.config import TrainConfig, imag_chunk_size, memory_budget_bytes
from supermimic.data.motion import MotionLibrary
from supermimic.envs.replay import Replay
from supermimic.envs.sim import VectorSim
from supermimic.envs.tracking import TrackingEnv
from supermimic.imagination.rollout import imagined_return
from supermimic.kinematics.fk import KinTree
from supermimic.mdp.features import body_kinematics, sensor_noise
from supermimic.mdp.observations import goal_frame, observation, state_error
from supermimic.mdp.rewards import tracking_reward, velocity_reward
from supermimic.mdp.terminations import tracking_failed, velocity_failed
from supermimic.policy.networks import Critic, Policy
from supermimic.robots.profile import RobotProfile
from supermimic.training.gae import critic_values, gae_targets
from supermimic.world_model.model import WorldModel


def _sigma(cfg: TrainConfig) -> dict[str, float]:
    return {"pos": cfg.sigma_pos, "rot": cfg.sigma_rot, "lin": cfg.sigma_lin, "ang": cfg.sigma_ang, "q": cfg.sigma_q, "qd": cfg.sigma_qd}


class Trainer:
    def __init__(self, cfg: TrainConfig, profile: RobotProfile, model_xml: str, library: MotionLibrary, kin: KinTree):
        self.cfg = cfg
        self.profile = profile
        self.library = library
        self.kin = kin.to(torch.device("cpu" if cfg.device == "cuda" and not torch.cuda.is_available() else cfg.device))
        if cfg.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            total = torch.cuda.get_device_properties(self.device).total_memory
        else:
            total = None
        self.budget = memory_budget_bytes(self.device.type, total, cfg.memory_budget)
        self.sim = VectorSim(kin_model_placeholder(profile, model_xml), profile, cfg.num_envs, profile.scene_xml)
        self.env = TrackingEnv(self.sim, profile, library, timeout_steps=int(cfg.episode_timeout_s / cfg.control_dt), seed=cfg.seed)
        n = profile.n_dof
        self.world = WorldModel(n, profile.xi_dim, profile.goal_frame_dim, cfg.hidden_dyn, cfg.hidden_term, _sigma(cfg)).to(self.device)
        goal_dim = cfg.future_frames * profile.goal_frame_dim
        self.policy = Policy(profile.obs_dim, goal_dim, n, cfg.history, cfg.latent_dim, cfg.hidden_enc, cfg.hidden_dec).to(self.device)
        feat = 13 + 2 * n
        self.critic = Critic(feat, goal_dim, profile.xi_dim, cfg.hidden_critic).to(self.device)
        self.v_critic = Critic(feat, 3, profile.xi_dim, cfg.hidden_critic).to(self.device)
        self.opt_w = torch.optim.Adam(self.world.parameters(), lr=cfg.lr)
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, betas=cfg.policy_betas)
        self.opt_c = torch.optim.Adam(list(self.critic.parameters()) + list(self.v_critic.parameters()), lr=cfg.lr)
        self.replay = Replay(cfg.num_envs, cfg.buffer_steps, 13 + 2 * n, n, profile.xi_dim)
        self.q0 = torch.tensor(profile.qpos0[7:], dtype=torch.float32, device=self.device)
        self.qmin = torch.tensor(profile.q_min, dtype=torch.float32, device=self.device)
        self.qmax = torch.tensor(profile.q_max, dtype=torch.float32, device=self.device)
        self.tracked = kin.body_ids(profile.tracked_bodies).to(self.device)
        self.anchor = profile.tracked_bodies.index(profile.anchor_body)
        self.term_body = kin.body_ids((profile.termination_body,)).to(self.device)
        self.hist = torch.zeros(cfg.num_envs, cfg.history, profile.obs_dim, device=self.device)
        self.prev = torch.zeros(cfg.num_envs, n, device=self.device)
        self._fill_history()
        self.update_index = 0
        self.total_updates = max(1, int(cfg.total_env_steps * cfg.updates_per_env_step))
        self.rng = np.random.default_rng(cfg.seed)
        self.acc = 0.0
        self.recent_return: list[float] = []
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def _fill_history(self) -> None:
        obs = self._obs_torch()
        self.hist = obs[:, None, :].expand(-1, self.cfg.history, -1).contiguous()

    def _obs_torch(self) -> torch.Tensor:
        state = torch.tensor(self.sim.states(), dtype=torch.float32, device=self.device)
        bias = torch.tensor(self.env.bias, dtype=torch.float32, device=self.device)
        return observation(state, self.prev, bias, self.q0, self.profile.n_dof)

    def _goals_for(self, clips, frames) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns packed future goals, the current frame vector, and the reference state at that frame."""
        k = self.cfg.future_frames
        goals, frames_g, refs = [], [], []
        for clip, frame in zip(clips, frames):
            packed = []
            first_state = None
            first_frame = None
            for j in range(k):
                qpos, qvel = self.library.frame(int(clip), int(frame) + j)
                state = np.concatenate([qpos, qvel]).astype(np.float32)
                qt = torch.tensor(qpos, dtype=torch.float32, device=self.device)
                vt = torch.tensor(qvel, dtype=torch.float32, device=self.device)
                packed.append(goal_frame(qt, vt, self.q0, self.profile.n_dof))
                if j == 0:
                    first_frame = packed[0]
                    first_state = torch.tensor(state, dtype=torch.float32, device=self.device)
            goals.append(torch.cat(packed, dim=-1))
            frames_g.append(first_frame)
            refs.append(first_state)
        return torch.stack(goals), torch.stack(frames_g), torch.stack(refs)

    def collect(self) -> None:
        goals, _, _ = self._goals_for(self.env.clip, self.env.frame)
        obs = self._obs_torch()
        if self.cfg.action_noise_std > 0:
            obs = obs + sensor_noise(obs, self.profile.n_dof)
        self.hist = torch.cat((self.hist[:, 1:], obs.unsqueeze(1)), dim=1)
        with torch.no_grad():
            action, _ = self.policy.act(self.hist, goals, "motion")
        noise = torch.randn_like(action) * self.cfg.action_noise_std
        executed = action + noise
        pre_state = self.sim.states().copy()
        pre_clip = self.env.clip.copy()
        pre_frame = self.env.frame.copy()
        info = self.env.step(executed.detach().cpu().numpy())
        post = torch.tensor(info["state"], dtype=torch.float32, device=self.device)
        ref = self._reference_states(pre_clip, pre_frame)
        reward = self._tracking_reward(post, ref, action, self.prev)
        failed = self._failed(post, ref)
        self.prev = action.detach()
        for i in range(self.cfg.num_envs):
            self.replay.add(
                i,
                pre_state[i],
                action[i].detach().cpu().numpy(),
                executed[i].detach().cpu().numpy(),
                float(reward[i]),
                float(failed[i]),
                self.env.steps[i] == 1,
                self.env.xi(i),
                int(pre_clip[i]),
                int(pre_frame[i]),
            )
            self.env.episode_rewards[i].append(float(reward[i]))
            self.env.episode_bins[i].append(self.library.bin_of(int(pre_clip[i]), int(pre_frame[i])))
            if bool(failed[i]) or bool(info["timeout"][i]):
                self.recent_return.append(float(sum(self.env.episode_rewards[i])))
                self.env.mark_reset(i, terminal=bool(failed[i]))
                self.prev[i] = 0
                fresh = self._obs_torch()[i]
                self.hist[i] = fresh.unsqueeze(0).expand(self.cfg.history, -1).contiguous()
        self.acc += self.cfg.updates_per_env_step

    def _reference_states(self, clips, frames) -> torch.Tensor:
        rows = []
        for clip, frame in zip(clips, frames):
            qpos, qvel = self.library.frame(int(clip), int(frame))
            rows.append(np.concatenate([qpos, qvel]).astype(np.float32))
        return torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)

    def _tracking_reward(self, state, ref, action, prev) -> torch.Tensor:
        pos, rot, lin, ang = body_kinematics(self.kin, state, self.tracked, self.profile.n_dof)
        rpos, rrot, rlin, rang = body_kinematics(self.kin, ref, self.tracked, self.profile.n_dof)
        q = state[:, 7 : 7 + self.profile.n_dof]
        return tracking_reward(
            pos, rot, lin, ang, rpos, rrot, rlin, rang, self.anchor, q, self.qmin, self.qmax, action, prev, self.cfg.control_dt, self.profile.pos_sigma
        )

    def _failed(self, state, ref) -> torch.Tensor:
        pos, rot, _, _ = body_kinematics(self.kin, state, self.term_body, self.profile.n_dof)
        rpos, rrot, _, _ = body_kinematics(self.kin, ref, self.term_body, self.profile.n_dof)
        quat = state[:, 3:7]
        rquat = ref[:, 3:7]
        return tracking_failed(pos[:, 0, 2], rpos[:, 0, 2], quat, rquat, self.profile.z_terminate)

    def maybe_update(self) -> dict[str, float] | None:
        if self.acc < 1:
            return None
        self.acc -= 1
        logs = {}
        logs.update(self._fit_world())
        logs.update(self._improve_policy())
        self._set_lr()
        self.update_index += 1
        return logs

    def _set_lr(self) -> None:
        progress = min(1.0, self.update_index / max(1, int(self.total_updates * self.cfg.cosine_fraction)))
        scale = self.cfg.lr_final_scale + (1 - self.cfg.lr_final_scale) * 0.5 * (1 + math.cos(math.pi * progress))
        for opt in (self.opt_w, self.opt_p, self.opt_c):
            for group in opt.param_groups:
                group["lr"] = self.cfg.lr * scale

    def _fit_world(self) -> dict[str, float]:
        self.world.train()
        losses = []
        for _ in range(self.cfg.wm_batches):
            batch = self.replay.sample_windows(self.cfg.wm_batch, self.cfg.k, self.rng)
            if batch is None:
                return {"wm_loss": float("nan")}
            batch = {k: v.to(self.device) for k, v in batch.items()}
            state = batch["state"][:, 0]
            loss = state.new_zeros(())
            for j in range(self.cfg.k):
                state = self.world.next_state(state, batch["action"][:, j], batch["xi"])
                target = batch["state"][:, j + 1]
                err = state_error(state, target, self.profile.n_dof, _sigma(self.cfg))
                frame = self._reference_frame(batch["clip"][:, j], batch["frame"][:, j])
                prob = self.world.termination_prob(state, frame).clamp(1e-4, 1 - 1e-4)
                loss = loss + err.pow(2).sum(-1).mean()
                loss = loss + self.cfg.term_loss_weight * torch.nn.functional.binary_cross_entropy(prob, batch["terminal"][:, j])
            self.opt_w.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.world.parameters(), self.cfg.grad_clip)
            self.opt_w.step()
            losses.append(float(loss.detach()))
        return {"wm_loss": float(np.mean(losses)) if losses else float("nan")}

    def _reference_frame(self, clips: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        rows = []
        for clip, frame in zip(clips.tolist(), frames.tolist()):
            qpos, qvel = self.library.frame(int(clip), int(frame))
            qt = torch.tensor(qpos, dtype=torch.float32, device=self.device)
            vt = torch.tensor(qvel, dtype=torch.float32, device=self.device)
            rows.append(goal_frame(qt, vt, self.q0, self.profile.n_dof))
        return torch.stack(rows)

    def _improve_policy(self) -> dict[str, float]:
        starts = self.replay.sample_starts(self.cfg.imag_batch, self.cfg.history, self.rng)
        if len(starts) < 2:
            return {"policy_loss": float("nan")}
        prepared = self._prepare_starts(starts)
        for p in self.world.parameters():
            p.requires_grad_(False)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        self.opt_p.zero_grad(set_to_none=True)
        allocated = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        chunk = imag_chunk_size(self.budget, allocated, len(starts))
        traces = []
        total = len(starts)
        policy_loss = 0.0
        while True:
            try:
                for s in range(0, total, chunk):
                    piece = self._slice(prepared, s, min(total, s + chunk))
                    objective, trace = self._rollout(piece, "motion")
                    weight = piece["s0"].shape[0] / total
                    (weight * (-objective)).backward()
                    policy_loss += float(objective.detach()) * weight
                    traces.append(trace)
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                self.opt_p.zero_grad(set_to_none=True)
                traces.clear()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        for p in self.policy.parameters():
            if p.grad is not None:
                p.grad.clamp_(-self.cfg.policy_grad_element_clip, self.cfg.policy_grad_element_clip)
        self.opt_p.step()
        for p in self.world.parameters():
            p.requires_grad_(True)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        self._fit_critic(traces, self.critic)
        if self.cfg.train_velocity:
            self._velocity_update(prepared)
        return {"policy_loss": policy_loss, "imag_chunk": float(chunk)}

    def _velocity_update(self, prepared: dict[str, torch.Tensor]) -> None:
        """Velocity reward updates the velocity encoder. Decoder weights are stop-grad."""
        for p in self.policy.decoder.parameters():
            p.requires_grad_(False)
        for p in self.policy.motion.parameters():
            p.requires_grad_(False)
        for p in self.world.parameters():
            p.requires_grad_(False)
        self.opt_p.zero_grad(set_to_none=True)
        objective, trace = self._rollout(prepared, "velocity")
        # Latent alignment toward the motion encoder on this same goal, stop-grad on the motion latent.
        with torch.no_grad():
            motion_z = self.policy.encode(prepared["hist"], prepared["goals"][:, 0], "motion")
        vel_z = self.policy.encode(prepared["hist"], prepared["command"], "velocity")
        align = self.cfg.latent_align_weight * (vel_z - motion_z).pow(2).mean()
        (-objective + align).backward()
        self.opt_p.step()
        for p in list(self.policy.decoder.parameters()) + list(self.policy.motion.parameters()) + list(self.world.parameters()):
            p.requires_grad_(True)
        self._fit_critic([trace], self.v_critic)

    def _fit_critic(self, traces: list[dict], critic: Critic) -> None:
        state = torch.cat([t["state"] for t in traces], dim=0)
        reward = torch.cat([t["reward"] for t in traces], dim=0)
        terminal = torch.cat([t["terminal"] for t in traces], dim=0)
        goal = torch.cat([t["goal"] for t in traces], dim=0)
        ref = torch.cat([t["ref"] for t in traces], dim=0)
        xi = torch.cat([t["xi"] for t in traces], dim=0)
        with torch.no_grad():
            value = critic_values(critic, state, goal, xi, ref, self.profile.n_dof)
            target = gae_targets(reward, terminal, value, self.cfg.gamma, self.cfg.lam)
        b, horizon = reward.shape
        feat_s = state[:, :horizon].reshape(b * horizon, -1).detach()
        feat = __import__("supermimic.mdp.observations", fromlist=["canonical"]).canonical(feat_s, self.profile.n_dof)
        from supermimic.mdp.observations import canonical as canon
        from supermimic.mdp.observations import root_error as rerr

        feat = canon(feat_s, self.profile.n_dof)
        root = rerr(feat_s, ref[:, :horizon].reshape(b * horizon, -1)[:, : 7 + self.profile.n_dof], self.profile.n_dof)
        goal_f = goal[:, :horizon].reshape(b * horizon, -1)
        xi_f = xi[:, None, :].expand(b, horizon, -1).reshape(b * horizon, -1)
        target_f = target.reshape(-1)
        n = feat.shape[0]
        mb = max(1, n // self.cfg.critic_minibatches)
        for _ in range(self.cfg.critic_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                pred = critic(feat[idx], goal_f[idx], xi_f[idx], root[idx])
                loss = (pred - target_f[idx]).pow(2).mean()
                self.opt_c.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.grad_clip)
                self.opt_c.step()

    def _prepare_starts(self, starts) -> dict[str, torch.Tensor]:
        s0, hist, xi, goals, frames, refs, bias, command = [], [], [], [], [], [], [], []
        k = self.cfg.future_frames
        for env, idx, hist_idx in starts:
            s0.append(self.replay.state[env, idx])
            obs_rows = []
            for h in hist_idx:
                st = torch.tensor(self.replay.state[env, h], dtype=torch.float32, device=self.device)
                prev = torch.tensor(self.replay.action[env, h], dtype=torch.float32, device=self.device)
                b = torch.tensor(self.replay.xi[env, h, 1 : 1 + self.profile.n_dof], dtype=torch.float32, device=self.device)
                obs_rows.append(observation(st[None], prev[None], b[None], self.q0, self.profile.n_dof)[0])
            hist.append(torch.stack(obs_rows))
            xi.append(self.replay.xi[env, idx])
            clip = int(self.replay.clip[env, idx])
            frame = int(self.replay.frame[env, idx])
            packed, frame_g, ref = self._goals_for([clip], [frame])
            # Future windows along the horizon are rebuilt inside _rollout from clip/frame.
            goals.append((clip, frame))
            frames.append(frame_g[0])
            refs.append(ref[0])
            bias.append(self.replay.xi[env, idx, 1 : 1 + self.profile.n_dof])
            qpos, qvel = self.library.frame(clip, frame)
            yaw = _yaw(qpos[3:7])
            c, s = math.cos(yaw), math.sin(yaw)
            vx, vy = qvel[0], qvel[1]
            command.append([c * vx + s * vy, -s * vx + c * vy, qvel[5]])
        # Build horizon goals per start below, once we know H.
        meta = goals
        H = self.cfg.horizon
        goal_seq, frame_seq, ref_seq = [], [], []
        for clip, frame in meta:
            g_steps, f_steps, r_steps = [], [], []
            for t in range(H + 1):
                g, f, r = self._goals_for([clip], [frame + t])
                g_steps.append(g[0])
                f_steps.append(f[0])
                r_steps.append(r[0])
            goal_seq.append(torch.stack(g_steps))
            frame_seq.append(torch.stack(f_steps))
            ref_seq.append(torch.stack(r_steps))
        return {
            "s0": torch.tensor(np.stack(s0), dtype=torch.float32, device=self.device),
            "hist": torch.stack(hist),
            "xi": torch.tensor(np.stack(xi), dtype=torch.float32, device=self.device),
            "goals": torch.stack(goal_seq),
            "frames": torch.stack(frame_seq),
            "ref": torch.stack(ref_seq),
            "bias": torch.tensor(np.stack(bias), dtype=torch.float32, device=self.device),
            "command": torch.tensor(command, dtype=torch.float32, device=self.device),
        }

    def _slice(self, prepared, a, b) -> dict[str, torch.Tensor]:
        return {k: v[a:b] for k, v in prepared.items()}

    def _rollout(self, piece: dict[str, torch.Tensor], kind: str):
        b = piece["s0"].shape[0]
        h = self.cfg.horizon
        n = self.profile.n_dof
        # Action noise is only for simulator collection. Imagined rollouts are deterministic.
        action_noise = torch.zeros(h, b, n, device=self.device)
        obs = torch.zeros(h, b, self.profile.obs_dim, device=self.device)
        sensor = sensor_noise(obs, n)
        goals = piece["goals"] if kind == "motion" else piece["command"][:, None, :].expand(-1, h + 1, -1)
        critic = self.critic if kind == "motion" else self.v_critic

        def reward_fn(state, action, prev, ref_qpos):
            ref_state = piece_ref(ref_qpos, state)
            if kind == "velocity":
                return self._velocity_reward(state, action, prev, piece["command"])
            return self._tracking_reward(state, ref_state, action, prev)

        def piece_ref(ref_qpos, state):
            # ref_qpos is [B, nq]; build a state by pairing it with zeros for velocity if needed.
            if ref_qpos.shape[-1] == state.shape[-1]:
                return ref_qpos
            qvel = ref_qpos.new_zeros(ref_qpos.shape[0], state.shape[-1] - ref_qpos.shape[-1])
            return torch.cat((ref_qpos, qvel), dim=-1)

        return imagined_return(
            self.world,
            self.policy,
            critic,
            piece["s0"],
            piece["hist"],
            piece["xi"],
            goals,
            piece["frames"],
            piece["ref"],
            action_noise,
            sensor,
            self.q0,
            piece["bias"],
            self.cfg.gamma,
            reward_fn,
            kind,
        )

    def _velocity_reward(self, state, action, prev, command):
        from supermimic.mathutil import quat_to_rot, yaw_from_rot, rotz

        n = self.profile.n_dof
        qpos = state[:, : 7 + n]
        qvel = state[:, 7 + n :]
        yaw = yaw_from_rot(quat_to_rot(qpos[:, 3:7]))
        v_h = torch.matmul(rotz(-yaw), qvel[:, :3].unsqueeze(-1)).squeeze(-1)[:, :2]
        # Yaw rate from body angular velocity's vertical component, approximated by the world yaw of w.
        rot = quat_to_rot(qpos[:, 3:7])
        w_world = torch.matmul(rot, qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)
        yaw_rate = w_world[:, 2]
        # Torso roll from the projected x axis.
        roll = torch.atan2(rot[:, 2, 1], rot[:, 2, 2])
        q = qpos[:, 7:]
        return velocity_reward(v_h, yaw_rate, command, roll, w_world[:, 1], w_world[:, 0], q, self.qmin, self.qmax, action, prev, self.cfg.control_dt)

    def train(self, steps: int | None = None) -> list[dict]:
        n = steps or self.cfg.total_env_steps
        logs = []
        for _ in range(n):
            self.collect()
            row = self.maybe_update()
            if row is not None:
                row["return"] = float(np.mean(self.recent_return[-20:])) if self.recent_return else 0.0
                row["step"] = self.update_index
                logs.append(row)
                if self.update_index % 10 == 0:
                    print(
                        f"update {self.update_index} wm {row.get('wm_loss', float('nan')):.3f} "
                        f"policy {row.get('policy_loss', float('nan')):.3f} return {row['return']:.3f}",
                        flush=True,
                    )
                if self.update_index % 10 == 0:
                    peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
                    row["peak_bytes"] = peak
                    if peak > self.budget and self.device.type == "cuda":
                        raise RuntimeError(f"peak allocation {peak} exceeds budget {self.budget}")
        return logs


def kin_model_placeholder(profile: RobotProfile, xml: str):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(profile.scene_xml))


def _yaw(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
