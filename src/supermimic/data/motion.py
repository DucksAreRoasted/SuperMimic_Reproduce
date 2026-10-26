"""50 Hz motion clips. A clip is a free-joint trajectory in MuJoCo order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermimic.robots.profile import RobotProfile


@dataclass
class MotionClip:
    name: str
    qpos: np.ndarray
    qvel: np.ndarray

    @property
    def length(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return self.length / 50.0


class MotionLibrary:
    def __init__(self, clips: list[MotionClip], fps: int = 50):
        if not clips:
            raise ValueError("empty motion library")
        self.clips = clips
        self.fps = fps
        self.bin_offsets = []
        self.n_bins = 0
        for clip in clips:
            self.bin_offsets.append(self.n_bins)
            self.n_bins += max(1, int(np.ceil(clip.duration)))

    def __len__(self) -> int:
        return len(self.clips)

    def frame(self, clip: int, frame: int) -> tuple[np.ndarray, np.ndarray]:
        motion = self.clips[clip]
        f = int(np.clip(frame, 0, motion.length - 1))
        return motion.qpos[f], motion.qvel[f]

    def bin_of(self, clip: int, frame: int) -> int:
        local = min(frame // self.fps, max(0, int(np.ceil(self.clips[clip].duration)) - 1))
        return self.bin_offsets[clip] + local


def standing_clip(profile: RobotProfile, seconds: float = 4.0, name: str = "stand") -> MotionClip:
    n = int(seconds * 50)
    nv = 6 + profile.n_dof
    qpos = np.repeat(profile.qpos0[None, :], n, axis=0)
    qvel = np.zeros((n, nv), dtype=np.float64)
    return MotionClip(name, qpos, qvel)


def sine_clip(profile: RobotProfile, seconds: float = 4.0, amp: float = 0.15) -> MotionClip:
    """A small in-place joint wave around the standing pose, for smoke training."""
    n = int(seconds * 50)
    t = np.arange(n) / 50.0
    qpos = np.repeat(profile.qpos0[None, :], n, axis=0)
    qvel = np.zeros((n, 6 + profile.n_dof), dtype=np.float64)
    phase = 2 * np.pi * 0.5 * t
    for j, name in enumerate(profile.joint_names):
        if "knee" in name or "hip_pitch" in name or "elbow" in name:
            sign = 1.0 if "left" in name or name.startswith("aa") else 1.0
            delta = amp * np.sin(phase) * (0.5 if "elbow" in name else 1.0) * sign
            qpos[:, 7 + j] = profile.qpos0[7 + j] + delta
            qvel[:, 6 + j] = np.gradient(qpos[:, 7 + j], 1 / 50.0)
    return MotionClip("sine", qpos.astype(np.float64), qvel.astype(np.float64))


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """q0, q1 wxyz. u shape [T]."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    u = u[:, None]
    w0 = np.where(sin < 1e-6, 1 - u, np.sin((1 - u) * theta) / np.clip(sin, 1e-6, None))
    w1 = np.where(sin < 1e-6, u, np.sin(u * theta) / np.clip(sin, 1e-6, None))
    out = w0 * q0 + w1 * q1
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float = 50.0) -> np.ndarray:
    n = qpos.shape[0]
    duration = (n - 1) / src_fps
    m = max(2, int(round(duration * dst_fps)) + 1)
    src_t = np.arange(n) / src_fps
    dst_t = np.linspace(0, src_t[-1], m)
    out = np.zeros((m, qpos.shape[1]), dtype=np.float64)
    out[:, :3] = np.vstack([np.interp(dst_t, src_t, qpos[:, i]) for i in range(3)]).T
    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    u = (dst_t - src_t[idx]) / (src_t[idx + 1] - src_t[idx])
    out[:, 3:7] = _slerp(qpos[idx, 3:7], qpos[idx + 1, 3:7], u)
    for j in range(7, qpos.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, qpos[:, j])
    return out


def qvel_from_qpos(qpos: np.ndarray, fps: float = 50.0) -> np.ndarray:
    dt = 1.0 / fps
    n_dof = qpos.shape[1] - 7
    qvel = np.zeros((qpos.shape[0], 6 + n_dof), dtype=np.float64)
    qvel[:, :3] = np.gradient(qpos[:, :3], dt, axis=0)
    q0 = qpos[:-1, 3:7]
    q1 = qpos[1:, 3:7].copy()
    q1[np.sum(q0 * q1, axis=1) < 0] *= -1
    w0, x0, y0, z0 = q0.T
    w1, x1, y1, z1 = q1.T
    w = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
    x = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
    y = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
    z = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
    v = np.stack((x, y, z), axis=1)
    speed = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(speed, np.clip(w, -1.0, 1.0))
    axis = v / np.clip(speed[:, None], 1e-8, None)
    body = axis * (angle / dt)[:, None]
    body[speed < 1e-8] = 0.0
    qvel[:-1, 3:6] = body
    qvel[-1, 3:6] = qvel[-2, 3:6]
    qvel[:, 6:] = np.gradient(qpos[:, 7:], dt, axis=0)
    return qvel


def load_motion_file(path: Path, n_dof: int, src_fps: float = 30.0) -> MotionClip:
    """Load one G1 clip. CSV is 30 Hz xyzw (LAFAN / Unitree). NPZ is already 50 Hz wxyz qpos."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.shape[1] < 7 + n_dof:
            raise ValueError(f"{path} qpos width {qpos.shape[1]}, need {7 + n_dof}")
        qpos = qpos[:, : 7 + n_dof]
        if "qvel" in data and data["qvel"].shape[0] == qpos.shape[0]:
            qvel = np.asarray(data["qvel"], dtype=np.float64)
        else:
            qvel = qvel_from_qpos(qpos, 50.0)
        name = str(data["name"]) if "name" in data else path.stem
        return MotionClip(name, qpos, qvel)
    return load_lafan_csv(path, n_dof, src_fps=src_fps)


def load_lafan_csv(path: Path, n_dof: int, src_fps: float = 30.0) -> MotionClip:
    rows = np.loadtxt(path, delimiter=",")
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] < 7 + n_dof:
        raise ValueError(f"{path} has {rows.shape[1]} columns, need {7 + n_dof}")
    # Unitree / LAFAN retargets store xyzw.
    pos = rows[:, 0:3]
    quat_xyzw = rows[:, 3:7]
    quat = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=1)
    joints = rows[:, 7 : 7 + n_dof]
    qpos = np.concatenate([pos, quat, joints], axis=1)
    qpos = resample_qpos(qpos, src_fps, 50.0)
    qvel = qvel_from_qpos(qpos, 50.0)
    return MotionClip(path.stem, qpos, qvel)


def load_directory(path: Path, n_dof: int) -> MotionLibrary:
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(path)
    return MotionLibrary([load_lafan_csv(f, n_dof) for f in files])


def load_npz_directory(path: Path, n_dof: int) -> MotionLibrary:
    files = sorted(Path(path).glob("*.npz"))
    if not files:
        raise FileNotFoundError(path)
    clips = []
    for file in files:
        data = np.load(file)
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64)
        if qpos.shape[1] != 7 + n_dof:
            raise ValueError(f"{file} has {qpos.shape[1]} qpos columns, expected {7 + n_dof}")
        name = str(data["name"]) if "name" in data else file.stem
        clips.append(MotionClip(name, qpos, qvel))
    return MotionLibrary(clips)
