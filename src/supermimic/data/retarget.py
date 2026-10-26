"""Differential-IK retarget from one robot's trajectory onto another.

Keypoints are the tracked bodies whose names are linked by ``pairs``. Positions are
scaled by the ratio of standing heights, about the source root, then tracked on the
target with damped least squares. The target root yaw and xy follow the scaled source.
"""

from __future__ import annotations

import mujoco
import numpy as np

from supermimic.data.motion import MotionClip, qvel_from_qpos
from supermimic.robots.profile import RobotProfile


def retarget_clip(
    source_model: mujoco.MjModel,
    source: RobotProfile,
    target_model: mujoco.MjModel,
    target: RobotProfile,
    clip: MotionClip,
    pairs: dict[str, str],
    iterations: int = 8,
    damping: float = 1e-3,
) -> MotionClip:
    scale = target.standing_height / source.standing_height
    src = mujoco.MjData(source_model)
    dst = mujoco.MjData(target_model)
    dst.qpos[:] = target.qpos0
    out = np.zeros((clip.length, target_model.nq), dtype=np.float64)
    src_ids = [mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, a) for a in pairs]
    dst_ids = [mujoco.mj_name2id(target_model, mujoco.mjtObj.mjOBJ_BODY, b) for b in pairs.values()]
    src_root = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, source.termination_body)
    for t in range(clip.length):
        src.qpos[:] = clip.qpos[t]
        mujoco.mj_forward(source_model, src)
        root = src.xpos[src_root].copy()
        targets = []
        for bid in src_ids:
            delta = src.xpos[bid] - root
            delta[:2] *= scale
            delta[2] *= scale
            targets.append(root * np.array([scale, scale, 1.0]) + np.array([0, 0, target.qpos0[2] - source.qpos0[2]]) + delta)
        # Place the root on the scaled xy and the target standing height plus scaled root motion.
        dst.qpos[:2] = clip.qpos[t, :2] * scale
        dst.qpos[2] = target.qpos0[2] + (clip.qpos[t, 2] - source.qpos0[2]) * scale
        dst.qpos[3:7] = clip.qpos[t, 3:7]
        for _ in range(iterations):
            mujoco.mj_forward(target_model, dst)
            err = []
            jacs = []
            for bid, goal in zip(dst_ids, targets):
                point = dst.xpos[bid]
                err.append(goal - point)
                jacp = np.zeros((3, target_model.nv))
                jacr = np.zeros((3, target_model.nv))
                mujoco.mj_jac(target_model, dst, jacp, jacr, point, bid)
                jacs.append(jacp[:, 6:])
            error = np.concatenate(err)
            jac = np.concatenate(jacs, axis=0)
            eye = np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * eye, error)
            dst.qpos[7:] = np.clip(dst.qpos[7:] + dq, target.q_min, target.q_max)
        out[t] = dst.qpos
    return MotionClip(f"{clip.name}_to_{target.name}", out, qvel_from_qpos(out))


K1_FROM_G1 = {
    "pelvis": "trunk",
    "left_knee_link": "left_knee_pitch_link",
    "right_knee_link": "right_knee_pitch_link",
    "left_elbow_link": "left_elbow_pitch_link",
    "right_elbow_link": "right_elbow_pitch_link",
    "left_wrist_yaw_link": "left_elbow_yaw_link",
    "right_wrist_yaw_link": "right_elbow_yaw_link",
}


def pairs_for(source: RobotProfile, target: RobotProfile) -> dict[str, str]:
    rename = K1_FROM_G1 if target.name == "k1" else None
    return shared_pairs(source, target, rename)


def shared_pairs(source: RobotProfile, target: RobotProfile, rename: dict[str, str] | None = None) -> dict[str, str]:
    rename = rename or {}
    pairs = {}
    target_names = set(target.tracked_bodies)
    for name in source.tracked_bodies:
        mapped = rename.get(name, name)
        if mapped in target_names:
            pairs[name] = mapped
    if len(pairs) < 4:
        raise RuntimeError(f"only {len(pairs)} shared keypoints between {source.name} and {target.name}")
    return pairs
