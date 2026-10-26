"""Table A.1 proprioception and the K=10 local reference goal."""

from __future__ import annotations

import torch

from supermimic.mathutil import projected_gravity, quat_to_rot, rot6d_of, rotz, yaw_of_quat


def split_state(state: torch.Tensor, n_dof: int):
    nq = 7 + n_dof
    qpos = state[..., :nq]
    qvel = state[..., nq:]
    return qpos, qvel


def observation(state: torch.Tensor, last_action: torch.Tensor, bias: torch.Tensor, q0: torch.Tensor, n_dof: int) -> torch.Tensor:
    qpos, qvel = split_state(state, n_dof)
    q = qpos[..., 7:]
    qd = qvel[..., 6:]
    rot = quat_to_rot(qpos[..., 3:7])
    gyro = qvel[..., 3:6]
    grav = projected_gravity(rot)
    return torch.cat((q - q0 + bias, qd, gyro, grav, last_action), dim=-1)


def canonical(state: torch.Tensor, n_dof: int) -> torch.Tensor:
    """Heading-invariant features. Dimension equals the raw state dimension."""
    qpos, qvel = split_state(state, n_dof)
    pos = qpos[..., :3]
    quat = qpos[..., 3:7]
    yaw = yaw_of_quat(quat)
    rot = quat_to_rot(quat)
    local = rotz(-yaw) @ rot
    v_h = torch.matmul(rotz(-yaw), qvel[..., :3].unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (
            pos[..., 2:3],
            rot6d_of(local),
            v_h,
            qvel[..., 3:6],
            qpos[..., 7:],
            qvel[..., 6:],
        ),
        dim=-1,
    )


def integrate_delta(state: torch.Tensor, delta: torch.Tensor, n_dof: int) -> torch.Tensor:
    """Apply a heading-frame residual. A zero residual returns the same state."""
    from supermimic.mathutil import quat_mul, quat_normalize, rotvec_to_quat

    qpos, qvel = split_state(state, n_dof)
    dpos, drot, dv, dw, dq, dqd = torch.split(delta, (3, 3, 3, 3, n_dof, n_dof), dim=-1)
    pos = qpos[..., :3]
    quat = qpos[..., 3:7]
    yaw = yaw_of_quat(quat)
    ry = rotz(yaw)
    pos = pos + torch.matmul(ry, dpos.unsqueeze(-1)).squeeze(-1)
    quat = quat_normalize(quat_mul(quat, rotvec_to_quat(drot)))
    v_h = torch.matmul(rotz(-yaw), qvel[..., :3].unsqueeze(-1)).squeeze(-1) + dv
    v_w = torch.matmul(ry, v_h.unsqueeze(-1)).squeeze(-1)
    w = qvel[..., 3:6] + dw
    q = qpos[..., 7:] + dq
    qd = qvel[..., 6:] + dqd
    return torch.cat((pos, quat, q, v_w, w, qd), dim=-1)


def state_error(pred: torch.Tensor, target: torch.Tensor, n_dof: int, sigma: dict[str, float]) -> torch.Tensor:
    """Per-dimension normalized error, expressed in the target's heading frame."""
    pq, pv = split_state(pred, n_dof)
    tq, tv = split_state(target, n_dof)
    yaw = yaw_of_quat(tq[..., 3:7])
    ry_t = rotz(-yaw)
    pos_err = torch.matmul(ry_t, (pq[..., :3] - tq[..., :3]).unsqueeze(-1)).squeeze(-1)
    r_err = rot_log_target(tq[..., 3:7], pq[..., 3:7])
    v_err = torch.matmul(ry_t, (pv[..., :3] - tv[..., :3]).unsqueeze(-1)).squeeze(-1)
    w_err = pv[..., 3:6] - tv[..., 3:6]
    q_err = pq[..., 7:] - tq[..., 7:]
    qd_err = pv[..., 6:] - tv[..., 6:]
    parts = (
        pos_err / sigma["pos"],
        r_err / sigma["rot"],
        v_err / sigma["lin"],
        w_err / sigma["ang"],
        q_err / sigma["q"],
        qd_err / sigma["qd"],
    )
    return torch.cat(parts, dim=-1)


def rot_log_target(q_target: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
    from supermimic.mathutil import rot_log

    rt = quat_to_rot(q_target)
    rp = quat_to_rot(q_pred)
    return rot_log(rt.transpose(-1, -2) @ rp)


def goal_frame(qpos: torch.Tensor, qvel: torch.Tensor, q0: torch.Tensor, n_dof: int) -> torch.Tensor:
    quat = qpos[..., 3:7]
    rot = quat_to_rot(quat)
    yaw = yaw_of_quat(quat)
    v_h = torch.matmul(rotz(-yaw), qvel[..., :3].unsqueeze(-1)).squeeze(-1)
    return torch.cat(
        (
            qpos[..., 7:] - q0,
            qvel[..., 6:],
            projected_gravity(rot),
            qvel[..., 3:6],
            v_h,
            qpos[..., 2:3],
        ),
        dim=-1,
    )


def root_error(state: torch.Tensor, ref_qpos: torch.Tensor, n_dof: int) -> torch.Tensor:
    """Privileged critic feature: root position error in the reference heading, plus yaw error."""
    qpos, _ = split_state(state, n_dof)
    yaw_r = yaw_of_quat(ref_qpos[..., 3:7])
    pos_err = torch.matmul(rotz(-yaw_r), (qpos[..., :3] - ref_qpos[..., :3]).unsqueeze(-1)).squeeze(-1)
    yaw_err = yaw_of_quat(qpos[..., 3:7]) - yaw_r
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
    return torch.cat((pos_err, yaw_err.unsqueeze(-1)), dim=-1)
