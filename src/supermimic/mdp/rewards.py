"""BeyondMimic tracking reward (Table A.2) without the self-collision term.

Body positions and orientations are compared after re-anchoring the reference in
xy and yaw onto the robot anchor. Root terms stay in the world frame. Every term
is multiplied by the control period.
"""

from __future__ import annotations

import torch

from supermimic.mathutil import rot_log, yaw_from_rot


def reanchor(robot_pos, robot_rot, ref_pos, ref_rot, anchor_index: int):
    """Align reference xy/yaw to the robot anchor. Heights, roll, and pitch stay."""
    r_pos = robot_pos[:, anchor_index]
    r_rot = robot_rot[:, anchor_index]
    f_pos = ref_pos[:, anchor_index]
    f_rot = ref_rot[:, anchor_index]
    dyaw = yaw_from_rot(r_rot) - yaw_from_rot(f_rot)
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    # 2d rotation of reference points into the robot yaw, then translate.
    rel = ref_pos[..., :2] - f_pos[..., :2].unsqueeze(1)
    x = c.unsqueeze(-1) * rel[..., 0] - s.unsqueeze(-1) * rel[..., 1]
    y = s.unsqueeze(-1) * rel[..., 0] + c.unsqueeze(-1) * rel[..., 1]
    # z, roll, and pitch stay on the reference. Only xy and yaw move onto the robot.
    out_pos = torch.stack((x + r_pos[..., 0:1], y + r_pos[..., 1:2], ref_pos[..., 2]), dim=-1)
    # Rotate reference orientations by the same yaw.
    row0 = torch.stack((c, -s, torch.zeros_like(c)), dim=-1)
    row1 = torch.stack((s, c, torch.zeros_like(c)), dim=-1)
    row2 = torch.stack((torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)), dim=-1)
    r_yaw = torch.stack((row0, row1, row2), dim=-2)
    out_rot = r_yaw.unsqueeze(1) @ ref_rot
    return out_pos, out_rot


def tracking_reward(
    robot_pos: torch.Tensor,
    robot_rot: torch.Tensor,
    robot_lin: torch.Tensor,
    robot_ang: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_rot: torch.Tensor,
    ref_lin: torch.Tensor,
    ref_ang: torch.Tensor,
    anchor_index: int,
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    dt: float,
    pos_sigma: float,
) -> torch.Tensor:
    des_pos, des_rot = reanchor(robot_pos, robot_rot, ref_pos, ref_rot, anchor_index)
    pos_err = ((des_pos - robot_pos) ** 2).sum(-1).mean(-1)
    ori = rot_log(des_rot.transpose(-1, -2) @ robot_rot)
    ori_err = (ori**2).sum(-1).mean(-1)
    lin_err = ((ref_lin - robot_lin) ** 2).sum(-1).mean(-1)
    ang_err = ((ref_ang - robot_ang) ** 2).sum(-1).mean(-1)
    root_pos = ((ref_pos[:, anchor_index] - robot_pos[:, anchor_index]) ** 2).sum(-1)
    root_ori = rot_log(ref_rot[:, anchor_index].transpose(-1, -2) @ robot_rot[:, anchor_index])
    root_ori_err = (root_ori**2).sum(-1)
    r_pos = torch.exp(-pos_err / pos_sigma**2)
    r_ori = torch.exp(-ori_err / 0.4**2)
    r_lin = torch.exp(-lin_err / 1.0**2)
    r_ang = torch.exp(-ang_err / 3.14**2)
    r_root_p = torch.exp(-root_pos / pos_sigma**2)
    r_root_o = torch.exp(-root_ori_err / 0.4**2)
    smooth = ((action - prev_action) ** 2).sum(-1)
    over = torch.relu(q - q_max) + torch.relu(q_min - q)
    limit = over.sum(-1)
    total = r_pos + r_ori + r_lin + r_ang + 0.5 * r_root_p + 0.5 * r_root_o - 0.1 * smooth - 2.0 * limit
    return total * dt


def velocity_reward(
    lin_vel_heading: torch.Tensor,
    yaw_rate: torch.Tensor,
    command: torch.Tensor,
    torso_roll: torch.Tensor,
    torso_pitch_rate: torch.Tensor,
    torso_roll_rate: torch.Tensor,
    q: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Table A.6. lin_vel_heading is xy in the yaw-only frame. yaw_rate is world yaw rate."""
    v_err = ((lin_vel_heading - command[..., :2]) ** 2).sum(-1)
    w_err = (yaw_rate - command[..., 2]) ** 2
    r_v = torch.exp(-v_err / 0.5**2)
    r_w = torch.exp(-w_err / 0.5)
    smooth = ((action - prev_action) ** 2).sum(-1)
    limit = (torch.relu(q - q_max) + torch.relu(q_min - q)).sum(-1)
    lean = torch.sin(torso_roll) ** 2
    rate = torso_roll_rate**2 + torso_pitch_rate**2
    total = 2 * r_v + 2 * r_w - 0.3 * smooth - 2 * limit - 4 * lean - 0.05 * rate
    return total * dt
