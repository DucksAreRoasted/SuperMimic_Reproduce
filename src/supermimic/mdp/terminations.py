"""Hard fall test from Appendix A, and the velocity-task fall test from Table A.6."""

from __future__ import annotations

import torch

from supermimic.mathutil import projected_gravity, quat_to_rot


def tilt_error(robot_quat: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
    g_r = projected_gravity(quat_to_rot(robot_quat))[..., 2]
    g_f = projected_gravity(quat_to_rot(ref_quat))[..., 2]
    return (g_r - g_f).abs()


def tracking_failed(robot_z: torch.Tensor, ref_z: torch.Tensor, robot_quat: torch.Tensor, ref_quat: torch.Tensor, z_limit: float) -> torch.Tensor:
    height = (robot_z - ref_z).abs() > z_limit
    tilt = tilt_error(robot_quat, ref_quat) > 0.8
    return height | tilt


def velocity_failed(root_z: torch.Tensor, root_quat: torch.Tensor, fall_height: float) -> torch.Tensor:
    grav_z = projected_gravity(quat_to_rot(root_quat))[..., 2]
    # 1 + u_z > 0.7, with u_z the z component of the unit gravity direction.
    # projected gravity is already R^T [0,0,-1], so u_z = grav_z.
    return (root_z < fall_height) | (1.0 + grav_z > 0.7)
