"""Robot profile: kinematics, PD, tracked bodies, and the SDK joint permutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from supermimic.robots.actuators import ActuatorSpec
from supermimic.robots.scene import prepare_scene

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "assets" / "raw"
SCENES = ROOT / "assets" / "scenes"


@dataclass
class RobotProfile:
    name: str
    raw_xml: Path
    joint_names: tuple[str, ...]
    sdk_indices: tuple[int, ...]
    sdk_width: int
    tracked_bodies: tuple[str, ...]
    anchor_body: str
    termination_body: str
    torso_body: str
    foot_bodies: tuple[str, str]
    default_joint: dict[str, float]
    actuators: dict[str, ActuatorSpec]
    pd_note: str
    vel_x: tuple[float, float] = (-1.5, 2.0)
    vel_y: tuple[float, float] = (-1.32, 1.32)
    vel_yaw: tuple[float, float] = (-3.35, 3.35)
    # Filled by load_robot.
    scene_xml: Path = field(default=Path("."))
    standing_height: float = 0.0
    length_scale: float = 1.0
    fall_height: float = 0.35
    z_terminate: float = 0.25
    pos_sigma: float = 0.3
    qpos0: np.ndarray | None = None
    action_scale: np.ndarray | None = None
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    effort: np.ndarray | None = None
    q_min: np.ndarray | None = None
    q_max: np.ndarray | None = None

    @property
    def n_dof(self) -> int:
        return len(self.joint_names)

    @property
    def obs_dim(self) -> int:
        return 3 * self.n_dof + 6

    @property
    def goal_frame_dim(self) -> int:
        return 2 * self.n_dof + 10

    @property
    def xi_dim(self) -> int:
        return 1 + self.n_dof + 3


def _foot_bottom(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> float:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise KeyError(geom_name)
    size_z = float(model.geom_size[gid, 2] if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX else model.geom_rbound[gid])
    return float(data.geom_xpos[gid, 2] - size_z)


def load_model(profile: RobotProfile) -> mujoco.MjModel:
    out = SCENES / f"{profile.name}.xml"
    prepare_scene(profile.raw_xml, out, profile.actuators, profile.foot_bodies)
    profile.scene_xml = out
    model = mujoco.MjModel.from_xml_path(str(out))
    hinge = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        hinge.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    if tuple(hinge) != profile.joint_names:
        raise RuntimeError(f"{profile.name} joint order {hinge} != {profile.joint_names}")
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[3] = 1.0
    for name, value in profile.default_joint.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[model.jnt_qposadr[jid]] = value
    # Start from the XML free-joint default height, then plant the feet.
    qpos[2] = float(model.qpos0[2]) if model.nq else 0.8
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    bottom = min(_foot_bottom(model, data, "left_foot_collision"), _foot_bottom(model, data, "right_foot_collision"))
    qpos[2] -= bottom
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.termination_body)
    profile.standing_height = float(data.xpos[root, 2])
    profile.qpos0 = qpos
    profile.action_scale = np.array([profile.actuators[n].action_scale for n in profile.joint_names], dtype=np.float64)
    profile.kp = np.array([profile.actuators[n].kp for n in profile.joint_names], dtype=np.float64)
    profile.kd = np.array([profile.actuators[n].kd for n in profile.joint_names], dtype=np.float64)
    profile.effort = np.array([profile.actuators[n].effort for n in profile.joint_names], dtype=np.float64)
    qmin = []
    qmax = []
    for name in profile.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            qmin.append(float(model.jnt_range[jid, 0]))
            qmax.append(float(model.jnt_range[jid, 1]))
        else:
            qmin.append(-np.inf)
            qmax.append(np.inf)
    profile.q_min = np.array(qmin)
    profile.q_max = np.array(qmax)
    # Leave model.qpos0 at the XML zero. This MuJoCo build treats joint angles as
    # offsets from qpos0, so overwriting it would move the kinematic tree.
    return model


def apply_length_scales(robots: dict[str, RobotProfile]) -> None:
    g1_h = robots["g1"].standing_height
    if g1_h <= 0:
        raise RuntimeError("load G1 before scaling")
    for robot in robots.values():
        scale = 1.0 if robot.name == "g1" else robot.standing_height / g1_h
        robot.length_scale = scale
        robot.z_terminate = 0.25 * scale
        robot.pos_sigma = 0.3 * scale
        robot.fall_height = 0.35 * scale
