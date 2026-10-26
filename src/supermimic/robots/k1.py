"""Booster K1, 22 DoF. Armature and effort come from the official MJCF.

kp and kd use the BeyondMimic formula on that armature. SDK order is JointIndexK1
and matches the kinematic joint order in K1_22dof.xml.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from supermimic.robots.actuators import ActuatorSpec, spec_from_armature
from supermimic.robots.profile import RAW, RobotProfile

K1_JOINTS = (
    "aahead_yaw_joint",
    "aahead_pitch_joint",
    "aaleft_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

K1_TRACKED = (
    "trunk",
    "left_hip_roll_link",
    "left_knee_pitch_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_pitch_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "aahead_pitch_link",
)


def _actuators() -> dict[str, ActuatorSpec]:
    root = ET.parse(RAW / "k1.xml").getroot()
    effort: dict[str, float] = {}
    for motor in root.findall("actuator/motor"):
        lo, hi = (float(x) for x in motor.get("forcerange", "-1 1").split())
        effort[motor.get("joint") or motor.get("name")] = max(abs(lo), abs(hi))
    armature: dict[str, float] = {}
    for joint in root.iter("joint"):
        if joint.get("type", "hinge") == "free":
            continue
        armature[joint.get("name")] = float(joint.get("armature", "0.01"))
    out = {}
    for name in K1_JOINTS:
        # Armature below 1e-3 makes a 10 Hz impedance too soft to hold the limb.
        # The XML value is still the inertia; kp uses at least 0.01 so the formula
        # stays in the same regime as the Unitree stacks.
        arm = max(armature[name], 0.01)
        out[name] = spec_from_armature(name, arm, effort.get(name, 20.0))
    return out


def k1_profile() -> RobotProfile:
    return RobotProfile(
        name="k1",
        raw_xml=RAW / "k1.xml",
        joint_names=K1_JOINTS,
        sdk_indices=tuple(range(22)),
        sdk_width=22,
        tracked_bodies=K1_TRACKED,
        anchor_body="trunk",
        termination_body="trunk",
        torso_body="trunk",
        foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
        default_joint={
            "left_hip_pitch_joint": -0.2,
            "right_hip_pitch_joint": -0.2,
            "left_knee_pitch_joint": 0.4,
            "right_knee_pitch_joint": 0.4,
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
            "aaleft_shoulder_pitch_joint": 0.2,
            "aaright_shoulder_pitch_joint": 0.2,
            "left_elbow_pitch_joint": -0.4,
            "right_elbow_pitch_joint": -0.4,
        },
        actuators=_actuators(),
        pd_note="BeyondMimic formula on official XML armature (floored at 0.01) and motor forcerange.",
        vel_x=(-0.3, 0.4),
        vel_y=(-0.3, 0.3),
        vel_yaw=(-1.5, 1.5),
    )
