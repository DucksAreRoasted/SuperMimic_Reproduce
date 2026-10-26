"""Unitree R1 as published in unitree_rl_mjlab: 24 actuated joints.

The EDU machine has a 2-DoF head. That head is not in the public RL MJCF, so the
profile follows the model Unitree ships for training (12 leg + 2 waist + 10 arm).
Gains are the published unitree_rl_mjlab values. The armature there is a placeholder
0.01, so these kp/kd are vendor gains, not the BeyondMimic armature formula.
Action scale is still 0.25 * effort / kp.
"""

from supermimic.robots.actuators import ActuatorSpec, spec_from_gains
from supermimic.robots.profile import RAW, RobotProfile

R1_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

# Indices into the 32-slot Unitree R1 low-level array (empty slots stay zero).
R1_SDK = (
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13,
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
)

R1_TRACKED = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
)


def _spec(name: str) -> ActuatorSpec:
    if any(k in name for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "waist_")):
        return spec_from_gains(name, 0.01, 60.0, 100.0, 2.0)
    if "ankle_" in name:
        return spec_from_gains(name, 0.01, 50.0, 40.0, 2.0)
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return spec_from_gains(name, 0.01, 60.0, 40.0, 2.0)
    return spec_from_gains(name, 0.01, 33.0, 20.0, 1.0)


def r1_profile() -> RobotProfile:
    return RobotProfile(
        name="r1",
        raw_xml=RAW / "r1_mjlab.xml",
        joint_names=R1_JOINTS,
        sdk_indices=R1_SDK,
        sdk_width=32,
        tracked_bodies=R1_TRACKED,
        anchor_body="torso_link",
        termination_body="pelvis",
        torso_body="torso_link",
        foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
        default_joint={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "right_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": 0.35,
            "left_elbow_joint": 0.87,
            "right_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        },
        actuators={name: _spec(name) for name in R1_JOINTS},
        pd_note="unitree_rl_mjlab vendor kp/kd. Armature in that stack is a placeholder, so the BeyondMimic formula is not applied.",
    )
