"""Unitree G1, 29 DoF. PD matches BeyondMimic / mjlab (10 Hz, damping ratio 2)."""

from supermimic.robots.actuators import g1_actuator
from supermimic.robots.profile import RAW, RobotProfile

G1_JOINTS = (
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Unitree G1 low-level DDS order matches this kinematic order.
G1_TRACKED = (
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
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def g1_profile() -> RobotProfile:
    actuators = {name: g1_actuator(name) for name in G1_JOINTS}
    return RobotProfile(
        name="g1",
        raw_xml=RAW / "g1_29dof.xml",
        joint_names=G1_JOINTS,
        sdk_indices=tuple(range(29)),
        sdk_width=29,
        tracked_bodies=G1_TRACKED,
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
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
            "left_elbow_joint": 1.28,
            "right_elbow_joint": 1.28,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
        },
        actuators=actuators,
        pd_note="BeyondMimic formula on Unitree 5020/7520/4010 reflected inertia.",
    )
