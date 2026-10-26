"""BeyondMimic PD: kp = armature * (2 pi * 10 Hz)^2, damping ratio 2, action scale 0.25 * effort / kp."""

from __future__ import annotations

import math
from dataclasses import dataclass

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


def pd_from_armature(armature: float, effort: float) -> tuple[float, float, float]:
    kp = armature * NATURAL_FREQ**2
    kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
    scale = 0.25 * effort / kp
    return kp, kd, scale


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    armature: float
    effort: float
    kp: float
    kd: float
    action_scale: float


def spec_from_armature(name: str, armature: float, effort: float) -> ActuatorSpec:
    kp, kd, scale = pd_from_armature(armature, effort)
    return ActuatorSpec(name, armature, effort, kp, kd, scale)


def spec_from_gains(name: str, armature: float, effort: float, kp: float, kd: float) -> ActuatorSpec:
    """Vendor gains. Action scale still follows 0.25 * effort / kp."""
    return ActuatorSpec(name, armature, effort, kp, kd, 0.25 * effort / kp)


_A5020 = reflected_inertia_from_two_stage_planetary(
    (0.139e-4, 0.017e-4, 0.169e-4),
    (1.0, 1.0 + 46.0 / 18.0, 1.0 + 56.0 / 16.0),
)
_A7520_14 = reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.098e-4, 0.533e-4),
    (1.0, 4.5, 1.0 + 48.0 / 22.0),
)
_A7520_22 = reflected_inertia_from_two_stage_planetary(
    (0.489e-4, 0.109e-4, 0.738e-4),
    (1.0, 4.5, 5.0),
)
_A4010 = reflected_inertia_from_two_stage_planetary(
    (0.068e-4, 0.0, 0.0),
    (1.0, 5.0, 5.0),
)

G1_5020 = spec_from_armature("5020", _A5020, 25.0)
G1_7520_14 = spec_from_armature("7520_14", _A7520_14, 88.0)
G1_7520_22 = spec_from_armature("7520_22", _A7520_22, 139.0)
G1_4010 = spec_from_armature("4010", _A4010, 5.0)
# Waist pitch/roll and ankles: two 5020s, nominal 1:1.
G1_5020_X2 = spec_from_armature("5020x2", _A5020 * 2, 25.0 * 2)


def g1_actuator(joint: str) -> ActuatorSpec:
    if joint.endswith("_wrist_pitch_joint") or joint.endswith("_wrist_yaw_joint"):
        return G1_4010
    if "ankle_" in joint or joint in ("waist_pitch_joint", "waist_roll_joint"):
        return G1_5020_X2
    if joint.endswith("_hip_roll_joint") or joint.endswith("_knee_joint"):
        return G1_7520_22
    if joint.endswith("_hip_pitch_joint") or joint.endswith("_hip_yaw_joint") or joint == "waist_yaw_joint":
        return G1_7520_14
    return G1_5020
