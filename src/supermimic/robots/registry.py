from __future__ import annotations

import mujoco

from supermimic.robots.g1 import g1_profile
from supermimic.robots.k1 import k1_profile
from supermimic.robots.profile import RobotProfile, apply_length_scales, load_model
from supermimic.robots.r1 import r1_profile

_BUILDERS = {"g1": g1_profile, "r1": r1_profile, "k1": k1_profile}
_CACHE: dict[str, tuple[RobotProfile, mujoco.MjModel]] = {}


def all_robots() -> dict[str, tuple[RobotProfile, mujoco.MjModel]]:
    if len(_CACHE) == 3:
        return _CACHE
    profiles = {name: builder() for name, builder in _BUILDERS.items()}
    models = {name: load_model(profile) for name, profile in profiles.items()}
    apply_length_scales(profiles)
    for name in profiles:
        _CACHE[name] = (profiles[name], models[name])
    return _CACHE


def get_robot(name: str) -> tuple[RobotProfile, mujoco.MjModel]:
    if name not in _CACHE:
        all_robots()
    return _CACHE[name]
