"""BONES-SEED name filter and the Appendix E feasibility checks.

The keyword list is SONIC's published list with the acrobatic terms removed.
Those terms would delete the flips and side rolls the paper keeps.
"""

from __future__ import annotations

SONIC_TERMS = (
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat",
    "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_", "scooter",
    "stepdown", "acrobatics_", "box_hspu", "cartwheel", "50cm_box_", "on_box",
    "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
    "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
    "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump", "safety_roll",
    "box_dips", "walking_on_edge", "push_obstacle",
)
ACROBATIC = {
    "acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll",
    "kozak_dance", "jump_to_shoulder_roll", "monkey_jump",
}
NAME_TERMS = tuple(term for term in SONIC_TERMS if term not in ACROBATIC) + (
    "_box_", "box_", "jump_off_50cm", "off_50cm", "fence", "_stair",
)


def name_rejected(filename: str, terms: tuple[str, ...] = NAME_TERMS) -> bool:
    text = filename.lower()
    return any(term in text for term in terms)


def feasibility_flags(
    body_z_min: float,
    seconds_above_020: float,
    fraction_below_008: float,
    both_feet_air_s: float,
    pelvis_upright: bool,
    pelvis_standing: bool,
    nearly_still: bool,
) -> list[str]:
    """Thresholds from Appendix E. Times are already measured by the caller."""
    reasons = []
    if seconds_above_020 >= 0.6:
        reasons.append("elevated_support")
    if body_z_min < -0.12 or fraction_below_008 > 0.8:
        reasons.append("ground_penetration")
    if both_feet_air_s >= 0.5 and pelvis_upright and pelvis_standing and nearly_still:
        reasons.append("floating_feet")
    return reasons
