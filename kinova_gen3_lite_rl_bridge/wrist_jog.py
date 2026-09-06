"""
Continuous wrist-rotation (joint_6) jog from held LB/RB buttons.

The gripper is rigidly mounted to end_effector_link (see
docs/ros_reference_model.md's kinematic tree) — "rotating the gripper"
means rotating joint_6, the wrist joint closest to it. This is the first
rotational axis anywhere in this system (observation_builder,
joy_cartesian_jog, and the potential field are all position-only), added
as a small, orthogonal augmentation rather than a general orientation
controller: LB/RB directly jog joint_6 while held, independent of
jog_mode (works alongside either joint or cartesian jogging — translate
with the sticks, rotate the wrist with the shoulder buttons at the same
time). Holding both cancels to zero rather than picking one arbitrarily.
"""

from typing import Sequence

BUTTONS = {
    'A': 0,
    'B': 1,
    'X': 2,
    'Y': 3,
    'LB': 4,
    'RB': 5,
    'back': 6,
    'start': 7,
}

ROTATE_CCW_BUTTON = 'LB'
ROTATE_CW_BUTTON = 'RB'
JOINT6_INDEX = 5


def buttons_to_wrist_delta(
    joy_buttons: Sequence[int], jog_scale_rad_s: float, control_dt: float
) -> float:
    """Return this cycle's joint_6 delta (radians) from held LB/RB."""
    def held(name: str) -> bool:
        idx = BUTTONS[name]
        return bool(joy_buttons[idx]) if idx < len(joy_buttons) else False

    direction = 0.0
    if held(ROTATE_CCW_BUTTON):
        direction += 1.0
    if held(ROTATE_CW_BUTTON):
        direction -= 1.0
    return direction * jog_scale_rad_s * control_dt
