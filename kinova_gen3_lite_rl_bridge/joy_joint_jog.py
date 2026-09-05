"""
Map a raw sensor_msgs/Joy reading straight to a per-joint jog delta.

Reuses gen3-lite-ros2/joy_teleop.py's AXES index layout (same physical
Xbox-style controller), but maps sticks directly to joints instead of a
Cartesian twist: shared_autonomy_node publishes to
joint_trajectory_controller, not twist_controller, so there is no
Cartesian-to-joint conversion step in this pipeline at all — a stick
maps straight to a joint delta.

Only the 4 stick axes are used (LT/RT are deliberately left unmapped):
trigger axes are single-ended on most controllers (0..1, not -1..+1), so
they are not well-suited to bidirectional jogging, and this project has no
hardware in front of it to verify a specific controller's trigger rest
polarity. joints 5/6 therefore get zero human input in this first version
— their motion comes entirely from the blended policy suggestion. Extend
AXIS_TO_JOINT (e.g. with a bank-toggle button, matching joy_teleop.py's
own translate/rotate toggle) if direct wrist jogging is needed later.
"""

from typing import Sequence

import numpy as np

AXES = {
    'left_stick_h': 0,
    'left_stick_v': 1,
    'right_stick_h': 3,
    'right_stick_v': 4,
    'LT': 2,
    'RT': 5,
}

JOINT_NAMES = ('joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6')

AXIS_TO_JOINT = {
    'left_stick_h': 'joint_1',
    'left_stick_v': 'joint_2',
    'right_stick_h': 'joint_3',
    'right_stick_v': 'joint_4',
}


def axes_to_joint_delta(
    joy_axes: Sequence[float], jog_scale_rad_s: float, control_dt: float
) -> np.ndarray:
    """
    Convert one Joy message's axes into a per-cycle joint-position delta.

    Returns a (6,) array in JOINT_NAMES order — zero for joint_5/joint_6
    (unmapped, see module docstring).
    """
    joint_index = {name: i for i, name in enumerate(JOINT_NAMES)}
    delta = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    for axis_name, joint_name in AXIS_TO_JOINT.items():
        axis_idx = AXES[axis_name]
        stick_value = joy_axes[axis_idx] if axis_idx < len(joy_axes) else 0.0
        delta[joint_index[joint_name]] = stick_value * jog_scale_rad_s * control_dt
    return delta
