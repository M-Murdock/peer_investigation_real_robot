"""
Map a raw sensor_msgs/Joy reading to a Cartesian (base-frame) jog delta.

Sibling to joy_joint_jog.py, not a modification of it — that module stays
available via shared_autonomy_node's `jog_mode` parameter. Reuses
gen3-lite-ros2/joy_teleop.py's AXES index layout and translate_x/y/z stick
mapping exactly (left_stick_v -> X, left_stick_h -> Y, right_stick_v -> Z),
so it feels identical to that tool's TRANSLATE mode. right_stick_h stays
unused, matching this whole project's position-only scope (no orientation
control anywhere yet). Unlike joy_teleop.py, this stays in the base frame
(no EEF-frame rotation) — matches every other Cartesian quantity already
computed in this codebase (ee_to_target, the potential field's error), and
the resulting delta is converted to a joint delta via
potential_field_client.compute_jog_delta(), not published directly to a
Cartesian velocity controller.
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

# (axis_name, sign) per Cartesian DOF — matches joy_teleop.py's AXIS_MAP
# translate_x/y/z entries exactly.
AXIS_MAP = {
    'x': ('left_stick_v', +1),
    'y': ('left_stick_h', +1),
    'z': ('right_stick_v', +1),
}


def axes_to_cartesian_delta(
    joy_axes: Sequence[float], jog_scale_m_s: float, control_dt: float
) -> np.ndarray:
    """Convert one Joy message's axes into a per-cycle Cartesian [x, y, z] delta."""
    step = jog_scale_m_s * control_dt
    delta = np.zeros(3, dtype=np.float32)
    for i, dof in enumerate(('x', 'y', 'z')):
        axis_name, sign = AXIS_MAP[dof]
        axis_idx = AXES[axis_name]
        stick_value = joy_axes[axis_idx] if axis_idx < len(joy_axes) else 0.0
        delta[i] = sign * stick_value * step
    return delta
