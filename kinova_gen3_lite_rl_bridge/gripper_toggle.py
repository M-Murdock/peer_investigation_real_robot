"""
Edge-triggered gripper open/close toggle from a single Joy button.

Pure button-edge detection, no rclpy dependency — shared_autonomy_node
owns actually sending the GripperCommand goal (see
gen3-lite-ros2/src/joy_teleop.py's own GripperCommand action-client
pattern, reused there for LT/RT triggers and here for a plain button).
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

TOGGLE_BUTTON = 'A'

# Same values gen3-lite-ros2/src/joy_teleop.py already uses for this
# gripper, kept identical so both teleop tools behave the same way.
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSE_POSITION = 0.8
GRIPPER_MAX_EFFORT = 10.0


def toggle_button_rising_edge(joy_buttons: Sequence[int], was_pressed: bool) -> tuple[bool, bool]:
    """
    Return (rising_edge, now_pressed) for the gripper toggle button.

    rising_edge is True only on the cycle the button transitions from
    released to pressed — a caller uses this to fire exactly one toggle
    per physical press, not once per cycle the button is held down.
    """
    idx = BUTTONS[TOGGLE_BUTTON]
    now_pressed = bool(joy_buttons[idx]) if idx < len(joy_buttons) else False
    rising_edge = now_pressed and not was_pressed
    return rising_edge, now_pressed
