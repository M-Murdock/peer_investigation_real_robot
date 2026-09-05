"""
Build the RL policy's 21-dim observation vector from real-robot state.

Pure functions, no rclpy dependency, so this is unit-testable without ROS
and reusable as-is by a future multi-goal (shared-autonomy) controller.

Order is fixed by the trained checkpoint's observation space (see
src/kinova_mjlab_reaching/tasks/reaching/reach_env_cfg.py) and must not be
changed: joint_pos_rel(6), joint_vel_rel(6), ee_to_target(3), last_action(6).
"""

from typing import Sequence

import numpy as np

ARM_JOINT_NAMES = ('joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6')
OBS_DIM = 21
ACTION_DIM = 6


def extract_arm_state(
    joint_state_names: Sequence[str],
    joint_state_positions: Sequence[float],
    joint_state_velocities: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick out the 6 arm joints, by name, in ARM_JOINT_NAMES order.

    sensor_msgs/JointState carries the gripper joints too and does not
    guarantee any particular ordering, so this must index by name, never
    by position.

    Raise KeyError if any arm joint name is missing from the message.
    """
    index = {name: i for i, name in enumerate(joint_state_names)}
    positions = np.array([joint_state_positions[index[name]] for name in ARM_JOINT_NAMES])
    velocities = np.array([joint_state_velocities[index[name]] for name in ARM_JOINT_NAMES])
    return positions, velocities


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    target_xyz_base: np.ndarray,
    ee_pos_base: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """
    Assemble the 21-dim observation the trained policy expects.

    ee_to_target is the vector from the end effector to the target,
    expressed in the robot base frame. In sim this is rotated by the
    base's orientation (quat_apply(quat_inv(base_quat_w), ...)); the real
    robot's base is fixed, so that rotation is the identity and this
    reduces to plain subtraction.
    """
    ee_to_target = np.asarray(target_xyz_base) - np.asarray(ee_pos_base)
    obs = np.concatenate(
        [
            np.asarray(joint_pos, dtype=np.float32),
            np.asarray(joint_vel, dtype=np.float32),
            ee_to_target.astype(np.float32),
            np.asarray(last_action, dtype=np.float32),
        ]
    )
    assert obs.shape == (OBS_DIM,), f'expected obs shape ({OBS_DIM},), got {obs.shape}'
    return obs
