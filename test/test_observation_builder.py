from kinova_gen3_lite_rl_bridge.observation_builder import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    build_observation,
    extract_arm_state,
    OBS_DIM,
)
import numpy as np
import pytest


def test_extract_arm_state_picks_out_arm_joints_by_name():
    # JointState carries gripper joints too, in an arbitrary order relative
    # to ARM_JOINT_NAMES — extraction must be name-indexed, not positional.
    names = [
        'right_finger_bottom', 'joint_3', 'joint_1', 'joint_2', 'joint_6', 'joint_5', 'joint_4',
    ]
    positions = [9.0, 3.0, 1.0, 2.0, 6.0, 5.0, 4.0]
    velocities = [90.0, 30.0, 10.0, 20.0, 60.0, 50.0, 40.0]

    pos, vel = extract_arm_state(names, positions, velocities)

    assert pos.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert vel.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_extract_arm_state_raises_on_missing_joint():
    names = ['joint_1', 'joint_2']
    with pytest.raises(KeyError):
        extract_arm_state(names, [0.0, 0.0], [0.0, 0.0])


def test_build_observation_shape_and_order():
    joint_pos = np.arange(6, dtype=np.float32)
    joint_vel = np.arange(6, dtype=np.float32) * 0.1
    target_xyz = np.array([1.0, 2.0, 3.0])
    ee_pos = np.array([0.5, 0.5, 0.5])
    last_action = np.full(ACTION_DIM, 0.7, dtype=np.float32)

    obs = build_observation(joint_pos, joint_vel, target_xyz, ee_pos, last_action)

    assert obs.shape == (OBS_DIM,)
    assert len(ARM_JOINT_NAMES) == 6
    np.testing.assert_allclose(obs[0:6], joint_pos)
    np.testing.assert_allclose(obs[6:12], joint_vel)
    # ee_to_target = target - ee, per the observation spec.
    np.testing.assert_allclose(obs[12:15], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(obs[15:21], last_action)
