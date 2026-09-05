from kinova_gen3_lite_rl_bridge.joy_joint_jog import axes_to_joint_delta
import numpy as np
import pytest


def test_maps_sticks_to_first_four_joints():
    # AXES: left_stick_h=0, left_stick_v=1, LT=2, right_stick_h=3, right_stick_v=4, RT=5
    axes = [1.0, -1.0, 0.0, 0.5, -0.5, 0.0]

    delta = axes_to_joint_delta(axes, jog_scale_rad_s=1.0, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.1, -0.1, 0.05, -0.05, 0.0, 0.0], atol=1e-6)


def test_joints_five_and_six_are_always_zero():
    axes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    delta = axes_to_joint_delta(axes, jog_scale_rad_s=1.0, control_dt=1.0)

    assert delta[4] == 0.0
    assert delta[5] == 0.0


def test_scales_by_jog_scale_and_control_dt():
    axes = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    delta = axes_to_joint_delta(axes, jog_scale_rad_s=2.0, control_dt=0.05)

    assert delta[0] == pytest.approx(0.1)


def test_short_axes_array_defaults_missing_axes_to_zero():
    # Only 2 axes present — right_stick_h/v (indices 3, 4) are out of range.
    axes = [1.0, 1.0]

    delta = axes_to_joint_delta(axes, jog_scale_rad_s=1.0, control_dt=1.0)

    np.testing.assert_allclose(delta, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
