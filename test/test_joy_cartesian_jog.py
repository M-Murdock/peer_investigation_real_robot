from kinova_gen3_lite_rl_bridge.joy_cartesian_jog import axes_to_cartesian_delta
import numpy as np


def test_left_stick_vertical_maps_to_x():
    # AXES: left_stick_h=0, left_stick_v=1, LT=2, right_stick_h=3, right_stick_v=4, RT=5
    axes = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.15, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.015, 0.0, 0.0], atol=1e-6)


def test_left_stick_horizontal_maps_to_y():
    axes = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.15, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.0, 0.015, 0.0], atol=1e-6)


def test_right_stick_vertical_maps_to_z():
    axes = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.15, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.0, 0.0, 0.015], atol=1e-6)


def test_right_stick_horizontal_is_unused():
    axes = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.15, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.0, 0.0, 0.0])


def test_scales_by_jog_scale_and_control_dt():
    axes = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.3, control_dt=0.05)

    np.testing.assert_allclose(delta, [0.015, 0.0, 0.0], atol=1e-6)


def test_short_axes_array_defaults_missing_axes_to_zero():
    axes = [1.0, 1.0]  # right_stick_v (index 4) out of range

    delta = axes_to_cartesian_delta(axes, jog_scale_m_s=0.15, control_dt=0.1)

    np.testing.assert_allclose(delta, [0.015, 0.015, 0.0], atol=1e-6)
