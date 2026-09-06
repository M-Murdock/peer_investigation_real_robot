from kinova_gen3_lite_rl_bridge.wrist_jog import buttons_to_wrist_delta
import pytest


def test_lb_held_rotates_ccw_positive():
    buttons = [0, 0, 0, 0, 1, 0, 0, 0]  # LB=index 4

    delta = buttons_to_wrist_delta(buttons, jog_scale_rad_s=0.3, control_dt=0.1)

    assert delta == pytest.approx(0.03)


def test_rb_held_rotates_cw_negative():
    buttons = [0, 0, 0, 0, 0, 1, 0, 0]  # RB=index 5

    delta = buttons_to_wrist_delta(buttons, jog_scale_rad_s=0.3, control_dt=0.1)

    assert delta == pytest.approx(-0.03)


def test_neither_held_is_zero():
    buttons = [0, 0, 0, 0, 0, 0, 0, 0]

    delta = buttons_to_wrist_delta(buttons, jog_scale_rad_s=0.3, control_dt=0.1)

    assert delta == 0.0


def test_both_held_cancels_to_zero():
    buttons = [0, 0, 0, 0, 1, 1, 0, 0]

    delta = buttons_to_wrist_delta(buttons, jog_scale_rad_s=0.3, control_dt=0.1)

    assert delta == 0.0


def test_scales_by_jog_scale_and_control_dt():
    buttons = [0, 0, 0, 0, 1, 0, 0, 0]

    delta = buttons_to_wrist_delta(buttons, jog_scale_rad_s=0.6, control_dt=0.05)

    assert delta == pytest.approx(0.03)


def test_short_buttons_array_treated_as_not_held():
    delta = buttons_to_wrist_delta([], jog_scale_rad_s=0.3, control_dt=0.1)

    assert delta == 0.0
