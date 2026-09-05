from kinova_gen3_lite_rl_bridge.safety_clamp import SafetyClamp, SafetyLimits
import numpy as np

_LIMITS = SafetyLimits(
    max_step_rad=np.full(6, 0.05),
    max_velocity_rad_s=np.full(6, 10.0),  # large — max_step_rad is the binding limit here
    joint_lower_rad=np.full(6, -1.0),
    joint_upper_rad=np.full(6, 1.0),
)


def test_small_step_passes_through_unclamped():
    clamp = SafetyClamp(_LIMITS, control_dt=0.1)
    prev_q = np.zeros(6)
    raw_target = np.full(6, 0.02)  # within max_step_rad

    result = clamp.clamp(raw_target, prev_q)

    np.testing.assert_allclose(result, raw_target)


def test_large_step_is_clamped_to_max_step_rad():
    clamp = SafetyClamp(_LIMITS, control_dt=0.1)
    prev_q = np.zeros(6)
    raw_target = np.full(6, 5.0)  # far beyond max_step_rad and joint bounds

    result = clamp.clamp(raw_target, prev_q)

    np.testing.assert_allclose(result, np.full(6, 0.05))


def test_negative_step_is_clamped_symmetrically():
    clamp = SafetyClamp(_LIMITS, control_dt=0.1)
    prev_q = np.zeros(6)
    raw_target = np.full(6, -5.0)

    result = clamp.clamp(raw_target, prev_q)

    np.testing.assert_allclose(result, np.full(6, -0.05))


def test_hard_joint_bounds_are_enforced_even_within_step_limit():
    # prev_q already at the upper bound; even a step within max_step_rad
    # must not push past joint_upper_rad.
    limits = SafetyLimits(
        max_step_rad=np.full(6, 0.05),
        max_velocity_rad_s=np.full(6, 10.0),
        joint_lower_rad=np.full(6, -1.0),
        joint_upper_rad=np.full(6, 1.0),
    )
    clamp = SafetyClamp(limits, control_dt=0.1)
    prev_q = np.full(6, 1.0)
    raw_target = np.full(6, 1.02)

    result = clamp.clamp(raw_target, prev_q)

    np.testing.assert_allclose(result, np.full(6, 1.0))


def test_velocity_limit_can_be_the_binding_constraint():
    # max_velocity_rad_s * control_dt = 0.2 * 0.1 = 0.02, tighter than
    # max_step_rad=0.05 — the effective limit is the min of the two.
    limits = SafetyLimits(
        max_step_rad=np.full(6, 0.05),
        max_velocity_rad_s=np.full(6, 0.2),
        joint_lower_rad=np.full(6, -1.0),
        joint_upper_rad=np.full(6, 1.0),
    )
    clamp = SafetyClamp(limits, control_dt=0.1)
    prev_q = np.zeros(6)
    raw_target = np.full(6, 1.0)

    result = clamp.clamp(raw_target, prev_q)

    np.testing.assert_allclose(result, np.full(6, 0.02))
