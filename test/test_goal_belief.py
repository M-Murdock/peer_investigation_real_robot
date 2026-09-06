from kinova_gen3_lite_rl_bridge.goal_belief import blend_action, GoalBelief
import numpy as np

GOAL_NAMES = ('kettle', 'mug', 'infuser')


def _policy_deltas(kettle, mug, infuser):
    return {'kettle': kettle, 'mug': mug, 'infuser': infuser}


def test_starts_uniform():
    belief = GoalBelief(GOAL_NAMES, temperature=10.0)
    probs = belief.probabilities()
    np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3])


def test_belief_concentrates_on_matching_goal():
    belief = GoalBelief(GOAL_NAMES, temperature=10.0)
    kettle_delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    mug_delta = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    infuser_delta = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    deltas = _policy_deltas(kettle_delta, mug_delta, infuser_delta)

    # Human consistently moves exactly like the "kettle" goal's policy would.
    for _ in range(20):
        belief.update(kettle_delta, deltas)

    probs = belief.probabilities()
    assert probs[0] > 0.9, f'expected kettle to dominate, got {probs}'
    assert np.isclose(np.sum(probs), 1.0)


def test_direction_matters_not_magnitude():
    # Regression test for the real bug found during hardware testing:
    # evidence must be direction-only. "kettle"'s policy_delta is huge but
    # points the WRONG way; "mug"'s is tiny but points the SAME way the
    # human is jogging. mug should win despite being 100x smaller.
    belief = GoalBelief(('kettle', 'mug'), temperature=10.0)
    human_delta = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
    deltas = {
        'kettle': np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # huge, opposite direction
        'mug': np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),  # tiny, same direction
    }

    for _ in range(10):
        belief.update(human_delta, deltas)

    probs = belief.probabilities()
    assert probs[1] > probs[0], f'expected mug (correct direction) to win, got {probs}'


def test_update_is_a_noop_when_human_delta_is_zero():
    belief = GoalBelief(GOAL_NAMES, temperature=10.0)
    deltas = _policy_deltas(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    zero_delta = np.zeros(6)

    for _ in range(10):
        belief.update(zero_delta, deltas)

    np.testing.assert_allclose(belief.probabilities(), [1 / 3, 1 / 3, 1 / 3])


def test_probabilities_always_sum_to_one_even_when_clipped():
    belief = GoalBelief(GOAL_NAMES, temperature=1000.0)  # very peaked evidence
    kettle_delta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    deltas = _policy_deltas(
        kettle_delta,
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    for _ in range(50):
        belief.update(kettle_delta, deltas)

    probs = belief.probabilities()
    assert np.all(probs <= 0.99 + 1e-9)
    assert np.isclose(np.sum(probs), 1.0)


def test_reset_returns_to_uniform():
    belief = GoalBelief(GOAL_NAMES, temperature=10.0)
    kettle_delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    deltas = _policy_deltas(
        kettle_delta,
        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    )
    for _ in range(10):
        belief.update(kettle_delta, deltas)
    assert not np.allclose(belief.probabilities(), [1 / 3, 1 / 3, 1 / 3])

    belief.reset()
    np.testing.assert_allclose(belief.probabilities(), [1 / 3, 1 / 3, 1 / 3])


def test_blend_action_weights_by_probability_and_adds_human_delta():
    # Already-unit-length deltas, so normalization is a no-op here — this
    # test isolates the probability-weighting behavior.
    human_delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    deltas = _policy_deltas(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    probs = np.array([0.5, 0.3, 0.2])
    assist_step_rad = 0.02

    blended = blend_action(human_delta, deltas, probs, GOAL_NAMES, assist_step_rad)

    expected_direction = 0.5 * deltas['kettle'] + 0.3 * deltas['mug'] + 0.2 * deltas['infuser']
    expected = expected_direction * assist_step_rad + human_delta
    np.testing.assert_allclose(blended, expected)


def test_blend_action_is_magnitude_invariant():
    # Regression test for the real bug found during hardware testing:
    # "the joystick doesn't seem to do anything." A goal delta with a huge
    # raw magnitude must not dominate the blend more than an equally-
    # confident goal with a small raw magnitude — both get normalized to
    # the same assist_step_rad before weighting.
    human_delta = np.zeros(6)
    small_deltas = _policy_deltas(
        np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
    )
    huge_deltas = _policy_deltas(
        np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
    )
    probs = np.array([0.5, 0.3, 0.2])

    blended_small = blend_action(human_delta, small_deltas, probs, GOAL_NAMES, 0.02)
    blended_huge = blend_action(human_delta, huge_deltas, probs, GOAL_NAMES, 0.02)

    np.testing.assert_allclose(blended_small, blended_huge)


def test_blend_action_scales_with_assist_step_rad():
    human_delta = np.zeros(6)
    deltas = _policy_deltas(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    probs = np.array([1.0, 0.0, 0.0])

    blended = blend_action(human_delta, deltas, probs, GOAL_NAMES, assist_step_rad=0.05)

    np.testing.assert_allclose(blended, [0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
