"""
Track a belief over candidate goals and blend their suggested actions.

Ports the log-space belief structure from ada_assistance_policy's
GoalPredictor.py (log-space update, log-sum-exp normalization, a
per-goal probability ceiling with redistribution), with the evidence term
swapped for an action-alignment heuristic: our per-goal signal is a
trained RL policy, not GoalPredictor's original analytic Huber cost
function, so there is no clean Q(s,a) to reconstruct Javdani-style
hindsight-optimization evidence from (see the project plan's Milestone 2
section for the full reasoning). blend_action() mirrors
AssistancePolicy.get_assisted_action()'s structure: a belief-weighted
average of each goal's policy action, plus the human's own input.

Evidence is direction-only (cosine similarity), not raw squared distance:
the human's jog input is deliberately small (a safety-scaled joystick
delta) while an undertrained policy's raw action magnitude can be much
larger and varies goal to goal for reasons unrelated to direction — a
squared-distance comparison lets that magnitude dominate and swamps
whether the directions actually agree, discovered via real hardware
testing (belief barely moved regardless of stick direction). Comparing
unit vectors instead makes belief respond to *which way* the human is
jogging, independent of how large the policy's own suggestion happens to
be.
"""

from typing import Sequence

import numpy as np

_MAX_GOAL_PROB = 0.99
_MIN_VECTOR_NORM = 1e-6


def _logsumexp(log_values: np.ndarray) -> float:
    m = np.max(log_values)
    return float(m + np.log(np.sum(np.exp(log_values - m))))


class GoalBelief:
    def __init__(self, goal_names: Sequence[str], temperature: float):
        self.goal_names = list(goal_names)
        self._temperature = temperature
        self._log_belief = self._uniform_log_belief()

    def _uniform_log_belief(self) -> np.ndarray:
        n = len(self.goal_names)
        return np.log(np.full(n, 1.0 / n))

    def reset(self) -> None:
        self._log_belief = self._uniform_log_belief()

    def update(self, human_delta: np.ndarray, policy_deltas: dict) -> None:
        """
        Accumulate one cycle's evidence into the belief, then renormalize.

        Goals whose policy_delta points in the same direction as the
        human's actual commanded delta gain belief relative to the others.
        A no-op if the human isn't actually pushing the stick this cycle
        (direction is undefined at zero magnitude, and every goal would
        otherwise get identical, uninformative evidence).
        """
        human_norm = float(np.linalg.norm(human_delta))
        if human_norm < _MIN_VECTOR_NORM:
            return
        human_dir = human_delta / human_norm

        evidence = np.array(
            [self._temperature * self._alignment(human_dir, policy_deltas[name])
             for name in self.goal_names]
        )
        self._log_belief = self._log_belief + evidence
        self._normalize()
        self._clip()

    @staticmethod
    def _alignment(human_dir: np.ndarray, policy_delta: np.ndarray) -> float:
        """Cosine similarity in [-1, 1]; 0 if the policy_delta is ~zero."""
        policy_norm = float(np.linalg.norm(policy_delta))
        if policy_norm < _MIN_VECTOR_NORM:
            return 0.0
        return float(np.dot(human_dir, policy_delta / policy_norm))

    def _normalize(self) -> None:
        self._log_belief = self._log_belief - _logsumexp(self._log_belief)

    def _clip(self, max_prob: float = _MAX_GOAL_PROB) -> None:
        probs = np.exp(self._log_belief)
        over = probs > max_prob
        if not np.any(over):
            return
        excess = float(np.sum(probs[over] - max_prob))
        probs[over] = max_prob
        under = ~over
        if np.any(under):
            under_sum = np.sum(probs[under])
            if under_sum > 0:
                probs[under] += excess * probs[under] / under_sum
            else:
                # All "losing" goals underflowed to exactly 0 (extreme
                # evidence) — proportional redistribution is 0/0 here, so
                # split the excess equally instead.
                probs[under] += excess / np.sum(under)
        probs = probs / np.sum(probs)
        with np.errstate(divide='ignore'):
            self._log_belief = np.log(probs)

    def probabilities(self) -> np.ndarray:
        return np.exp(self._log_belief)


def blend_action(
    human_delta: np.ndarray,
    policy_deltas: dict,
    probs: np.ndarray,
    goal_names: Sequence[str],
    assist_step_rad: float,
) -> np.ndarray:
    """
    Blend the belief-weighted goal directions with the human's own delta.

    Each goal's raw delta is normalized to a unit direction, scaled to
    assist_step_rad, before weighting and summing — not used raw. A raw
    sum lets whichever goal's delta has the largest magnitude dominate
    the blend regardless of belief, and that magnitude can be far larger
    than the human's deliberately small, safety-scaled jog input (a
    potential field's raw delta grows with distance-to-goal; an
    undertrained RL policy's raw action magnitude is similarly
    uncalibrated) — discovered via real hardware testing as "the
    joystick doesn't seem to do anything," because after the final
    safety clamp, the dominant unnormalized goal direction alone already
    saturated the per-cycle step, swamping the human's contribution.
    Normalizing first keeps the human's steering meaningfully influential
    regardless of how far away any goal currently is.

    The final combination is a straight average of the human's delta and
    the belief-weighted assistance direction — not an additive sum — so
    each side counts for half regardless of belief confidence.
    """
    weighted_direction = np.zeros_like(human_delta)
    for prob, name in zip(probs, goal_names):
        delta = policy_deltas[name]
        norm = float(np.linalg.norm(delta))
        direction = delta / norm if norm > _MIN_VECTOR_NORM else delta
        weighted_direction = weighted_direction + prob * direction
    return (human_delta + weighted_direction * assist_step_rad) / 2.0
