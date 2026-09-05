"""
Rate/limit clamp strictly downstream of the policy's raw output.

Per the project runbook: never retune the trained policy's action_scale,
PD gains, or control decimation "for safety" — that mismatches
train/inference dynamics. Speed and joint-limit capping belongs here,
between the policy and the published command, not inside the policy.

Pure class, no rclpy dependency — unit-testable in isolation and reusable
unchanged by a future multi-goal (shared-autonomy) controller, since it
doesn't care where the pre-clamp action came from.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyLimits:
    max_step_rad: np.ndarray  # (6,) hard cap on |target - prev| per control cycle
    max_velocity_rad_s: np.ndarray  # (6,) secondary cap, converted to a step limit via control_dt
    joint_lower_rad: np.ndarray  # (6,) hard backstop — real URDF joint limits
    joint_upper_rad: np.ndarray  # (6,)


class SafetyClamp:
    def __init__(self, limits: SafetyLimits, control_dt: float):
        velocity_step_limit = np.asarray(limits.max_velocity_rad_s) * control_dt
        self._max_step = np.minimum(np.asarray(limits.max_step_rad), velocity_step_limit)
        self._lower = np.asarray(limits.joint_lower_rad)
        self._upper = np.asarray(limits.joint_upper_rad)

    def clamp(self, raw_target_q: np.ndarray, prev_commanded_q: np.ndarray) -> np.ndarray:
        """Limits the per-cycle step from prev_commanded_q, then the hard joint bounds."""
        raw_target_q = np.asarray(raw_target_q)
        prev_commanded_q = np.asarray(prev_commanded_q)
        delta = np.clip(raw_target_q - prev_commanded_q, -self._max_step, self._max_step)
        return np.clip(prev_commanded_q + delta, self._lower, self._upper)
