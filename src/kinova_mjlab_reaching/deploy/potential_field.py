"""
Analytic attractive-potential-field joint-delta computation.

An alternative per-goal action source to the trained RL policy
(policy_loader.py), selected via shared_autonomy_node's `action_source`
parameter — the RL path stays fully intact and selectable. Uses kinpy for
FK/Jacobian, the same library already used and cross-validated for this
exact robot in docs/fk_validation.md, so no new, unvalidated kinematics
code.

Position-only, attraction-only: no orientation term (matches the RL
path's scope) and no repulsive/obstacle terms (the downstream
SafetyClamp already bounds per-cycle step size regardless of the field's
raw magnitude, and this project's own precedent is not to build avoidance
that hasn't been asked for — see the project plan's Milestone 3 section).
"""

from pathlib import Path
from typing import Optional

import kinpy as kp
import numpy as np

DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[3] / 'assets' / 'kinova_gen3_lite' / 'gen3_lite_resolved.urdf'
)
TIP_FRAME = 'tool_frame'
ROOT_FRAME = 'base_link'


def load_chain(urdf_path: Optional[Path] = None) -> kp.chain.SerialChain:
    """Build the kinpy serial chain for this robot, base_link to tool_frame."""
    path = urdf_path or DEFAULT_URDF_PATH
    with open(path, 'rb') as f:
        urdf_bytes = f.read()
    return kp.build_serial_chain_from_urdf(urdf_bytes, TIP_FRAME, ROOT_FRAME)


def compute_field_delta(
    chain: kp.chain.SerialChain,
    joint_pos: np.ndarray,
    target_xyz: np.ndarray,
    k_p: float,
    damping_lambda: float,
) -> np.ndarray:
    """
    Compute one cycle's attractive-potential-field joint delta.

    error = target_xyz - ee_pos (via FK), converted to a joint delta
    through the damped least-squares pseudoinverse of the position-only
    (3x6) Jacobian — robust near kinematic singularities, unlike the
    plain Jacobian transpose.
    """
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    ee_pos = np.array(chain.forward_kinematics(joint_pos).pos)
    error = np.asarray(target_xyz, dtype=np.float64) - ee_pos

    jacobian_pos = chain.jacobian(joint_pos)[:3, :]
    damped_pinv = jacobian_pos.T @ np.linalg.inv(
        jacobian_pos @ jacobian_pos.T + (damping_lambda**2) * np.eye(3)
    )
    delta = damped_pinv @ (k_p * error)
    return delta.astype(np.float32)
