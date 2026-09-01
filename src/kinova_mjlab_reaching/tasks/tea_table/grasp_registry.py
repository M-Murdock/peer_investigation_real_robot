"""Object-relative grasp/pre-grasp transform chain (runbook v2 section 21C
and 23):

    T_base_pregrasp = T_base_object x T_object_grasp x T_grasp_pregrasp_offset

`T_object_grasp` (config/tea_table_grasps.yaml) is the same for a given
object class in both Env A and Env B - only `T_base_object`
(object_pose_registry) differs between scenes. This is the mechanism that
keeps "one grasp strategy per object class, reused across environments"
(runbook v2 invariant 3/9) instead of six independent hand-authored world
poses.

ALL VALUES ARE PLACEHOLDERS - see tea_table_grasps.yaml's own header.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import (
    Pos,
    QuatWxyz,
    get_object_pose,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEA_TABLE_GRASPS_YAML = PROJECT_ROOT / "config" / "tea_table_grasps.yaml"
assert TEA_TABLE_GRASPS_YAML.exists(), f"Not found: {TEA_TABLE_GRASPS_YAML}"


def _wxyz_to_xyzw(q: tuple[float, float, float, float]) -> list[float]:
    w, x, y, z = q
    return [x, y, z, w]


def _xyzw_to_wxyz(q) -> QuatWxyz:
    x, y, z, w = q
    return (w, x, y, z)


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(TEA_TABLE_GRASPS_YAML) as f:
        return yaml.safe_load(f)


def get_grasp_spec(object_id: str) -> dict:
    """Raw grasp spec dict for one object class - grasp_pose_in_object_frame,
    pregrasp_offset_in_grasp_frame, descent_distance_m, gripper_close_command
    (see the yaml file for exact shape). object_id not being a key raises a
    plain KeyError."""
    return _load()[object_id]


def compute_grasp_pose(env_id: str, object_id: str) -> tuple[Pos, QuatWxyz]:
    """T_base_object x T_object_grasp, i.e. the nominal grasp pose for this
    object in this environment, in the robot base frame."""
    obj_pos, obj_quat_wxyz = get_object_pose(env_id, object_id)
    r_obj = Rotation.from_quat(_wxyz_to_xyzw(obj_quat_wxyz))

    spec = get_grasp_spec(object_id)["grasp_pose_in_object_frame"]
    grasp_pos_local = np.array(spec["position_xyz"])
    r_grasp_local = Rotation.from_quat(spec["orientation_xyzw"])

    p_base_grasp = np.array(obj_pos) + r_obj.apply(grasp_pos_local)
    r_base_grasp = r_obj * r_grasp_local
    return tuple(p_base_grasp.tolist()), _xyzw_to_wxyz(r_base_grasp.as_quat())


def compute_pregrasp_pose(env_id: str, object_id: str) -> tuple[Pos, QuatWxyz]:
    """T_base_grasp x T_grasp_pregrasp_offset. Orientation is identical to
    the grasp pose - the offset is translation-only (pull back along the
    grasp frame's own axes), matching runbook section 22.1's usage: only
    the translation component (pregrasp_xyz) goes to the RL policy, the
    orientation is retained unchanged for the future deterministic ALIGN
    stage."""
    grasp_pos, grasp_quat_wxyz = compute_grasp_pose(env_id, object_id)
    r_grasp = Rotation.from_quat(_wxyz_to_xyzw(grasp_quat_wxyz))

    offset_local = np.array(
        get_grasp_spec(object_id)["pregrasp_offset_in_grasp_frame"]["position_xyz"]
    )
    p_base_pregrasp = np.array(grasp_pos) + r_grasp.apply(offset_local)
    return tuple(p_base_pregrasp.tolist()), grasp_quat_wxyz
