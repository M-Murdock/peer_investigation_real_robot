"""Loads measured Env A / Env B object 6D poses from
config/tea_table_objects.yaml (runbook v2 section 21B).

Every value currently returned is a placeholder - see the yaml file's own
header. This module only owns loading + the xyzw->wxyz convention switch;
it does not know or care that the numbers aren't real measurements yet.
"""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEA_TABLE_OBJECTS_YAML = PROJECT_ROOT / "config" / "tea_table_objects.yaml"
assert TEA_TABLE_OBJECTS_YAML.exists(), f"Not found: {TEA_TABLE_OBJECTS_YAML}"

ENV_IDS = ("env_a", "env_b")
OBJECT_IDS = ("kettle", "mug", "infuser")

Pos = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


def _xyzw_to_wxyz(q: list[float]) -> QuatWxyz:
    """ROS/runbook scalar-last -> MuJoCo/mjlab scalar-first."""
    x, y, z, w = q
    return (w, x, y, z)


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(TEA_TABLE_OBJECTS_YAML) as f:
        return yaml.safe_load(f)


def _pose_from_entry(entry: dict) -> tuple[Pos, QuatWxyz]:
    pos = tuple(entry["position_xyz"])
    quat = _xyzw_to_wxyz(entry["orientation_xyzw"])
    return pos, quat


def get_table_pose() -> tuple[Pos, QuatWxyz]:
    return _pose_from_entry(_load()["table"])


def get_object_pose(env_id: str, object_id: str) -> tuple[Pos, QuatWxyz]:
    """Measured (position, orientation) of `object_id` in `env_id`, both in
    the robot base frame. Raises KeyError with the available options if
    either id is unrecognized, rather than silently returning None."""
    data = _load()
    if env_id not in data or env_id not in ENV_IDS:
        raise KeyError(f"Unknown env_id {env_id!r}; expected one of {ENV_IDS}")
    if object_id not in data[env_id] or object_id not in OBJECT_IDS:
        raise KeyError(
            f"Unknown object_id {object_id!r}; expected one of {OBJECT_IDS}"
        )
    return _pose_from_entry(data[env_id][object_id])
