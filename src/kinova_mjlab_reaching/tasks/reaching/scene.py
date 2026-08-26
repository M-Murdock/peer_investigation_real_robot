"""Static scene objects for the reaching task: a tea-serving table and
obstacles the arm must learn to avoid while reaching.

The robot's fixed base sits at the world origin (see
docs/mjlab_asset.md). All objects are placeholder primitives with
no freejoint — they are static/immovable collision bodies, not manipulable
objects.

Object positions below are authored in a "mount frame": x = depth out from
the table's back edge (nearest the robot), y = lateral offset. This frame is
then rotated into world coordinates by MOUNT_YAW_DEG, so the whole table +
obstacle layout can be re-oriented relative to the arm's base without
touching any individual position (2026-08-25: rotated 90 deg CCW as seen
from above, i.e. what used to be "straight ahead" of the mount is now off
to one side).
"""

import math

import mujoco

from mjlab.entity import EntityCfg

##
# Mount-frame -> world-frame rotation.
##

MOUNT_YAW_DEG = 90.0
"""Rotation of the mount frame about world +z. Negative = clockwise as seen
from above (looking down -z, standard right-handed x-right/y-up/z-toward-
viewer top view)."""

_yaw = math.radians(MOUNT_YAW_DEG)
_cos, _sin = math.cos(_yaw), math.sin(_yaw)


def _mount_to_world(x_depth: float, y_lateral: float) -> tuple[float, float]:
    """Rotate a (depth, lateral) mount-frame point into world (x, y)."""
    return (
        _cos * x_depth - _sin * y_lateral,
        _sin * x_depth + _cos * y_lateral,
    )


##
# Table.
##

TABLE_HALF_EXTENTS = (0.30, 0.30, 0.015)  # (depth, lateral, height) half-extents, m.
"""Square footprint so the rotation above only moves the table's center, not
its shape."""

_table_center_world_xy = _mount_to_world(TABLE_HALF_EXTENTS[0], 0.0)
TABLE_POS = (*_table_center_world_xy, -TABLE_HALF_EXTENTS[2])
"""Tabletop spans depth=[0, 0.6], lateral=[-0.3, 0.3] in mount frame, top
surface at z=0 (flush with the robot base), back edge at the mount point."""


def get_table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table", pos=TABLE_POS)
    body.add_geom(
        name="tabletop",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=TABLE_HALF_EXTENTS,
        rgba=(0.55, 0.4, 0.28, 1.0),
    )
    return spec


##
# Obstacles (placeholder tea-set geometry, all resting on the tabletop).
##

# Mount-frame (depth, lateral) placement, sized against the sampled reachable
# envelope at table height (median radial reach ~0.37 m, p95 ~0.69 m — see
# docs/reaching_scene.md): comfortably reachable, not edge-of-range.
_TEAPOT_MOUNT_XY = (0.40, 0.0)
_CUP1_MOUNT_XY = (0.30, 0.18)
_CUP2_MOUNT_XY = (0.30, -0.18)


def _cylinder_spec(
    name: str,
    pos: tuple[float, float, float],
    radius: float,
    half_height: float,
    rgba: tuple[float, float, float, float],
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=name, pos=pos)
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, half_height, 0.0),
        rgba=rgba,
    )
    return spec


def get_teapot_spec() -> mujoco.MjSpec:
    x, y = _mount_to_world(*_TEAPOT_MOUNT_XY)
    return _cylinder_spec(
        "teapot",
        pos=(x, y, 0.06),
        radius=0.045,
        half_height=0.06,
        rgba=(0.75, 0.75, 0.78, 1.0),
    )


def get_cup1_spec() -> mujoco.MjSpec:
    x, y = _mount_to_world(*_CUP1_MOUNT_XY)
    return _cylinder_spec(
        "cup_1",
        pos=(x, y, 0.035),
        radius=0.028,
        half_height=0.035,
        rgba=(0.9, 0.9, 0.85, 1.0),
    )


def get_cup2_spec() -> mujoco.MjSpec:
    x, y = _mount_to_world(*_CUP2_MOUNT_XY)
    return _cylinder_spec(
        "cup_2",
        pos=(x, y, 0.035),
        radius=0.028,
        half_height=0.035,
        rgba=(0.9, 0.9, 0.85, 1.0),
    )


def get_scene_object_cfgs() -> dict[str, EntityCfg]:
    """Table + obstacle entities, keyed by the name they'll be attached under."""
    return {
        "table": EntityCfg(spec_fn=get_table_spec),
        "teapot": EntityCfg(spec_fn=get_teapot_spec),
        "cup_1": EntityCfg(spec_fn=get_cup1_spec),
        "cup_2": EntityCfg(spec_fn=get_cup2_spec),
    }
