"""Local-frame collision geometry for the tea-table objects (runbook v2
section 21A): kettle, mug, infuser, and the shared table.

Kettle/mug/infuser dimensions are measured (see each section below). The
table's footprint and height offset are measured too (see the Table
section); only its thickness (TABLE_HALF_EXTENTS z) is still a guess - real
table dimensions were never given, only its footprint and mounting height.
Handle/rim mounting heights not covered by a real measurement are called
out individually where they occur.

Object-frame origin convention (shared with config/tea_table_objects.yaml):
each object's local origin is the center of its base, resting exactly on
the tabletop - z=0 in the object's own frame is its bottom face. World
placement (which differs between Env A and Env B) is applied on top of this
local geometry via EntityCfg.InitialStateCfg(pos=..., rot=...), not baked
into the spec here - see scene_registry.py. This mirrors how
robots/gen3_lite.py separates the robot's local mesh from its world
placement (HOME_KEYFRAME), rather than reaching/scene.py's approach of
baking world position into each spec_fn (that task only ever has one scene,
so there was no need to separate the two).
"""

import mujoco

from mjlab.entity import EntityCfg

_IN = 0.0254  # inches -> meters

##
# Table (shared/fixed across Env A and Env B).
#
# Real measured footprint: 23.5 in x 29 in. The arm base sits on a 2-in-
# thick cylindrical fixture bolted to the table, so the origin (arm base)
# is 2 in ABOVE the tabletop surface, and the arm's mounting point on the
# table is offset from the table's own edges - see
# config/tea_table_objects.yaml for the resulting world pose (table center
# + surface height) derived from these numbers. Tabletop thickness (1 3/16
# in) is measured too - see ARM_BASE_FIXTURE below for the fixture itself.
##

TABLE_HALF_EXTENTS = (0.3683, 0.29845, 1.1875 * _IN / 2)  # (x, y, z), m.
"""x/y from the measured 23.5in x 29in footprint, z from the measured
1 3/16in tabletop thickness."""


def get_table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    body.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=TABLE_HALF_EXTENTS,
        rgba=(0.55, 0.4, 0.28, 1.0),
    )
    return spec


##
# Arm base fixture: the cylindrical riser the arm is actually bolted to,
# bridging the 2 in gap between the tabletop surface and the arm's own
# base (world origin) - without it the arm renders floating in mid-air
# above the table. Fixed at world (x=0, y=0), directly under the arm;
# unlike the table/tea objects it has no per-environment pose (it's a
# permanent installation fact, not something that moves between Env A/B),
# so scene_registry.py places it with a hardcoded position rather than
# reading one from tea_table_objects.yaml.
##

ARM_BASE_FIXTURE_RADIUS = 3.5 * _IN / 2  # 3.5 in diameter
ARM_BASE_FIXTURE_HEIGHT = 2.0 * _IN  # matches the measured arm-to-table gap


def get_arm_base_fixture_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="arm_base_fixture")
    body.add_geom(
        name="arm_base_fixture_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(ARM_BASE_FIXTURE_RADIUS, ARM_BASE_FIXTURE_HEIGHT / 2, 0.0),
        pos=(0.0, 0.0, ARM_BASE_FIXTURE_HEIGHT / 2),
        rgba=(0.05, 0.05, 0.05, 1.0),
    )
    return spec


##
# Kettle: measured body + top loop (bail) handle.
##

KETTLE_BODY_RADIUS = 6.5 * _IN / 2  # base diameter 6.5 in
KETTLE_BODY_HEIGHT = 5.5 * _IN  # bottom to top of lid
KETTLE_HANDLE_BAR_RADIUS = 1.0 * _IN / 2  # bar is 1 in wide/thick -> diameter 1 in
KETTLE_HANDLE_RISE = 3.5 * _IN  # bar sits this far above the lid top
KETTLE_PLATE_WIDTH = 0.5 * _IN  # tangential extent
KETTLE_PLATE_HEIGHT = 3.5 * _IN  # the two plates ARE the bar's support height
KETTLE_PLATE_THICKNESS = 0.25 * _IN  # radial extent

KETTLE_HANDLE_BAR_Z = KETTLE_BODY_HEIGHT + KETTLE_HANDLE_RISE
"""Bar sits atop the two plates, so its height above the base equals the
lid height plus the full plate height (= the measured "3.5 in offset
higher from the top of the lid")."""
KETTLE_GRASP_SITE_POS = (0.0, 0.0, KETTLE_HANDLE_BAR_Z)  # handle-bar midpoint


def get_kettle_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="kettle")
    body.add_geom(
        name="kettle_body",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(KETTLE_BODY_RADIUS, KETTLE_BODY_HEIGHT / 2, 0.0),
        pos=(0.0, 0.0, KETTLE_BODY_HEIGHT / 2),
        rgba=(0.75, 0.75, 0.78, 1.0),
    )
    # Two vertical support plates, mounted at the rim on opposite sides
    # (+-x), straddling the lid edge, rising from the lid top to the bar.
    plate_z = KETTLE_BODY_HEIGHT + KETTLE_PLATE_HEIGHT / 2
    for side, sign in (("pos", 1.0), ("neg", -1.0)):
        body.add_geom(
            name=f"kettle_handle_plate_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                KETTLE_PLATE_THICKNESS / 2,
                KETTLE_PLATE_WIDTH / 2,
                KETTLE_PLATE_HEIGHT / 2,
            ),
            pos=(sign * KETTLE_BODY_RADIUS, 0.0, plate_z),
            rgba=(0.2, 0.2, 0.2, 1.0),
        )
    # Horizontal loop bar spanning between the two plates' tops.
    body.add_geom(
        name="kettle_handle_bar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(KETTLE_HANDLE_BAR_RADIUS, 0.0, 0.0),
        fromto=(
            -KETTLE_BODY_RADIUS,
            0.0,
            KETTLE_HANDLE_BAR_Z,
            KETTLE_BODY_RADIUS,
            0.0,
            KETTLE_HANDLE_BAR_Z,
        ),
        rgba=(0.2, 0.2, 0.2, 1.0),
    )
    return spec


##
# Mug: measured body + side loop handle (added for collision-interference
# checking, per runbook section 21A.1 - grasp strategy stays at the rim,
# not the handle, so the handle isn't wired into tea_table_grasps.yaml).
##

MUG_BODY_RADIUS = 3.0 * _IN / 2  # base diameter 3 in
MUG_BODY_HEIGHT = 3.5 * _IN
MUG_RIM_SITE_POS = (MUG_BODY_RADIUS, 0.0, MUG_BODY_HEIGHT - 0.01)

# Side loop handle, same two-arms-plus-bar construction as the infuser's
# (see INFUSER_HANDLE_* for the width/thickness axis convention).
MUG_HANDLE_REACH = 0.75 * _IN
MUG_HANDLE_LOOP_HEIGHT = 2.4 * _IN
MUG_HANDLE_OFFSET_FROM_BASE = 0.75 * _IN  # bottom of the loop, above the base
MUG_HANDLE_STRIP_WIDTH = 0.675 * _IN  # out-of-plane (sideways) dimension
MUG_HANDLE_STRIP_THICKNESS = (3.0 / 16.0) * _IN  # in-plane (silhouette) dimension


def get_mug_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="mug")
    body.add_geom(
        name="mug_body",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(MUG_BODY_RADIUS, MUG_BODY_HEIGHT / 2, 0.0),
        pos=(0.0, 0.0, MUG_BODY_HEIGHT / 2),
        rgba=(0.9, 0.9, 0.85, 1.0),
    )

    z_bot = MUG_HANDLE_OFFSET_FROM_BASE
    z_top = z_bot + MUG_HANDLE_LOOP_HEIGHT
    x_inner = MUG_BODY_RADIUS
    x_outer = MUG_BODY_RADIUS + MUG_HANDLE_REACH
    x_arm_mid = (x_inner + x_outer) / 2
    for side, z_arm in (("top", z_top), ("bottom", z_bot)):
        body.add_geom(
            name=f"mug_handle_arm_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                MUG_HANDLE_REACH / 2,
                MUG_HANDLE_STRIP_WIDTH / 2,
                MUG_HANDLE_STRIP_THICKNESS / 2,
            ),
            pos=(x_arm_mid, 0.0, z_arm),
            rgba=(0.9, 0.9, 0.85, 1.0),
        )
    body.add_geom(
        name="mug_handle_bar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
            MUG_HANDLE_STRIP_THICKNESS / 2,
            MUG_HANDLE_STRIP_WIDTH / 2,
            MUG_HANDLE_LOOP_HEIGHT / 2,
        ),
        pos=(x_outer, 0.0, (z_top + z_bot) / 2),
        rgba=(0.9, 0.9, 0.85, 1.0),
    )
    return spec


##
# Infuser: measured tapered (frustum) body + side loop handle.
#
# MuJoCo has no native frustum primitive, so the taper is approximated with
# 2 stacked cylinders, each sized to the true cone radius at its own
# segment's vertical midpoint - a stepped but reasonably faithful
# approximation of a straight-sided cone, not a real curved-wall shape.
##

INFUSER_BASE_RADIUS = 4.0 * _IN / 2  # base diameter 4 in
INFUSER_TOP_RADIUS = 3.25 * _IN / 2  # top diameter 3.25 in
INFUSER_BODY_HEIGHT = 4.5 * _IN  # base to top lid
INFUSER_BODY_BOTTOM_SEGMENT_HEIGHT = 2.0 * _IN
INFUSER_BODY_TOP_SEGMENT_HEIGHT = INFUSER_BODY_HEIGHT - INFUSER_BODY_BOTTOM_SEGMENT_HEIGHT


def _infuser_radius_at(z: float) -> float:
    """Linear taper from INFUSER_BASE_RADIUS (z=0) to INFUSER_TOP_RADIUS
    (z=INFUSER_BODY_HEIGHT) - the true cone profile the stacked-cylinder
    approximation is sampled from."""
    frac = z / INFUSER_BODY_HEIGHT
    return INFUSER_BASE_RADIUS + (INFUSER_TOP_RADIUS - INFUSER_BASE_RADIUS) * frac


# Side loop handle: a bent flat strip (0.6 in wide x 0.25 in thick),
# mounted horizontally-then-vertically-then-horizontally like the kettle's
# bail handle but rotated 90 degrees - two horizontal arms reaching
# INFUSER_HANDLE_REACH out from the body wall, closed by one vertical bar,
# spanning INFUSER_HANDLE_LOOP_HEIGHT top-to-bottom. You'd grip it by
# passing fingers through sideways (+-y), same as a mug handle.
#
# "Width" is the strip's out-of-plane (sideways, +-y) extent - how far the
# material itself sticks out from the vertical x-z silhouette plane the
# loop is bent in. "Thickness" is the in-plane extent (z for the arms, x
# for the bar) - the dimension you'd see edge-on looking at the loop from
# the side.
INFUSER_HANDLE_LOOP_HEIGHT = 2.5 * _IN
INFUSER_HANDLE_REACH = 1.5 * _IN  # how far the loop extrudes from the body
INFUSER_HANDLE_STRIP_WIDTH = 0.6 * _IN  # out-of-plane (sideways) dimension
INFUSER_HANDLE_STRIP_THICKNESS = 0.25 * _IN  # in-plane (silhouette) dimension

INFUSER_HANDLE_MOUNT_Z = INFUSER_BODY_HEIGHT / 2
"""Vertical placement of the handle's own center on the body - not measured,
a placeholder (mid-height is a generic, plausible side-handle position).
Confirm against the real infuser when possible."""
_INFUSER_HANDLE_MOUNT_RADIUS = _infuser_radius_at(INFUSER_HANDLE_MOUNT_Z)
INFUSER_GRASP_SITE_POS = (
    _INFUSER_HANDLE_MOUNT_RADIUS + INFUSER_HANDLE_REACH,
    0.0,
    INFUSER_HANDLE_MOUNT_Z,
)  # outer bar midpoint - the natural grip point


def get_infuser_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="infuser")

    z_cursor = 0.0
    for name, seg_h in (
        ("bottom", INFUSER_BODY_BOTTOM_SEGMENT_HEIGHT),
        ("top", INFUSER_BODY_TOP_SEGMENT_HEIGHT),
    ):
        z_mid = z_cursor + seg_h / 2
        body.add_geom(
            name=f"infuser_body_{name}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(_infuser_radius_at(z_mid), seg_h / 2, 0.0),
            pos=(0.0, 0.0, z_mid),
            rgba=(0.6, 0.75, 0.6, 1.0),
        )
        z_cursor += seg_h

    z_top = INFUSER_HANDLE_MOUNT_Z + INFUSER_HANDLE_LOOP_HEIGHT / 2
    z_bot = INFUSER_HANDLE_MOUNT_Z - INFUSER_HANDLE_LOOP_HEIGHT / 2
    x_inner = _INFUSER_HANDLE_MOUNT_RADIUS
    x_outer = _INFUSER_HANDLE_MOUNT_RADIUS + INFUSER_HANDLE_REACH
    x_arm_mid = (x_inner + x_outer) / 2
    for side, z_arm in (("top", z_top), ("bottom", z_bot)):
        body.add_geom(
            name=f"infuser_handle_arm_{side}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(
                INFUSER_HANDLE_REACH / 2,
                INFUSER_HANDLE_STRIP_WIDTH / 2,
                INFUSER_HANDLE_STRIP_THICKNESS / 2,
            ),
            pos=(x_arm_mid, 0.0, z_arm),
            rgba=(0.2, 0.2, 0.2, 1.0),
        )
    body.add_geom(
        name="infuser_handle_bar",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(
            INFUSER_HANDLE_STRIP_THICKNESS / 2,
            INFUSER_HANDLE_STRIP_WIDTH / 2,
            INFUSER_HANDLE_LOOP_HEIGHT / 2,
        ),
        pos=(x_outer, 0.0, INFUSER_HANDLE_MOUNT_Z),
        rgba=(0.2, 0.2, 0.2, 1.0),
    )
    return spec


OBJECT_SPEC_FNS = {
    "kettle": get_kettle_spec,
    "mug": get_mug_spec,
    "infuser": get_infuser_spec,
}


def get_static_entity_cfg(spec_fn) -> EntityCfg:
    """Wrap a spec_fn as a static (no-freejoint) EntityCfg. World placement
    (position/orientation, which differs per environment) is set separately
    via InitialStateCfg by the caller - see scene_registry.py."""
    return EntityCfg(spec_fn=spec_fn)
