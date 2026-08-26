# Reaching Scene — Table + Static Obstacles

Adds the physical setting for the reaching-with-avoidance task: the arm is
mounted at the back edge of a tea-serving table, with static obstacles
(placeholder tea-set geometry) the policy must learn to avoid. Built in
`src/kinova_mjlab_reaching/tasks/reaching/scene.py`, on top of the
M1-validated arm (`docs/validation/mjlab_asset.md`).

## Design

- **Mount point:** robot base stays at the world origin (unchanged from
  M1/M1.2) — its own validated MJCF/kinematics are never touched. The table
  and obstacles are positioned relative to it, not the other way around.
- **Mount-frame rotation:** object positions are authored in a "mount frame"
  (depth out from the table's back edge, lateral offset) and rotated into
  world coordinates by `MOUNT_YAW_DEG = 90°` (counterclockwise as seen from
  above), per user request (2026-08-25, corrected from an initial CW guess)
  to reorient the whole layout relative to the arm's base. This is
  physically equivalent to rotating the arm's mounting bracket 90° CCW and
  is implemented as a scene-layout
  rotation rather than a change to the robot entity's own frame, because a
  fixed-base entity's `init_state.rot` only takes effect through a reset
  event (none exist yet — no MDP/events have been built) — rotating the
  scene layout gives the identical relative pose immediately and
  independent of the event system, without risking the M1-validated
  kinematics.
- **Table:** square box, half-extents `(0.30, 0.30, 0.015)` m (square so the
  rotation only moves its center, not its shape) — tabletop spans
  depth=[0, 0.6], lateral=[-0.3, 0.3] in mount frame, top surface flush with
  z=0 (the robot's base height), back edge at the mount point.
- **Obstacles:** placeholder cylinders resting on the tabletop, in mount
  frame (depth, lateral) — `teapot` (r=0.045, h=0.12) at (0.40, 0.0),
  `cup_1`/`cup_2` (r=0.028, h=0.07) at (0.30, ±0.18). Moved further out from
  the arm per user request while confirmed still comfortably reachable (see
  Verification). All static — no freejoint, immovable.
- All objects are placeholder primitives per user decision (2026-08-25):
  real dimensions/layout can replace these later without touching the
  robot config or (once written) the MDP, since obstacle positions are
  read from entity state, not hardcoded.

## Verification

Reachability at table height was sampled directly rather than assumed:
200k random joint configs (within limits) were forward-kinematics'd on the
validated `gen3_lite.xml`, filtered to `ee_site` points with
`z ∈ [0.02, 0.15]` (the teapot/cup height band):

```
samples in table-height band: 21608 / 200000
radial reach at table height -> min 0.002  p5 0.094  median 0.374  p95 0.694  max 0.757
```

All three obstacle positions (radial distance 0.35–0.40 m from the base)
fall in the comfortable middle of this range, not near the workspace edge —
each has a randomly-sampled EE point within ~1 cm of it, a strong feasibility
proxy.

Scene built via `mjlab.scene.Scene` (not the full `ManagerBasedRlEnvCfg`,
since the MDP doesn't exist yet — scene composition is being validated
first, independently):

```
nq: 10  nv: 10  nu: 6  nbody: 20
ncon at home pose: 0
table pos:  [ 0.00,  0.30, -0.015]
teapot pos: [ 0.00,  0.40,  0.06]
cup_1 pos:  [-0.18,  0.30,  0.035]
cup_2 pos:  [ 0.18,  0.30,  0.035]
ee_site (home pose): [0.057, -0.010, 1.003]  — matches fk_validation.md exactly
```

No interpenetration between the robot, table, or obstacles at the home
(all-zero) pose. The robot's zero-configuration FK is unaffected by adding
or rotating the scene objects, confirming they didn't accidentally perturb
the robot entity itself.

Visual check: `uv run python scripts/model_validation/view_reaching_scene.py`

## Status

- [x] Table positioned relative to robot mount point (back edge, flush surface)
- [x] Obstacles rest on tabletop without clipping
- [x] Zero-contact check at home pose
- [x] ee_site FK unaffected by scene composition
- [ ] Visual confirmation in viewer (run the script above)
- [ ] Reaching MDP (commands/observations/rewards/terminations) — next step
