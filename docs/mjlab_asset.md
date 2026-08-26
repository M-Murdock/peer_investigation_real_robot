# MJLab Asset — Kinova Gen3 Lite (M1.2)

Wraps the M1-validated MJCF (`assets/kinova_gen3_lite/gen3_lite.xml`) as an
mjlab `EntityCfg` in `src/kinova_mjlab_reaching/robots/gen3_lite.py`. The MJCF
itself is unmodified — only referenced.

## Approach

- **Actuators:** `XmlActuatorCfg` wraps the 6 existing `<position>` actuators
  by joint-name regex (`joint_[1-6]`) instead of re-specifying kp/kv/effort
  limits, so the M1-validated gains (kp=100, kv=10) carry through unchanged.
- **Gripper joints:** left unactuated (no `ActuatorCfg` targets them),
  consistent with `docs/joint_mapping.md`.
- **Home keyframe:** all joints at 0 rad, matching the "zero" configuration
  in `docs/fk_validation.md`.
- **Soft joint limits:** `soft_joint_pos_limit_factor=0.9`, matching the
  convention used by every robot in mjlab's own asset zoo (YAM, G1, Go1).
- **Collisions:** left at MJCF defaults (no dedicated collision geoms exist
  yet — visual mesh geoms double as collision). No self-collision issues
  were observed in M1. Revisit only if training throughput demands
  simplified collision geometry (runbook §19/§29).

## Verification

```
nq: 10   nv: 10   nu: 6
actuator names: act_joint_1 .. act_joint_6
is_fixed_base: True
is_articulated: True
joint_names: joint_1..joint_6, right_finger_bottom_joint, right_finger_tip_joint,
             left_finger_bottom_joint, left_finger_tip_joint
```

Matches M1's `mujoco_joint_table.md` exactly (`nq=10 nv=10 njnt=10 nu=6`
after actuators were added).

## Known benign warning

Compiling the entity standalone (`robot.spec.compile()`) prints:

```
Attach conflict when attaching 'gen3_lite_gen3_lite_2f', policy is 'warning'
integrator: parent has 0 (default), child has 3, keeping parent value
```

Cause: `mjlab.utils.spec.auto_wrap_fixed_base_mocap` wraps every fixed-base
entity in a fresh mocap parent spec (default `<option>`, so integrator=Euler)
to allow per-env positioning. This silently discards our MJCF's
`<option integrator="implicitfast"/>` at the spec level.

**Not a real issue:** `mjlab.sim.MujocoCfg.apply()` unconditionally sets
`model.opt.integrator` on the final compiled `MjModel` regardless of what
the spec-level option was, and `MujocoCfg`'s default is
`integrator="implicitfast"`, `timestep=0.002` — identical to what M1
validated. Confirmed by reading `mjlab/sim/sim.py`. No project-level
override is required, but if `MujocoCfg` is ever overridden when building
the reaching env, keep `integrator="implicitfast"` — the actuator gains
(kp=100/kv=10) were validated against it.

## Status

- [x] EntityCfg builds and compiles without dynamics-relevant warnings
- [x] nq/nv/nu match M1 exactly
- [x] Actuator gains preserved from validated MJCF (not re-specified)
- [x] Home keyframe matches M1's zero-configuration FK validation
- [ ] `num_envs=1` scene/env smoke test (next: reaching task scene assembly)
