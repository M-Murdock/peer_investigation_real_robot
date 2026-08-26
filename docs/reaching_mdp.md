# Reaching MDP — Goal-Conditioned Environment (M2)

Builds the full `ManagerBasedRlEnvCfg` for the reaching-with-avoidance task
on top of the M1-validated arm and the table+obstacle scene
(`docs/mjlab_asset.md`, `docs/reaching_scene.md`).
Implements runbook sections 21–26. Code: `src/kinova_mjlab_reaching/tasks/reaching/`.

## Dependency fix (blocking, unrelated to the MDP itself)

Constructing *any* `ManagerBasedRlEnv` — including mjlab's own built-in
lift-cube example — crashed with:

```
AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1.
```

mjlab 1.3.0 (originally pinned) unconditionally sets `SimulationCfg.ls_parallel`
on the compiled warp model; `mujoco-warp` removed that attribute in 3.9.1,
and the project had `mujoco>=3.12.0` (M1 validated against 3.12.0), which
resolves to a `mujoco-warp` well past the removal.

No released mjlab version supports `mujoco==3.12.x` yet (checked 1.3.0
through 1.6.0 — latest, 1.6.0, requires `mujoco~=3.11.0`). Resolved (user
decision, 2026-08-25) by relaxing `pyproject.toml`'s `mujoco` pin to
`~=3.11.0` and bumping `mjlab` to `>=1.6.0` (has the `ls_parallel` fix).

**Re-validated after the downgrade** — nothing regressed:

```
mujoco 3.11.0: nq=10 nv=10 njnt=10 nbody=11 nu=6
zero-config ee_site: [0.057, -0.010, 1.0032] — matches fk_validation.md exactly
```

mjlab 1.6.0 also changed the `CommandTerm._update_command` signature (now
requires `env_ids: torch.Tensor | None = None`); updated `mdp/commands.py`
accordingly.

## Design

- **Action space:** `JointPositionActionCfg`, `use_default_offset=True`,
  `scale=1.0` — `target = action * scale + default_joint_pos` (default =
  home = 0). See "Bug caught by the sanity check" below for why this was
  chosen over the runbook's literal incremental-Δq description.
- **Command (`mdp/commands.py`):** `ReachingCommandCfg`/`ReachingCommand`
  samples one target position per episode in the scene's mount frame
  (depth 0.15–0.55 m, lateral ±0.28 m, height 0.03–0.30 m above the
  tabletop) — overlapping the obstacle layout rather than the arm's full
  workspace, since the task is reaching *around* the obstacles specifically.
  `resampling_time_range=(4.0, 4.0)` (= episode length) so exactly one
  target is sampled per episode, at reset.
- **Observations:** `joint_pos_rel`/`joint_vel_rel` (6 arm joints only, not
  the passive gripper joints), `ee_to_target_distance` (custom, base-frame
  vector), `last_action`. 21-dim actor/critic (identical groups — no
  privileged info yet).
- **Rewards:** `reaching_distance_reward` (dense Gaussian kernel,
  std=0.2, weight 1.0), `target_reached_bonus` (+10 while within 3 cm,
  runbook section 24's "+10" literally), `action_rate_l2` (-0.01),
  `joint_pos_limits` soft-limit penalty (-1.0, arm joints only). Kept
  deliberately minimal per section 24 — no collision-distance shaping yet.
- **Obstacle avoidance:** hard termination, not reward shaping. Three
  `ContactSensorCfg`s (`primary`: subtree rooted at `shoulder_link` — the
  whole arm+gripper — `secondary`: each obstacle's body), checked via a
  generic `illegal_contact` termination (adapted from mjlab's manipulation
  task pattern, copied rather than imported to avoid a cross-task
  dependency). Any contact between the arm and teapot/cup_1/cup_2 ends the
  episode immediately.
- **Events:** none configured explicitly — the default `reset_scene_to_default`
  event already resets every entity (robot + all 4 static objects) to its
  configured `init_state` with `env_origins` offsetting, which is sufficient
  since the obstacles are static (no per-episode object randomization).
- **Episode:** `decimation=4`, `episode_length_s=4.0` → 500 steps/episode at
  the default 0.002 s physics timestep (125 Hz control).

## Bug caught by the sanity check (runbook section 26 doing its job)

First implementation used `RelativeJointPositionActionCfg`
(`target = live_measured_current_pos + action`), matching the runbook's
literal "q_target = q_current + Δq" description. The **zero-action rollout
failed the intent of the check**: joint positions drifted from 0 to 1.96 rad
over a single 500-step episode with `action ≡ 0`.

Root cause: `RelativeJointPositionAction.apply_actions()`
(`mjlab.envs.mdp.actions.actions`) re-reads the *live* joint position every
control step and retargets to `current + action`. With action=0 this
retargets to "wherever the joint already is" — between control steps
gravity pulls the joint slightly, and the next retarget locks that sag in
as the new setpoint instead of correcting it, compounding over the episode.
Confirmed in isolation: holding a genuinely fixed `ctrl` target keeps the
arm within 0.015 rad over 0.5 s (matching the M1 actuator validation);
`RelativeJointPositionActionCfg` cannot do this by construction.

**Fix:** switched to `JointPositionActionCfg` (fixed default-offset anchor —
what mjlab's own reference tasks use). Re-ran the sanity check:
zero-action now holds at 0.014 rad (matches the isolated diagnostic exactly).

## Verification (runbook section 26)

`scripts/model_validation/test_reaching_env.py` — zero-action and
random-action rollouts, 1000 steps (2 full episodes) each:

```
zero-action:   all obs/reward finite, joint_pos_max_abs settles at 0.014 rad
               (holds home pose), 2 resets observed (500-step episodes), PASS
random-action: all obs/reward finite, joint_pos bounded (0.0-0.85 rad,
               reactive to input, no runaway/explosion), 2 resets, PASS
```

## Status

- [x] mjlab/mujoco dependency conflict resolved and re-validated
- [x] Environment constructs (scene + 4 obstacle/table entities + 3 contact
      sensors + command/action/observation/reward/termination managers)
- [x] Zero-action sanity check passes (after action-space fix)
- [x] Random-action sanity check passes
- [x] Visual confirmation via viewer (2026-08-25) — zero-action holds the
      home pose (arm stays pointing up, confirming the action-space fix);
      random-action moves the arm within bounds without ever reaching the
      target (expected — random has no goal-directed behavior); table,
      teapot, cup_1, cup_2, and the green target sphere all confirmed in
      correct positions
- [ ] PPO baseline (runbook section 27) — next step, not started
