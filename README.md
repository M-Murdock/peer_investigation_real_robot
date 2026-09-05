# kinova_mjlab_reaching / kinova_gen3_lite_rl_bridge

This repository holds two coupled projects that deliberately run in
**separate Python environments**:

| | What it is | Environment |
|---|---|---|
| `kinova_mjlab_reaching` | Trains a PPO reaching policy for the Kinova Gen3 Lite arm in MuJoCo/mjlab | `uv`-managed Python 3.13 (`pyproject.toml`) |
| `kinova_gen3_lite_rl_bridge` | ROS 2 package that deploys a trained checkpoint onto the **real** Gen3 Lite arm | System Python 3.12 (ROS 2 Jazzy, `rclpy`) |

They can't share a process: `rclpy` is a Jazzy apt package built for
Python 3.12, while `torch`/`mjlab`/`rsl_rl`/`mujoco` only exist in this
repo's `uv` venv (mjlab hard-requires `python>=3.13`). The two talk to
each other over a small local socket protocol instead — see
[Real-hardware deployment](#real-hardware-deployment-kinova_gen3_lite_rl_bridge)
below.

Background and full design rationale for both halves lives in
[`kinova_gen3_lite_tea_table_grasp_adaptation_runbook_v2.md`](kinova_gen3_lite_tea_table_grasp_adaptation_runbook_v2.md)
(the project runbook) and [`docs/`](docs/) (robot-model validation:
joint mapping, FK validation, MJCF asset notes, PPO baseline results).

## Repository layout

```
pyproject.toml, uv.lock, .python-version   # kinova_mjlab_reaching (uv project)
src/kinova_mjlab_reaching/
    robots/gen3_lite.py                    # mjlab EntityCfg wrapping the validated MJCF
    tasks/reaching/                        # M2: reach-with-avoidance task (env/MDP/PPO cfg)
    tasks/tea_table/                       # tea-table adaptation: reuses the reaching env,
                                            #   overrides the target via a grasp registry
    deploy/policy_loader.py                # shared checkpoint-loading path (see below)
scripts/                                   # training/evaluation/visualization/model-validation
                                            #   entry points, all run via `uv run python scripts/...`
assets/kinova_gen3_lite/                   # URDF + generated MJCF + meshes
checkpoints/<run>/model_<n>.pt             # trained policies (gitignored except force-added ones)
docs/                                      # robot-model validation notes referenced above

package.xml, CMakeLists.txt                # kinova_gen3_lite_rl_bridge (ROS 2 ament_cmake package)
kinova_gen3_lite_rl_bridge/                # the ROS node + its pure-Python internals
launch/rl_reach.launch.py                  # brings up the ROS node + the uv policy server together
config/safety_limits.yaml, rl_bridge_params.yaml
test/                                      # pytest unit tests + ament lint tests
```

## Training / evaluating the policy (`kinova_mjlab_reaching`)

Everything here runs via `uv run` from the repo root (or wherever this
project's `pyproject.toml` lives) — do not run these with system Python.

```bash
# Train the reaching-with-avoidance baseline (runbook §27)
uv run python scripts/train_reaching.py                        # default: 256 envs, 1000 iters
uv run python scripts/train_reaching.py --max-iterations 20    # quick smoke run

# Watch a run live in the browser (mjlab's Viser viewer), hot-swapping checkpoints as they save
uv run python scripts/watch_training.py
uv run python scripts/watch_training.py --checkpoint-dir checkpoints/2026-08-25_22-35-31

# Held-out evaluation of a frozen checkpoint (success/collision breakdown — runbook §28)
uv run python scripts/evaluate_reaching.py --checkpoint checkpoints/<run>/model_<n>.pt

# Tea-table variant: same env/policy, target pinned to a grasp-registry pregrasp pose
uv run python scripts/watch_tea_table_reach.py
uv run python scripts/evaluate_tea_table_reach.py --checkpoint checkpoints/<run>/model_<n>.pt
```

`scripts/model_validation/` and `scripts/model_conversion/` hold the
one-off tools used to derive and validate the MJCF asset from the real
robot's URDF (see `docs/ros_reference_model.md`, `docs/fk_validation.md`,
`docs/joint_mapping.md` for what each check found).

### Observation / action spec (frozen, do not change without retraining)

The reaching task's policy (`src/kinova_mjlab_reaching/tasks/reaching/reach_env_cfg.py`)
uses a 21-dim observation and 6-dim action, in this exact order — the ROS
bridge below depends on it bit-for-bit:

- **Observation** (float32×21): `joint_pos_rel`(6, joints 1–6, home=0) ·
  `joint_vel_rel`(6) · `ee_to_target`(3, vector from end effector to
  target, in the robot base frame) · `last_action`(6, zeros on episode
  start).
- **Action** (float32×6): `target_q = action` directly (radians,
  joints 1–6, `scale=1.0`, zero default offset) — the policy outputs an
  absolute joint-position target, not a delta and not a torque.

## Real-hardware deployment (`kinova_gen3_lite_rl_bridge`)

A ROS 2 (Jazzy, `ament_cmake`) package that runs a trained checkpoint
against the **real** arm's `/joint_states` and TF, through
`joint_trajectory_controller` (from `ros2_kortex`/`kortex_bringup`).
Scoped to REACH-only for now (see the plan/runbook for the deferred
ALIGN/DESCEND/CLOSE grasp sequence).

### Architecture

```
 ROS 2 (Python 3.12)                              uv (Python 3.13)
┌─────────────────────────────┐   UDS socket    ┌───────────────────────┐
│ policy_bridge_node           │◄───────────────►│ scripts/policy_server │
│  observation_builder.py      │  ipc_protocol   │  policy_loader.py     │
│  safety_clamp.py             │   (PING/INFER)  │  (torch/mjlab/rsl_rl) │
│  policy_client.py            │                 └───────────────────────┘
└──────────────┬────────────────┘
               │ /joint_states, TF (base_link→tool_frame),
               │ /joint_trajectory_controller/joint_trajectory
               ▼
        real Kinova Gen3 Lite (via ros2_kortex / kortex_bringup)
```

- **`policy_bridge_node.py`** — the only ROS node. Builds the 21-dim
  observation from `/joint_states` + a TF lookup, sends it to the policy
  server, safety-clamps the returned action, and publishes a
  single-point `JointTrajectory`. Runtime-selectable `mode` parameter
  gates how far it goes:
  - `observe_only` — logs observations, never calls the policy server,
    never publishes. Validates joint-state/TF wiring in isolation.
  - `inference_only` — also runs inference and logs
    `(obs, raw_action, would-be-clamped-command)`, never publishes.
  - `live` — publishes the safety-clamped command to the real arm.
- **`observation_builder.py`**, **`safety_clamp.py`**, **`policy_client.py`**
  — pure Python, no `rclpy` import, unit-tested independently
  (`test/test_observation_builder.py`, `test/test_safety_clamp.py`).
- **`ipc_protocol.py`** — the wire format shared by both processes:
  fixed-size struct-packed messages (`PING`/`PONG`, `INFER_REQ`/`INFER_RESP`)
  over one persistent Unix-domain-socket connection. No JSON — the
  observation/action dimensions are fixed by the trained checkpoint.
- **`scripts/policy_server.py`** (uv side, zero ROS imports) — loads a
  checkpoint via `kinova_mjlab_reaching.deploy.policy_loader.load_reaching_policy()`
  (the same loading path `evaluate_reaching.py` uses) and serves
  inference requests over that socket.
- **`safety_clamp.py`** sits strictly downstream of the policy's raw
  output: it never touches the trained policy's `action_scale`/PD
  gains/decimation (retuning those post-hoc would mismatch
  train/inference dynamics — see the runbook), it only rate- and
  bound-limits the *published* command. `config/safety_limits.yaml`
  ships conservative defaults for first-contact testing.

### Build and test

```bash
cd ~/ros2_ws
colcon build --packages-select kinova_gen3_lite_rl_bridge
colcon test --packages-select kinova_gen3_lite_rl_bridge
colcon test-result --verbose --test-result-base build/kinova_gen3_lite_rl_bridge
```

### Running it

```bash
# 1. Bring up the real (or fake) hardware driver separately, e.g.:
#    (robot_ip is required even for fake hardware — it's just never used)
ros2 launch kortex_bringup gen3_lite.launch.py use_fake_hardware:=true robot_ip:=192.168.1.10

# 2. Stage 1 — wiring only, no policy server, no motion:
ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \
    mode:=observe_only start_policy_server:=false

# 3. Stage 2 — inference wiring, still zero motion:
ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \
    mode:=inference_only target_xyz:="[0.35, 0.05, 0.28]"

# 4. Stage 3 — first live motion (real robot, e-stop in hand):
ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \
    mode:=live target_xyz:="[0.35, 0.05, 0.28]"
```

`mode` and `target_xyz` are also changeable at runtime via
`ros2 param set /policy_bridge_node <name> <value>`, so an operator can
step through the staged rollout without relaunching. **Do not skip
stages** — see the runbook (§33) and the project plan for the full
staged-rollout checklist and why each step exists. Before switching a
launch from `live` to any other target or re-launching after a fault,
re-home the arm first (e.g. `ros2 run gen3-lite-ros2 joint_playback
<home-waypoint-file>`, all-zero joints, a generous duration — a single
5s point-to-point move across a large joint swing can itself exceed
`joint_limits.yaml`'s velocity cap at the trajectory's peak spline
velocity, ~1.875× the naive average, and trip a real Kinova joint fault).

#### Targeting a tea-table object by name

Instead of a raw `target_xyz`, `object_id`/`env_id` resolve a real
pregrasp target from `config/tea_table_objects.yaml` +
`tea_table_grasps.yaml` (via `scripts/print_pregrasp_target.py`, run
under `uv` at launch time — this is launch-time-only resolution, not a
runtime node parameter like `target_xyz` is):

```bash
ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \
    mode:=inference_only object_id:=kettle env_id:=env_a
```

`object_id` is `kettle`, `mug`, or `infuser`; `env_id` is `env_a` or
`env_b` (the two measured table layouts). Note the object *positions* in
`tea_table_objects.yaml` are real measurements, but the grasp *offset*
relative to each object (`tea_table_grasps.yaml`) is still a placeholder
tuned only against sim collision-testing — treat a resolved object target
with the same staged caution as any new target, not as a validated grasp
point.

### Shared autonomy (`shared_autonomy_node`)

A sibling to `policy_bridge_node` — not a replacement — that maintains a
belief over *all* of one env's tea-table objects at once (instead of one
fixed target) by comparing a human's live joystick input against what
each object's policy would do, then blends the assistance action with the
human's input. Modeled on
[`ada_assistance_policy`](https://github.com/personalrobotics/ada_assistance_policy),
adapted for a trained RL policy instead of that project's analytic Huber
cost function — see the project plan's "Milestone 2" section for the full
algorithm and why it works in joint-delta space rather than Cartesian
twist.

**Requires a joystick publishing `/joy`** (e.g. `ros2 run joy joy_node`)
**and must not run alongside `gen3-lite-ros2/joy_teleop.py`** — that tool
drives `twist_controller`, this node drives `joint_trajectory_controller`,
and ros2_control only runs one of those at a time. This node reads `/joy`
directly itself.

```bash
ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=observe_only start_policy_server:=false   # Stage 1: wiring only

ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=inference_only env_id:=env_a               # Stage 2: watch belief shift, no motion

ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=live env_id:=env_a                         # Stage 3: real motion, e-stop in hand
```

Only the left/right sticks jog joints 1–4 directly (`jog_scale_rad_s`,
default 0.3 rad/s); joints 5/6 get no direct joystick input in this first
version (trigger axes are single-ended on most controllers, not
well-suited to bidirectional jogging) — their motion comes entirely from
the blended policy suggestion. `temperature` (default 3.0) controls how
many consistent cycles of jogging it takes to concentrate belief on the
matching goal — evidence is cosine similarity between the human's jog
direction and each goal's policy direction (direction-only, deliberately
not raw distance: an undertrained policy's action magnitude varies goal
to goal for reasons unrelated to direction, and would otherwise swamp
whether the directions actually agree).

### Why two Python environments, concretely

- `policy_bridge_node` and everything under `kinova_gen3_lite_rl_bridge/`
  runs under system Python 3.12 (installed via `colcon build` /
  `ros2 run`) — it only needs `rclpy`, `numpy`, and stdlib.
- `scripts/policy_server.py` runs under `uv run` from this repo's `uv`
  environment (Python 3.13) — it needs `torch`/`mjlab`/`rsl_rl`/`mujoco`,
  none of which are installed for system Python.
- `launch/rl_reach.launch.py` starts both together, launching the policy
  server as `uv run python scripts/policy_server.py ...` with `cwd` set
  to this repo's source checkout (`uv` operates on the checkout, not a
  colcon install space — see the `rl_project_dir` launch argument).
