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
each goal's action source would do, then blends the assistance action
with the human's input. Modeled on
[`ada_assistance_policy`](https://github.com/personalrobotics/ada_assistance_policy)
— see the project plan's "Milestone 2" section for the full algorithm and
why it works in joint-delta space rather than Cartesian twist.

**Requires a joystick publishing `/joy`** (e.g. `ros2 run joy joy_node`)
**and must not run alongside `gen3-lite-ros2/joy_teleop.py`** — that tool
drives `twist_controller`, this node drives `joint_trajectory_controller`,
and ros2_control only runs one of those at a time. This node reads `/joy`
directly itself.

Each goal's suggested action comes from one of two interchangeable
backends, chosen by `action_source`:
- **`potential_field`** (default) — an analytic Jacobian-based attractor
  toward each goal (`src/kinova_mjlab_reaching/deploy/potential_field.py`,
  served by `scripts/potential_field_server.py`, using `kinpy` — the same
  library already validated for FK in `docs/fk_validation.md`). No
  trained weights, no checkpoint-loading wait, and — unlike the RL
  policy — well-defined from *any* starting joint configuration, not just
  ones near what a checkpoint happened to be trained on. See the project
  plan's "Milestone 3" section for the full math (damped least-squares
  pseudoinverse of the position-only Jacobian, robust near singularities).
- **`policy`** — the trained RL policy from Milestone 2, unmodified and
  still fully available; pass `action_source:=policy
  start_policy_server:=true start_potential_field_server:=false`.

```bash
ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=observe_only start_potential_field_server:=false   # Stage 1: wiring only

ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=inference_only env_id:=env_a               # Stage 2: watch belief shift, no motion

ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \
    mode:=live env_id:=env_a                         # Stage 3: real motion, e-stop in hand
```

The human's jog input similarly picks between two sources, `jog_mode`:
- **`cartesian`** (default) — the left/right sticks move the end effector
  in X/Y/Z (base frame), converted to a joint delta via the same Jacobian
  machinery as the potential field (`jog_scale_m_s`, default 0.15 m/s —
  the same value `joy_teleop.py` already uses as `LINEAR_SCALE`).
  `left_stick_v`→X, `left_stick_h`→Y, `right_stick_v`→Z, matching
  `joy_teleop.py`'s own TRANSLATE-mode convention; `right_stick_h` stays
  unused (no orientation jogging). Needs `potential_field_server` running
  for its Jacobian *even when* `action_source:=policy`.
- **`joint`** — sticks map straight to joints 1–4 (`jog_scale_rad_s`,
  default 0.3 rad/s); joints 5/6 get no direct input (trigger axes are
  single-ended on most controllers, not well-suited to bidirectional
  jogging). The original Milestone 2 mapping — unintuitive for "move in
  3D" since the same stick push means something different depending on
  the arm's current configuration, but still available:
  `jog_mode:=joint`.

`temperature` (default 3.0) controls how many consistent cycles of
jogging it takes to concentrate belief on the matching goal — evidence is
cosine similarity between the human's jog direction and each goal's
suggested direction (direction-only, deliberately not raw distance — an
undertrained RL policy's action magnitude in particular varies goal to
goal for reasons unrelated to direction, and would otherwise swamp
whether the directions actually agree). `assist_step_rad` (default 0.03,
matching `safety.max_step_rad`) normalizes each goal's suggested delta to
this fixed magnitude before blending with the human's input — without it,
a goal delta whose raw magnitude happens to be much larger than the
human's deliberately tiny jog (a potential field far from its target,
especially) dominates the final safety-clamped command regardless of
belief, making the joystick seem to do nothing; found via real hardware
testing. The final command is a straight average of the human's delta
and the belief-weighted assistance direction (`(human_delta +
assistance) / 2`) — not an additive sum — so each side always counts for
half regardless of belief confidence; a full-strength joystick push
alone now moves the arm at roughly half the rate it would with
assistance disabled entirely, which is a real tradeoff to keep in mind
when retuning `jog_scale_rad_s`/`jog_scale_m_s`/`assist_step_rad`.
`k_p` (default 2.0) and `damping_lambda` (default 0.05) tune the
potential field's gain and singularity-robustness, used whenever
`action_source:=potential_field` or `jog_mode:=cartesian`.

**Gripper**: press the `A` button (`gripper_toggle.py`'s `TOGGLE_BUTTON`)
to toggle the gripper open/closed, via the same `GripperCommand` action
and position values (`0.0` open / `0.8` close)
`gen3-lite-ros2/src/joy_teleop.py` already uses for its own LT/RT
triggers — both teleop tools drive the same
`gen3_lite_2f_gripper_controller`, which stays active alongside
`joint_trajectory_controller` (no conflict, unlike `twist_controller`).
Edge-triggered (one toggle per physical press, not per cycle held down)
and gated on `mode:=live` like everything else this node actuates — a
press in `observe_only`/`inference_only` is logged and ignored.

**Wrist rotation**: hold `LB`/`RB` to continuously rotate the wrist
(`joint_6`) CCW/CW (`wrist_jog_scale_rad_s`, default 0.3 rad/s) — the
gripper is rigidly mounted, so this is what "rotating the gripper" means.
Independent of `jog_mode`: works at the same time as either Cartesian or
joint-space stick jogging (translate with the sticks, rotate the wrist
with the shoulder buttons simultaneously). Holding both LB and RB cancels
to no rotation. Unlike the gripper toggle, this has no separate actuation
of its own — it's just another component of `human_delta`, so it's
already safety-clamped and respects `mode` the same way the rest of the
blended command does.

### Study control panel (`study_control_panel`)

A small browser-based control panel for running study sessions without a
terminal: Home button, participant ID field, env A/B selector, Start/Stop
Study. Built on Tornado (`gen3-lite-ros2/src/joy_web_bridge.py`'s existing
pattern — already a workspace dependency; no `rclpy` needed here, since
every action is just triggering a `ros2` CLI subprocess, not publishing a
live topic) and modeled on `investigating_peer_real_robot/session_gui.py`'s
(on `main`) subprocess-lifecycle pattern (SIGINT-first stop, `Popen` held
as state).

```bash
ros2 run kinova_gen3_lite_rl_bridge study_control_panel
# then open http://localhost:8080/ in a browser
```

**The real safety interlocks are server-side**, not just disabled
buttons: only one study can run at a time; homing is rejected while a
study is running; Start Study is rejected unless the robot has been
homed at least once this server session (`has_homed`) and unless a
numeric participant ID and a valid `env_id` were given. The panel
computes nothing new about robot motion — it only shells out to
already-validated commands: `ros2 run gen3-lite-ros2 joint_playback
config/home_pose.txt` for Home, and `ros2 launch
kinova_gen3_lite_rl_bridge shared_autonomy.launch.py mode:=live
env_id:=<selected>` for Start Study. **Does not start `kortex_bringup`**
— hardware bringup stays a separate manual step, same as every other
launch file in this project.

Starting a study creates one session directory,
`~/kinova_study_sessions/<user_id>_<timestamp>/`, and appends a line to
`sessions.csv` there for human-readable bookkeeping. Inside it, three
recordings start together:
- `ros2 bag record` (topics: `/joint_states`, `/joy`,
  `/joint_trajectory_controller/joint_trajectory` — arm state, human
  input, and the actual blended commands sent) into that session
  directory's `rosbag/` subdirectory.
- `pw-record` into that session directory's `audio.wav` — same intent
  as `investigating_peer_real_robot/session_gui.py`'s (on `main`) audio
  recording, but via PipeWire's own tool rather than raw `arecord -D
  ...`: on a PipeWire-managed system (this one included), PipeWire holds
  the ALSA hardware nodes itself, so a direct `arecord -D
  hw:.../plughw:.../sysdefault:...` open collides with "Device or
  resource busy" for anything but PipeWire's own `default` alias — this
  was hit and confirmed empirically. `pw-record --target <node.name>`
  talks to PipeWire's graph instead of grabbing hardware directly, so it
  doesn't have that problem. Best-effort: if `pw-record` itself can't be
  launched, the study still starts (the panel surfaces the failure via
  `/status`'s `audio_start_error` rather than blocking arm control or the
  bag on a missing microphone) — but note a *bad target name* doesn't
  fail loudly this way (pw-record degrades to recording silence rather
  than exiting), so `pw-record`'s own stderr is kept in
  `<session_dir>/audio_stderr.log` for after-the-fact debugging instead
  of being discarded.
- `gst-launch-1.0 -e v4l2src device=<node> ! videoconvert ! x264enc ! mp4mux
  ! filesink ...` into that session directory's `video.mp4`. GStreamer
  rather than `ffmpeg`/`v4l2-ctl` because only `gstreamer1.0-tools` and
  its plugin sets were already present in this workspace's environment;
  the `-e` flag matters specifically because Stop Study's SIGINT-first
  sequence needs the pipeline to push EOS through the muxer on interrupt
  so `video.mp4`'s trailer gets written — without it, an mp4 killed
  mid-recording can end up truncated/unseekable (confirmed the finalized
  file is seekable and valid via `gst-discoverer-1.0` after a SIGINT
  round trip). Same best-effort philosophy as audio: no capture-capable
  camera found or the launch itself fails, and the study still starts
  (surfaced via `/status`'s `video_start_error`; `gst-launch-1.0`'s own
  stderr goes to `<session_dir>/video_stderr.log`).

Stop Study sends SIGINT to the launch, the bag-record process, and the
audio/video-record processes (lets the bag/audio/video files finalize
cleanly), falling back to terminate/kill if any doesn't exit within 10s.

The page includes audio-device and camera dropdowns. The audio one is
populated from `GET /audio_devices` (parses `pw-dump`'s real
`Audio/Source` nodes — sound card and USB mic included, not just
`default`) and preselected to the server's `--audio-device` default; the
camera one is populated from `GET /video_devices` (parses `udevadm`
properties for every `/dev/videoN` node, keeping only the ones reporting
the `capture` capability — a single physical webcam typically exposes
more than one `/dev/videoN` node, and only one of them is the real
capture device). Both are overridable per session without a restart.

Useful flags: `--http-port` (default 8080), `--sessions-dir` (default
`~/kinova_study_sessions`), `--rl-project-dir` (default this checkout's
path — used to locate `config/home_pose.txt`), `--audio-device` (default
`default`, a PipeWire node.name, or `default` meaning "let PipeWire
auto-pick"; shown as the dropdown's preselected value), `--video-device`
(default empty string, a `/dev/videoN` path, or empty meaning "auto-pick
the first capture-capable camera udev reports"). Requires
`pw-record`/`pw-dump` (Debian/Ubuntu: `pipewire-bin`), `udevadm`
(Debian/Ubuntu: `udev`), and `gst-launch-1.0` with the v4l2/x264/mp4
plugins (Debian/Ubuntu: `gstreamer1.0-tools`,
`gstreamer1.0-plugins-{base,good,ugly}`) — all already declared as
`exec_depend`s.

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
