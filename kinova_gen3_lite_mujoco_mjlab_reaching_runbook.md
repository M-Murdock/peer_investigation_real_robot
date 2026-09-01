# Kinova Gen3 Lite — MuJoCo / MJLab RL Reaching Project Runbook

> **Project goal:** Train a goal-conditioned reinforcement learning policy for a Kinova Gen3 Lite arm to reach arbitrary reachable 3D end-effector target positions in MJLab/MuJoCo, then deploy the learned policy through ROS 2 and extend the system toward Shared Autonomy (SA).

> **Primary environment:** Ubuntu 24.04 + ROS 2 Jazzy + NVIDIA GPU  
> **Development strategy:** MuJoCo/MJLab for RL training; ROS 2 only enters after the simulation model and reaching policy are validated.

---

## 0. Agent operating instructions

This document is intended to be followed by a coding agent.

### Rules for the agent

1. **Work sequentially. Do not skip milestones.**
2. **Do not start PPO training until the standalone MuJoCo model passes all P0 validation checks.**
3. **Do not connect to or command the physical Kinova arm until the simulation policy passes P1/P2 evaluation.**
4. Do not modify or reinstall the existing ROS 2 installation, NVIDIA driver, or CUDA stack unless a concrete incompatibility is demonstrated.
5. Keep ROS 2 system Python separate from the RL Python environment.
6. Prefer a project-local `uv` environment for MuJoCo/MJLab.
7. Save every important generated file in version control.
8. Whenever a joint ordering, name, limit, sign, frame, or unit is assumed, verify it and record it.
9. Before changing model dynamics to fix RL behavior, determine whether the issue is:
   - model conversion,
   - actuator configuration,
   - reward design,
   - observation/action scaling,
   - PPO configuration.
10. At the end of each milestone, write a short validation report under `docs/validation/`.

### Stop conditions

Stop and fix the current stage before proceeding if any of the following occurs:

- MuJoCo cannot load the model without warnings/errors relevant to dynamics.
- Any arm joint rotates about the wrong axis.
- A joint has an unexpected range or order.
- The base is not fixed.
- Links explode, drift, collapse, or penetrate severely at initialization.
- End-effector FK disagrees materially with the ROS/Kinova model.
- An actuator command produces unstable or physically unreasonable motion.
- MJLab cannot reproduce the validated standalone MuJoCo model.
- The RL environment cannot pass zero-action and random-action sanity checks.

---

# 1. Final architecture

```text
                           TRAINING

Kinova ros2_kortex
       │
       ├── Xacro / URDF / meshes
       ↓
Validated Gen3 Lite MJCF
       │
       ↓
     MuJoCo
  model validation
       │
       ↓
      MJLab
       │
       ├── random initial joint state
       ├── random reachable XYZ target
       ├── observations
       ├── rewards
       └── joint-space actions
       │
       ↓
      PPO
       │
       ↓
 reaching_policy.pt


                         DEPLOYMENT

             /joint_states
                   │
                   ↓
             ROS 2 policy node
                   │
              policy.pt
                   │
                   ↓
           safe joint command
                   │
                   ↓
             ros2_control
                   │
                   ↓
         Physical Gen3 Lite


                     FUTURE SA EXTENSION

Human joystick / intent
          │
          ↓
  Shared Autonomy layer
          ↑
          │
 Autonomous reaching policy
          │
          ↓
      Safety filter
          │
          ↓
      Gen3 Lite
```

---

# 2. Priority map

| Priority | Stage | Main objective | Exit criterion |
|---|---|---|---|
| P0 | Environment | Linux + GPU + ROS + Python tooling healthy | All prerequisite checks pass |
| P0 | Robot source | Obtain official Gen3 Lite Xacro/URDF/meshes | Source files located and reproducible |
| P0 | URDF → MJCF | Build standalone MuJoCo model | Viewer loads model correctly |
| P0 | Model validation | Validate joints, limits, FK, dynamics, actuators | Validated controllable 6-DOF model |
| P1 | MJLab asset | Port validated model into MJLab | 1 environment runs correctly |
| P1 | Reaching MDP | Implement observations/actions/reward/reset | Zero/random agent sanity checks pass |
| P1 | PPO baseline | Learn arbitrary XYZ reaching | High success rate on held-out goals |
| P2 | Robustness | Curriculum + randomization + safety penalties | Stable performance across test distribution |
| P2 | ROS 2 deployment | Run policy from ROS state and command interface | Same policy works through ROS 2 simulation/mock |
| P2 | Sim-to-real | Conservative real-arm deployment | Safe real reaching demonstrated |
| P3 | Shared Autonomy | Add human input / blending / intent | SA baseline operational |
| P3/P4 | Research | Adaptive assistance + user study | Experimental comparison completed |

---

# 3. Local packages and environment setup

## 3.1 Do this on Ubuntu, not Windows

Use:

```text
Ubuntu 24.04
├── ROS 2 Jazzy
├── ros2_kortex
├── MuJoCo
├── MJLab
├── PyTorch / CUDA dependencies
└── NVIDIA GPU
```

Do not split the main project between Windows and Ubuntu.

---

## 3.2 System health check

Run:

```bash
lsb_release -a
uname -a

ros2 --help >/dev/null && echo "ROS2 OK"
echo "$ROS_DISTRO"

nvidia-smi

git --version
curl --version
```

Expected:

```text
Ubuntu: 24.04
ROS_DISTRO: jazzy
NVIDIA GPU visible in nvidia-smi
```

If `ROS_DISTRO` is empty:

```bash
source /opt/ros/jazzy/setup.bash
```

Then check again:

```bash
echo "$ROS_DISTRO"
```

### Important

Do **not** reinstall the NVIDIA driver if `nvidia-smi` works.

MJLab currently recommends:

- Linux
- NVIDIA GPU
- Python 3.10+
- CUDA 12.4+ recommended

The Python environment will be isolated from ROS.

---

## 3.3 Install basic Ubuntu packages

```bash
sudo apt update

sudo apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    libegl-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    ros-jazzy-xacro \
    liburdfdom-tools
```

Do not install `mujoco-py`.

Use the modern official `mujoco` package.

---

## 3.4 Install `uv`

MJLab's recommended project workflow uses `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload shell:

```bash
source ~/.bashrc
```

Verify:

```bash
uv --version
```

If `uv` is not found, try:

```bash
source "$HOME/.local/bin/env"
uv --version
```

---

# 4. Create the project workspace

Recommended layout:

```bash
mkdir -p ~/projects
cd ~/projects

uv init --package kinova_mjlab_reaching
cd kinova_mjlab_reaching
```

Create directories:

```bash
mkdir -p \
    assets/kinova_gen3_lite/meshes \
    scripts/model_conversion \
    scripts/model_validation \
    src/kinova_mjlab_reaching/robots \
    src/kinova_mjlab_reaching/tasks/reaching/mdp \
    src/kinova_mjlab_reaching/tasks/reaching/agents \
    src/kinova_mjlab_reaching/ros \
    checkpoints \
    docs/validation
```

Target structure:

```text
kinova_mjlab_reaching/
│
├── assets/
│   └── kinova_gen3_lite/
│       ├── source/
│       ├── meshes/
│       ├── gen3_lite.urdf
│       └── gen3_lite.xml
│
├── scripts/
│   ├── model_conversion/
│   └── model_validation/
│
├── src/kinova_mjlab_reaching/
│   ├── robots/
│   │   └── gen3_lite.py
│   │
│   ├── tasks/reaching/
│   │   ├── env_cfg.py
│   │   ├── mdp/
│   │   │   ├── observations.py
│   │   │   ├── actions.py
│   │   │   ├── rewards.py
│   │   │   ├── commands.py
│   │   │   ├── events.py
│   │   │   └── terminations.py
│   │   └── agents/
│   │       └── ppo_cfg.py
│   │
│   └── ros/
│       └── policy_node.py
│
├── checkpoints/
├── docs/validation/
└── pyproject.toml
```

---

# 5. Install MuJoCo first

For P0 model conversion/validation, start with standalone MuJoCo.

```bash
cd ~/projects/kinova_mjlab_reaching

uv add mujoco numpy
```

Verify:

```bash
uv run python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
```

Test the viewer:

```bash
uv run python -m mujoco.viewer
```

If a GUI window opens successfully, standalone MuJoCo is ready.

### Headless systems

For a machine without a display:

```bash
export MUJOCO_GL=egl
```

Do not make this permanent until EGL operation is verified.

---

# 6. Install MJLab

Once standalone MuJoCo works:

```bash
cd ~/projects/kinova_mjlab_reaching

uv add mjlab
```

Verify:

```bash
uv run demo
```

The MJLab demo should launch successfully.

Also verify CUDA/GPU visibility from the project environment:

```bash
uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
cuda available: True
gpu: <your NVIDIA GPU>
```

### Do not continue to P1 if CUDA is unavailable

MJLab training is intended for NVIDIA GPU execution.

---

# 7. Obtain official Kinova Gen3 Lite description

Official source:

```text
https://github.com/Kinovarobotics/ros2_kortex
```

Kinova's official `kortex_description` package contains URDF/Xacro, STL meshes, and robot configuration.

Recommended ROS workspace:

```bash
mkdir -p ~/workspace/ros2_kortex_ws/src
cd ~/workspace/ros2_kortex_ws/src

git clone -b jazzy https://github.com/Kinovarobotics/ros2_kortex.git
```

Import ROS dependencies:

```bash
cd ~/workspace/ros2_kortex_ws

source /opt/ros/jazzy/setup.bash

vcs import src \
    --skip-existing \
    --input src/ros2_kortex/ros2_kortex.jazzy.repos

vcs import src \
    --skip-existing \
    --input src/ros2_kortex/ros2_kortex-not-released.jazzy.repos
```

Install dependencies:

```bash
rosdep update

rosdep install \
    --ignore-src \
    --from-paths src \
    -y \
    -r
```

Build:

```bash
colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --parallel-workers 3
```

Source:

```bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
```

Check:

```bash
ros2 pkg prefix kortex_description
```

---

# 8. Locate Gen3 Lite files

Run:

```bash
cd ~/workspace/ros2_kortex_ws/src/ros2_kortex

find kortex_description -iname "*gen3_lite*"
```

Important official description entry:

```text
kortex_description/robots/gen3_lite_gen3_lite_2f.xacro
```

Also inspect:

```bash
find kortex_description/arms/gen3_lite -type f | sort
```

Record:

- Xacro files
- mesh directories
- joint definitions
- inertial definitions
- joint limits
- end-effector/tool frame

---

# 9. Validate official model in ROS before conversion

This establishes a trusted reference model.

Source:

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash
```

Use the official description visualization/launch mechanism available in the checked-out branch.

Before running a launch command, inspect available launch files:

```bash
ros2 pkg prefix kortex_description

find \
  "$(ros2 pkg prefix kortex_description)/share/kortex_description" \
  -maxdepth 3 \
  -type f \
  -name "*.launch.py"
```

Agent rule:

> Use the launch file and arguments from the **checked-out Jazzy branch**, not an old online tutorial.

Validate visually:

- base orientation
- link geometry
- 6 arm joints
- zero configuration
- end-effector frame

Create:

```text
docs/validation/ros_reference_model.md
```

Record the model source commit:

```bash
cd ~/workspace/ros2_kortex_ws/src/ros2_kortex
git rev-parse HEAD
```

---

# 10. Generate a pure URDF

## Important

Xacro is a ROS preprocessing format.

MuJoCo should receive a resolved URDF or MJCF, not unresolved Xacro.

Inspect the Xacro arguments first:

```bash
grep -R "<xacro:arg" \
    ~/workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description/robots
```

Do not guess arguments.

For the first reaching model:

> Prefer the 6-DOF arm without gripper if the official description structure allows it cleanly.

If an arm-only configuration is inconvenient, retaining the Gen3 Lite gripper initially is acceptable, but keep its joints passive/fixed until reaching works.

Generate:

```text
assets/kinova_gen3_lite/gen3_lite.urdf
```

Use the checked-out Xacro entry and its actual required arguments.

Example pattern only:

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspace/ros2_kortex_ws/install/setup.bash

xacro <KINOVA_XACRO_ENTRY> \
    <VALID_ARGUMENTS_FROM_SOURCE> \
    > ~/projects/kinova_mjlab_reaching/assets/kinova_gen3_lite/gen3_lite.urdf
```

Validate:

```bash
check_urdf \
  ~/projects/kinova_mjlab_reaching/assets/kinova_gen3_lite/gen3_lite.urdf
```

Inspect:

```bash
grep -n "<joint" assets/kinova_gen3_lite/gen3_lite.urdf
grep -n "<limit" assets/kinova_gen3_lite/gen3_lite.urdf
grep -n "<inertial" assets/kinova_gen3_lite/gen3_lite.urdf
grep -n "mesh filename" assets/kinova_gen3_lite/gen3_lite.urdf
```

---

# 11. Copy meshes and resolve ROS package URIs

MuJoCo does not use the ROS package resolver in the same way as ROS.

Typical URDF references may look like:

```xml
<mesh filename="package://kortex_description/.../mesh.stl"/>
```

Create a self-contained asset directory:

```text
assets/kinova_gen3_lite/
├── gen3_lite.urdf
├── gen3_lite.xml
└── meshes/
```

Copy only the meshes actually referenced by the generated URDF.

Rewrite mesh paths to project-relative paths.

Agent must create:

```text
scripts/model_conversion/resolve_mesh_paths.py
```

Requirements:

1. Parse URDF as XML.
2. Find all `<mesh filename="...">`.
3. Resolve `package://kortex_description/...`.
4. Copy referenced mesh into the asset directory while preserving a sensible subdirectory hierarchy.
5. Rewrite filenames to paths relative to the output URDF.
6. Do not silently ignore missing meshes.
7. Print a conversion report.

Output:

```text
assets/kinova_gen3_lite/gen3_lite_resolved.urdf
```

---

# 12. First direct MuJoCo URDF load

Create:

```text
scripts/model_validation/load_urdf.py
```

Minimal behavior:

```python
import mujoco

model = mujoco.MjModel.from_xml_path(
    "assets/kinova_gen3_lite/gen3_lite_resolved.urdf"
)

print("nq:", model.nq)
print("nv:", model.nv)
print("njnt:", model.njnt)
print("nbody:", model.nbody)
print("nu:", model.nu)
```

Run:

```bash
uv run python scripts/model_validation/load_urdf.py
```

Do not proceed until loading succeeds.

---

# 13. Convert/save as MJCF

Create:

```text
scripts/model_conversion/urdf_to_mjcf.py
```

Conceptual implementation:

```python
import mujoco

model = mujoco.MjModel.from_xml_path(
    "assets/kinova_gen3_lite/gen3_lite_resolved.urdf"
)

mujoco.mj_saveLastXML(
    "assets/kinova_gen3_lite/gen3_lite.xml",
    model,
)
```

Run:

```bash
uv run python scripts/model_conversion/urdf_to_mjcf.py
```

The generated MJCF is only a starting point.

---

# 14. P0 MuJoCo model validation

## Milestone M1

> Produce a controllable, kinematically consistent Gen3 Lite MuJoCo model.

Do not start MJLab task development before this milestone passes.

---

## 14.1 Viewer test

Create:

```text
scripts/model_validation/view_model.py
```

Load:

```text
assets/kinova_gen3_lite/gen3_lite.xml
```

Run:

```bash
uv run python scripts/model_validation/view_model.py
```

Check:

- correct model orientation
- fixed base
- meshes aligned
- no exploding links
- no unstable initial contacts
- sensible zero pose

---

## 14.2 Print all MuJoCo joints

Create:

```text
scripts/model_validation/inspect_joints.py
```

For each joint record:

- MuJoCo index
- name
- type
- axis
- range
- body
- qpos address
- velocity address

Save the output to:

```text
docs/validation/mujoco_joint_table.md
```

---

## 14.3 Create ROS ↔ MuJoCo joint mapping

Create:

```text
docs/validation/joint_mapping.md
```

Template:

| Index | ROS/Kinova joint | MuJoCo joint | Axis | Lower | Upper | Home | Units | Verified |
|---:|---|---|---|---:|---:|---:|---|---|
| 0 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD | rad | No |

All six rows must be verified before deployment work.

---

# 15. Add actuators

The converted model may not contain actuators appropriate for RL.

For the first reaching task, prefer one of these:

### Option A — joint position target

RL outputs normalized joint target increments:

```text
action ∈ [-1, 1]^6
       ↓
scaled Δq
       ↓
position actuator target
```

This is usually the easiest baseline.

### Option B — joint velocity command

```text
action ∈ [-1, 1]^6
       ↓
scaled qdot command
```

This may map more naturally to some shared-control formulations later.

### Recommendation

Start with:

```text
position target increments
```

unless the intended final SA controller specifically requires velocity actions.

Do not expose raw torque control in the first reaching baseline.

---

# 16. Actuator validation

Create:

```text
scripts/model_validation/test_actuators.py
```

Requirements:

1. Start from a safe neutral configuration.
2. Move exactly one joint at a time.
3. Use small commands.
4. Verify sign and axis visually.
5. Check limit behavior.
6. Ensure other joints remain stable.
7. Verify no uncontrolled oscillation.

Test sequence:

```text
joint 1 only
joint 2 only
joint 3 only
joint 4 only
joint 5 only
joint 6 only
```

Exit criterion:

> All six joints respond predictably and stably.

---

# 17. Add and validate end-effector site

Add a MuJoCo `site` at the intended control point.

Example concept:

```xml
<site
    name="ee_site"
    pos="0 0 0"
    size="0.01"/>
```

The exact parent body and offset must correspond to the Kinova reference frame used for deployment.

Access:

```python
ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ee_site",
)

ee_xyz = data.site_xpos[ee_id]
```

Do not define an arbitrary site and later call it the Kinova tool frame without validation.

---

# 18. Forward-kinematics cross-validation

This is a mandatory P0 check.

Choose at least:

- neutral/home configuration
- 3–5 safe random joint configurations

For each configuration:

```text
same q[6]
   │
   ├── ROS/Kinova reference FK → p_ros, R_ros
   │
   └── MuJoCo                 → p_mj,  R_mj
```

Compare:

```text
position error
orientation error
```

Create:

```text
docs/validation/fk_validation.md
```

Suggested acceptance target for the model geometry:

```text
position mismatch: millimeter-scale
orientation mismatch: small numerical tolerance
```

If errors are centimeters or frames are rotated substantially, fix the frame/model before proceeding.

---

# 19. Check collision and inertial behavior

Inspect:

- inertial masses
- centers of mass
- inertia tensors
- collision meshes/geometries
- obvious self-collision problems
- ground interaction
- damping

For initial reaching:

> Collision geometry may be simplified later for training speed, but only after the original model has been validated.

Do not tune physical parameters merely to make PPO easier.

---

# 20. M1 completion checklist

M1 is complete only when:

- [ ] Official Kinova description source is recorded.
- [ ] Pure URDF is reproducibly generated.
- [ ] All mesh paths are self-contained.
- [ ] MuJoCo loads the model.
- [ ] MJCF is stored in project assets.
- [ ] Base is fixed.
- [ ] Six arm joints are identified.
- [ ] Joint names/order are mapped to ROS.
- [ ] Joint axes are correct.
- [ ] Joint limits are correct.
- [ ] Actuators are stable.
- [ ] Every joint was individually tested.
- [ ] `ee_site` is defined.
- [ ] FK is cross-validated against ROS/Kinova.
- [ ] Validation report is committed.

Only now begin MJLab integration.

---

# 21. P1 — Port Gen3 Lite into MJLab

Create:

```text
src/kinova_mjlab_reaching/robots/gen3_lite.py
```

The MJLab robot asset must reference the validated MJCF.

Do not create a second independent robot model.

Goal:

```text
validated gen3_lite.xml
        ↓
MJLab robot configuration
        ↓
1 environment
```

First test:

```text
num_envs = 1
```

Then:

```text
1 → 16 → 256 → 1024
```

Do not immediately start at 4096.

---

# 22. P1 — Reaching MDP definition

## 22.1 Task

At episode reset:

1. sample safe initial joint configuration,
2. place robot,
3. sample reachable target XYZ,
4. reset velocities,
5. reset episode counters.

Policy objective:

```text
move end-effector to target
```

---

## 22.2 Observation baseline

Recommended first observation:

```text
q[6]
dq[6]
target_xyz[3]
ee_xyz[3]
```

Better representation after baseline:

```text
q[6]
dq[6]
target_xyz - ee_xyz [3]
```

Optional later:

- previous action
- EE velocity
- obstacle features
- human command
- goal belief

Normalize observations.

---

## 22.3 Action baseline

Recommended:

```text
action[6] ∈ [-1, 1]
```

Map to bounded joint increments:

```text
Δq = action_scale * action
q_target = q_current + Δq
```

Clamp to safe joint limits.

Alternative later:

```text
joint velocity action
```

---

# 23. Reachable goal sampling

Do not sample arbitrary XYZ from a rectangular box and assume all targets are reachable.

Start by generating targets from valid joint configurations:

```text
sample safe q_goal
      ↓
MuJoCo FK
      ↓
target_xyz = EE(q_goal)
```

Advantages:

- every target is kinematically reachable,
- no inverse-kinematics filter is needed,
- simple curriculum control.

Later expand toward explicit workspace sampling.

---

# 24. Reward baseline

Keep the first reward simple.

Define:

```text
d_t = ||p_ee - p_goal||
```

Possible baseline:

```text
reward =
    progress_reward
  + success_bonus
  - action_penalty
```

where:

```text
progress_reward = k * (d_previous - d_current)
```

Success:

```text
d_t < 0.03 m
```

Example:

```text
success bonus = +10
```

Do not add many reward terms before the baseline is understood.

Later additions:

- action rate penalty
- joint velocity penalty
- collision penalty
- joint-limit penalty
- smoothness
- orientation objective

---

# 25. Episode termination

Terminate on:

### Success

```text
EE-target distance < threshold
```

### Timeout

Example:

```text
2–5 s simulated task time
```

### Failure later

- severe self collision
- forbidden workspace
- joint-limit violation
- unsafe state

---

# 26. Mandatory environment sanity checks

Before PPO:

```bash
uv run play <REACHING_TASK_ID> --agent zero
```

and:

```bash
uv run play <REACHING_TASK_ID> --agent random
```

The exact command depends on the task registration, but MJLab provides zero/random agents for this purpose.

Check:

### Zero action

- robot remains stable
- reward finite
- observations finite
- no unexplained drift
- reset works repeatedly

### Random action

- bounded movement
- no NaNs
- no exploding joints
- action limits respected
- episodes terminate/reset correctly

Do not start PPO if these fail.

---

# 27. PPO baseline training

Start conservatively:

```text
num_envs = 256
```

Then scale:

```text
256 → 1024
```

Only increase further if GPU utilization and simulation stability justify it.

Train the simplest reachable-goal version first.

Track:

- episodic reward
- success rate
- final EE-goal distance
- episode length
- action magnitude
- joint velocity
- simulation throughput
- GPU memory

---

# 28. M2 success criterion

Do not call the reaching task complete because reward increases.

Use a held-out evaluation set.

Recommended metrics:

```text
success threshold: 3 cm
held-out target success rate
median final position error
90th percentile final position error
time-to-target
collision rate
```

Example research baseline target:

```text
>90% success on held-out reachable targets
```

The exact threshold can be refined later.

---

# 29. P2 — Robustness after baseline success

Only after the basic policy converges:

### Initial-state randomization

```text
q0 ~ safe distribution
```

### Target curriculum

Easy:

```text
target near current EE
```

Later:

```text
full reachable workspace
```

### Dynamics randomization

Later and only where needed:

- damping
- friction
- actuator response
- control delay
- encoder/state noise
- action latency

### Safety terms

- soft joint-limit penalties
- collision penalties
- action-rate penalty
- velocity limits

---

# 30. Keep ROS 2 out of the RL training loop

Recommended architecture:

```text
MJLab training:
GPU tensors
    ↓
policy
    ↓
MuJoCo Warp

NO ROS 2 per simulation step
```

Do not build:

```text
MJLab → ROS topic → policy → ROS topic → MJLab
```

for training.

ROS enters at deployment/integration.

---

# 31. P2 — ROS 2 policy deployment

After M2:

Create:

```text
src/kinova_mjlab_reaching/ros/policy_node.py
```

Concept:

```text
/joint_states
      ↓
 joint mapping
      ↓
 observation builder
      ↓
   policy.pt
      ↓
 safety clamp
      ↓
 ROS command interface
      ↓
 ros2_control / Kinova
```

Critical deployment equivalences:

```text
joint order
joint sign
units
home pose
control frequency
action scale
EE frame
joint limits
```

---

# 32. Test ROS deployment before real hardware

Use one or more:

1. Kinova fake hardware
2. ROS-side simulation
3. MuJoCo + ROS integration if desired
4. Offline recorded joint-state replay

Verify:

- observation vector is identical to training definition,
- action post-processing is identical,
- timing is bounded,
- safety clamps work.

---

# 33. P2 — Conservative Sim-to-Real

First physical tests should **not** be full random-workspace RL demonstrations.

Recommended progression:

```text
1. neutral pose only
2. policy inference with command output disabled
3. inspect predicted actions
4. extremely small action scale
5. one nearby target
6. several nearby targets
7. gradually expand workspace
```

Use robot-side limits and an external emergency stop.

Do not bypass Kinova safety mechanisms.

---

# 34. Real-robot parameters to record

No large real dataset is needed at P0/P1.

Before deployment, record:

```text
ROS joint names
joint ordering
joint units
safe home pose
joint limits
velocity limits
control interface
command rate
state update rate
tool / EE frame
gripper configuration
```

Only perform deeper system identification if sim-to-real testing shows a concrete mismatch.

Possible later identification:

- control latency
- velocity response
- actuator lag
- damping
- friction
- payload effects

---

# 35. P3 — Shared Autonomy extension

The reaching policy becomes the autonomous skill:

```text
π_R(s, g) → a_R
```

Add human input:

```text
u_H
```

Start with fixed blending:

```text
a = (1 - α) u_H + α a_R
```

---

## Stage SA-1 — Fixed blending

Compare:

```text
human only
vs
autonomy only
vs
fixed shared control
```

---

## Stage SA-2 — Goal uncertainty

Multiple candidate goals:

```text
P(g | human input, state)
```

Use:

```text
human joystick history
robot state
candidate target positions
```

---

## Stage SA-3 — Adaptive assistance

Learn or design:

```text
α = f(
    goal confidence,
    task state,
    human behavior,
    safety margin
)
```

---

## Stage SA-4 — Simulated human for scaling

Create simulated human policies with:

- skill variation
- Gaussian joystick noise
- delay
- hesitation
- incorrect goal movement
- correction behavior

MJLab can then run many human/robot interactions in parallel.

Example:

```text
Env 1: skilled human
Env 2: noisy human
Env 3: delayed human
Env 4: uncertain human
...
```

---

## Stage SA-5 — Safety layer

Potential architecture:

```text
human action
      │
      ↓
SA / RL assistance
      │
      ↓
safety filter
      │
      ↓
robot command
```

Possible research directions:

- control barrier functions
- constrained optimization
- collision avoidance
- human-aware constraints

---

# 36. Suggested Git milestones

## Commit/Milestone M0

```text
Environment setup + MuJoCo/MJLab installation verified
```

## M1

```text
Validated Kinova Gen3 Lite MuJoCo MJCF
```

## M1.1

```text
Actuator + EE site + FK validation
```

## M1.2

```text
Gen3 Lite MJLab asset
```

## M2

```text
Goal-conditioned reaching environment
```

## M2.1

```text
PPO baseline convergence
```

## M2.2

```text
Robustness + held-out evaluation
```

## M3

```text
ROS2 policy deployment
```

## M3.1

```text
Conservative physical Gen3 Lite reaching
```

## M4

```text
Shared autonomy fixed-blending baseline
```

## M4.1

```text
Adaptive / learning-based SA
```

---

# 37. What the agent should do first

The immediate task is **not PPO**.

The first execution sequence is:

```text
[1] Verify Ubuntu / Jazzy / NVIDIA
          ↓
[2] Install uv + system dependencies
          ↓
[3] Create kinova_mjlab_reaching project
          ↓
[4] Install standalone MuJoCo
          ↓
[5] Verify MuJoCo viewer
          ↓
[6] Install MJLab
          ↓
[7] Verify MJLab demo + CUDA
          ↓
[8] Clone/build official ros2_kortex Jazzy
          ↓
[9] Locate Gen3 Lite Xacro + meshes
          ↓
[10] Generate pure URDF
          ↓
[11] Resolve mesh paths
          ↓
[12] Load URDF in MuJoCo
          ↓
[13] Save/clean MJCF
          ↓
[14] Validate 6 joints
          ↓
[15] Add/test actuators
          ↓
[16] Add EE site
          ↓
[17] ROS↔MuJoCo FK comparison
          ↓
       MILESTONE M1
```

The agent should stop here and report validation results before implementing the reaching MDP.

---

# 38. Immediate success definition

The first concrete objective of this project is:

> **Create a validated, controllable, kinematically consistent Kinova Gen3 Lite MJCF model that reproduces the official Kinova/ROS joint structure and end-effector kinematics in standalone MuJoCo.**

Only after that objective is achieved should MJLab PPO training begin.

---

# 39. Useful verification commands

## ROS

```bash
echo "$ROS_DISTRO"
ros2 pkg prefix kortex_description
```

## GPU

```bash
nvidia-smi
```

## uv

```bash
uv --version
```

## MuJoCo

```bash
uv run python -c "import mujoco; print(mujoco.__version__)"
```

## MJLab

```bash
uv run demo
```

## PyTorch GPU

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Kinova source

```bash
cd ~/workspace/ros2_kortex_ws/src/ros2_kortex
git status
git rev-parse HEAD
find kortex_description -iname "*gen3_lite*"
```

## URDF

```bash
check_urdf assets/kinova_gen3_lite/gen3_lite.urdf
```

---

# 40. Packages summary

## Ubuntu packages

```bash
sudo apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    libegl-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    ros-jazzy-xacro \
    liburdfdom-tools
```

## Python / RL project

Installed inside the `uv` project:

```bash
uv add mujoco numpy
uv add mjlab
```

MJLab brings its own Python-side training dependencies through its package dependency graph. Do not manually install random CUDA/JAX/PyTorch versions before checking whether the standard MJLab installation works.

## ROS workspace

Use:

```text
Kinovarobotics/ros2_kortex — jazzy branch
```

and let:

```bash
rosdep install --ignore-src --from-paths src -y -r
```

resolve ROS package dependencies.

---

# 41. Official references

MJLab:

```text
https://mujocolab.github.io/mjlab/
https://mujocolab.github.io/mjlab/v1.1.1/source/installation.html
https://github.com/mujocolab/mjlab
```

MuJoCo:

```text
https://mujoco.readthedocs.io/en/latest/python.html
https://github.com/google-deepmind/mujoco
```

Kinova ROS 2:

```text
https://github.com/Kinovarobotics/ros2_kortex
https://github.com/Kinovarobotics/ros2_kortex/tree/jazzy/kortex_description
```

---

# 42. Research roadmap in one line

```text
Validated MuJoCo Robot
    → MJLab Goal-Conditioned Reaching
    → Robust PPO Policy
    → ROS2 Deployment
    → Sim-to-Real
    → Human Input
    → Shared Autonomy
    → Adaptive Assistance / User Study
```
