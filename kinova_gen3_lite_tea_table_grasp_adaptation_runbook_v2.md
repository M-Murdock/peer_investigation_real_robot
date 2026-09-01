# Kinova Gen3 Lite — MuJoCo / MJLab RL Reaching Project Runbook

> **Project goal:** Train a goal-conditioned reinforcement learning policy for a Kinova Gen3 Lite arm to reach arbitrary reachable 3D end-effector target positions in MJLab/MuJoCo, then deploy the learned policy through ROS 2 and extend the system toward Shared Autonomy (SA).

> **Primary environment:** Ubuntu 24.04 + ROS 2 Jazzy + NVIDIA GPU
> **Development strategy:** MuJoCo/MJLab for RL training; ROS 2 only enters after the simulation model and reaching policy are validated.

> **[MODIFIED PROJECT STATUS]** A goal-conditioned reaching policy is already trained and can move the Kinova Gen3 Lite end-effector, using FK-derived end-effector position, to arbitrary reachable XYZ goals while avoiding obstacles present in its training distribution. The objective of this updated runbook is **not** to train six independent neural-network policies. It is to preserve one trained goal-conditioned policy checkpoint and adapt/query it as six task instances: two tea-table environments × three target containers.

# Task background — tea-table adaptation

> **[NEW] Experimental task context**

The physical task contains a tea table and three target containers:

- `kettle`
- `mug`
- `infuser`

Two environments are required:

```text
Env A: kettle / mug / infuser at configuration A
Env B: kettle / mug / infuser at configuration B
```

The three container poses differ between Env A and Env B. Their real-world positions may initially be obtained by manual measurement and expressed in the Kinova robot base frame. The same poses are then reproduced in MuJoCo/MJLab. Perception is therefore **not required for the first deterministic baseline**.

The six autonomous reaching skills are conceptual task instances of one policy:

```text
pi_theta(s, g_A_kettle)
pi_theta(s, g_A_mug)
pi_theta(s, g_A_infuser)
pi_theta(s, g_B_kettle)
pi_theta(s, g_B_mug)
pi_theta(s, g_B_infuser)
```

All six calls use the **same policy parameters `theta` and the same checkpoint**. Only the scene configuration and goal input differ.

The immediate engineering goal is:

> Use the existing trained goal-conditioned reaching checkpoint in two tea-table scenes, validate all six `(env_id, object_id)` goal instances, add the minimum robustness/domain-randomization needed for sim-to-real, and expose the resulting goal-conditioned skill through ROS 2.

The longer-term research goal remains Shared Autonomy, where the three candidate container goals can be evaluated by repeatedly querying the same autonomous policy with different goals.

---

# **[NEW] Grasp-skill scope and responsibility boundary**

The six tea-object skills are now defined as **pre-grasp reaching + deterministic local grasp execution**. The project does **not** ask RL to discover a grasp pose or to learn the complete grasp sequence.

For each `(env_id, object_id)` instance:

```text
known object 6D pose in robot-base frame
        ↓
fixed object-relative grasp transform
        ↓
compute nominal grasp pose
        ↓
compute pre-grasp pose above / before grasp pose
        │
        ├── position component    → goal-conditioned RL reaching
        │
        └── orientation component → deterministic pose alignment
                                       ↓
                                exact pre-grasp pose
                                       ↓
                              fixed Cartesian descent
                                       ↓
                                  close gripper
                                       ↓
                                      DONE
```

**Current task stops after gripper closing. Do not implement lifting in this milestone.**

The responsibility boundary is:

| Component | Responsibility |
|---|---|
| Existing goal-conditioned RL policy | Collision-aware global reaching toward the pre-grasp target position |
| Object pose registry | Store measured object position + orientation for Env A and Env B |
| Grasp transform registry | Store fixed object-relative grasp/pre-grasp geometry for kettle, mug, and infuser |
| Deterministic pose-alignment controller | Enforce the known pre-grasp end-effector orientation |
| Deterministic Cartesian descent | Move from pre-grasp to grasp along a fixed local approach direction/distance |
| Gripper controller | Close gripper using a defined command/limit |
| ROS 2 skill/state-machine layer | Orchestrate `REACH → ALIGN → DESCEND → CLOSE → DONE` after simulation validation |

> **[IMPORTANT CHANGE]** The existing RL checkpoint remains a position-goal-conditioned skill. Do not change the network observation to a 6-DoF pose goal merely to satisfy fixed grasp orientations in the current experiment. The orientation is known from the deterministic object setup and should be handled outside the learned policy unless a later research question explicitly studies learned grasp-pose selection.

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

> **[MODIFIED]** The architecture now separates **policy parameters** from **task instances**. Do not create six independent PPO checkpoints unless an experiment explicitly requires that ablation.

```text
                           EXISTING TRAINED SKILL

       validated Gen3 Lite + reaching MDP
                     ↓
         goal-conditioned PPO checkpoint
              reaching_policy.pt
                     ↓
       pi_theta(robot_state, goal_xyz)
                     ↓
             ONE shared policy


                         TEA-TABLE ADAPTATION

                TeaTable scene registry
                     ↓
          ┌──────────┴──────────┐
          │                     │
        Env A                 Env B
   kettle/mug/infuser    kettle/mug/infuser
          │                     │
          └──────────┬──────────┘
                     ↓
             Goal registry / YAML
                     ↓
           (env_id, object_id)
                     ↓
               target_xyz
                     ↓
       same reaching_policy.pt
                     ↓
            safe joint command


                           DEPLOYMENT

        goal request: (env_id, object_id)
                     ↓
              ROS 2 goal manager
                     ↓
                  target_xyz
                     ↓
        /joint_states ──┐
              ┌──────────┴──────────┐
              │                     │
       observation builder     FK / EE state
              │
        reaching_policy.pt
              │
          safety clamp
              │
     ros2_control / Kinova
              │
      Physical Gen3 Lite


                      FUTURE SHARED AUTONOMY

         Candidate goals: kettle / mug / infuser
                     ↓
          query same pi_theta for each goal
                     ↓
       a_kettle, a_mug, a_infuser
                     ↓
          goal belief / assistance layer
                     ↓
                 safety filter
                     ↓
                 Gen3 Lite
```

---

# 2. Priority map

> **[MODIFIED]** P0/P1 robot-model validation remains unchanged. The main change is that the project now treats the trained reaching checkpoint as an existing asset and inserts a tea-table adaptation stage before robustness and ROS deployment.

| Priority | Stage | Main objective | Exit criterion |
|---|---|---|---|
| P0 | Environment | Linux + GPU + ROS + Python tooling healthy | All prerequisite checks pass |
| P0 | Robot source | Obtain official Gen3 Lite Xacro/URDF/meshes | Source files located and reproducible |
| P0 | URDF → MJCF | Build standalone MuJoCo model | Viewer loads model correctly |
| P0 | Model validation | Validate joints, limits, FK, dynamics, actuators | Validated controllable 6-DOF model |
| P1 | MJLab asset | Port validated model into MJLab | 1 environment runs correctly |
| P1 | Reaching MDP | Preserve the validated goal-conditioned observation/action/reward definitions | Existing policy can be reproduced/evaluated |
| P1 | **[EXISTING] Reaching policy** | Load the already-trained arbitrary XYZ reaching checkpoint | Baseline checkpoint passes held-out reaching evaluation |
| P2 | **[NEW] Tea-table scene integration** | Add table + kettle + mug + infuser and Env A / Env B pose configurations | Both deterministic scenes load correctly |
| P2 | **[NEW] Six-goal adaptation** | Map `(env_id, object_id)` to target XYZ and query the same checkpoint | All six task instances can be executed |
| P2 | **[MODIFIED] Robustness** | Add scene randomization first, then dynamics/randomization as needed | Stable performance around measured configurations |
| P2 | ROS 2 deployment | Run one policy from ROS state + selected goal | Same checkpoint works through ROS 2 simulation/mock |
| P2 | Sim-to-real | Conservative real-arm deployment on measured tea-table layout | Safe reaching to all required container goals |
| P3 | Shared Autonomy | Add human input / goal inference / blending | Three-goal SA baseline operational |
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
    src/kinova_mjlab_reaching/tasks/tea_table \
    assets/tea_table \
    config \
    checkpoints \
    docs/validation
```

Target structure:

```text
kinova_mjlab_reaching/
│
├── assets/
│   ├── kinova_gen3_lite/
│   │   ├── source/
│   │   ├── meshes/
│   │   ├── gen3_lite.urdf
│   │   └── gen3_lite.xml
│   │
│   └── tea_table/
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
│   ├── tasks/tea_table/
│   │   ├── env_cfg.py
│   │   ├── scene_registry.py
│   │   ├── object_pose_registry.py
│   │   ├── grasp_registry.py
│   │   └── evaluation.py
│   │
│   └── ros/
│       └── policy_node.py
│
├── config/
│   ├── tea_table_objects.yaml
│   └── tea_table_grasps.yaml
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

# 21A. **[MODIFIED] Tea-table assets and collision-faithful object models**

Do this only after the Gen3 Lite MJLab asset is validated.

Create reusable scene assets rather than duplicating the robot model:

```text
assets/tea_table/
├── table.xml
├── kettle.xml
├── mug.xml
└── infuser.xml

config/
├── tea_table_objects.yaml
└── tea_table_grasps.yaml

src/kinova_mjlab_reaching/tasks/tea_table/
├── env_cfg.py
├── scene_registry.py
├── object_pose_registry.py
├── grasp_registry.py
└── evaluation.py
```

## 21A.1 Collision-model objective

The tea objects do **not** need photorealistic meshes. They do need simplified geometry that preserves the collision features relevant to the intended grasp.

### Kettle

Model at minimum:

- main kettle body,
- handle geometry/opening,
- approximate handle thickness and position relative to body,
- any large protrusion that can interfere with the gripper or wrist.

Primitive MuJoCo geometries such as boxes, capsules, cylinders, or a small number of convex meshes are preferred over unnecessarily detailed visual meshes.

### Infuser

Model at minimum:

- main infuser body,
- handle geometry/opening,
- approximate handle thickness and handle-to-body transform.

### Mug

The current intended grasp is around/at the cup rim rather than the mug handle. Model at minimum:

- outer body radius/width,
- height,
- rim location,
- wall/outer collision geometry sufficient to detect an invalid descent or gripper collision.

A mug handle is optional for this milestone unless it can interfere with the planned approach trajectory.

## 21A.2 Geometry validation rule

The simplified models exist to answer:

> Can the arm/gripper execute the known pre-grasp, approach, and closing sequence without intersecting the object body, handle, table, or non-target objects?

They are **not** introduced so that RL can discover the grasp orientation.

For each asset, create a validation view and record:

```text
object reference frame
body dimensions
handle/rim dimensions relevant to grasp
collision geometry
visual geometry if separate
object-frame origin convention
```

Create:

```text
docs/validation/tea_object_collision_geometry.md
```

---

# 21B. **[MODIFIED] Measure and define Env A / Env B object 6D poses**

The two environments contain the same three object classes but use different fixed object placements.

For each object in each environment, measure/store:

```text
x, y, z
roll, pitch, yaw    # or quaternion, but use one canonical representation internally
```

All physical measurements must be converted to a documented common frame, preferably the Kinova base frame used by deployment.

Conceptual registry:

```python
TEA_TABLE_SCENES = {
    "env_a": {
        "kettle":  {"xyz": [...], "quat": [...]},
        "mug":     {"xyz": [...], "quat": [...]},
        "infuser": {"xyz": [...], "quat": [...]},
    },
    "env_b": {
        "kettle":  {"xyz": [...], "quat": [...]},
        "mug":     {"xyz": [...], "quat": [...]},
        "infuser": {"xyz": [...], "quat": [...]},
    },
}
```

Recommended configuration file:

```yaml
# config/tea_table_objects.yaml

env_a:
  kettle:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]
  mug:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]
  infuser:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]

env_b:
  kettle:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]
  mug:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]
  infuser:
    position_xyz: [X, Y, Z]
    orientation_xyzw: [QX, QY, QZ, QW]
```

## 21B.1 Measurement procedure

For each of the six physical placements:

1. Place the robot/table in the intended experimental configuration.
2. Establish the robot-base frame and, if useful, a table frame.
3. Define an object-frame convention that is reproducible across Env A and Env B.
4. Measure object position.
5. Measure/record object yaw and any non-zero roll/pitch relevant to the grasp.
6. Convert the pose into the robot-base frame.
7. Enter the pose into `tea_table_objects.yaml`.
8. Reproduce the same pose in MuJoCo.
9. Visually validate the simulated placement against the physical layout.
10. Record measurement uncertainty and repeatability.

Create:

```text
docs/validation/tea_table_frame_and_measurement.md
```

The document must record:

```text
robot base-frame definition
table-frame definition if used
object-frame definition for each object class
measurement method
position units
orientation convention
measurement uncertainty
placement repeatability
```

> **[IMPORTANT]** Object orientation is now part of the deterministic task definition because the kettle/infuser handle direction and mug-rim grasp geometry depend on it.

---

# 21C. **[NEW] Define fixed object-relative grasp and pre-grasp transforms**

Do **not** hard-code six unrelated world-frame grasp poses. Define one reusable object-relative grasp strategy per object class.

For object `o`:

```text
T_base_object(env, o)
        ×
T_object_grasp(o)
        ↓
T_base_grasp(env, o)
```

Then define pre-grasp as a fixed transform from grasp:

```text
T_base_pregrasp
    = T_base_grasp × T_grasp_pregrasp_offset
```

Recommended registry:

```yaml
# config/tea_table_grasps.yaml

kettle:
  # grasp the fixed known handle
  grasp_pose_in_object_frame:
    position_xyz: [GX, GY, GZ]
    orientation_xyzw: [QX, QY, QZ, QW]
  pregrasp_offset_in_grasp_frame:
    position_xyz: [PX, PY, PZ]
  descent_distance_m: D
  gripper_close_command: VALUE

mug:
  # grasp at/around the cup rim according to the experimental design
  grasp_pose_in_object_frame:
    position_xyz: [GX, GY, GZ]
    orientation_xyzw: [QX, QY, QZ, QW]
  pregrasp_offset_in_grasp_frame:
    position_xyz: [PX, PY, PZ]
  descent_distance_m: D
  gripper_close_command: VALUE

infuser:
  # grasp the fixed known handle
  grasp_pose_in_object_frame:
    position_xyz: [GX, GY, GZ]
    orientation_xyzw: [QX, QY, QZ, QW]
  pregrasp_offset_in_grasp_frame:
    position_xyz: [PX, PY, PZ]
  descent_distance_m: D
  gripper_close_command: VALUE
```

The same `T_object_grasp` is reused in Env A and Env B. Only `T_base_object` changes.

## 21C.1 Current grasp assumptions

- `kettle`: grasp the known fixed handle.
- `infuser`: grasp the known fixed handle.
- `mug`: grasp at the known cup-rim location/geometry according to the selected gripper orientation.
- Object position and orientation are fixed and known within each deterministic environment.
- RL does not infer handle direction or discover a grasp pose.
- No lifting is required in the current milestone.

## 21C.2 Pre-grasp target passed to RL

The current checkpoint remains position-goal-conditioned. Therefore compute the full pre-grasp transform, but pass only its translational component to RL:

```text
T_base_pregrasp
      │
      ├── p_base_pregrasp → RL goal_xyz
      └── R_base_pregrasp → deterministic ALIGN stage
```

This preserves the trained checkpoint without changing observation dimensions.

---

# 22. P1/P2 — Goal-conditioned reaching task definition

## 22.1 **[MODIFIED] Task modes**

Preserve the original arbitrary-reaching task. Add a tea-table mode rather than replacing it.

```text
Mode 1: random_reaching
Mode 2: tea_table_reaching
```

Original random-reaching reset:

1. sample safe initial joint configuration,
2. place robot,
3. sample reachable target XYZ,
4. reset velocities,
5. reset episode counters.

Tea-table reset / evaluation:

1. select `env_id ∈ {env_a, env_b}`,
2. load the corresponding three object poses,
3. select `object_id ∈ {kettle, mug, infuser}`,
4. load the object's measured 6D pose,
5. apply the fixed object-relative grasp transform,
6. compute the full pre-grasp pose,
7. reset robot to a safe initial state,
8. pass `pregrasp_xyz` to the same trained policy,
9. retain `pregrasp_orientation` for the deterministic ALIGN stage.

Policy objective remains:

```text
move end-effector to target_xyz
```

### **[IMPORTANT CHANGE] Resolve the pre-grasp from object pose, not from object center**

Do not define tea-table goals as an arbitrary world-frame offset from the geometric center. Compute them through transforms:

```text
T_base_pregrasp
    = T_base_object
    × T_object_grasp
    × T_grasp_pregrasp_offset
```

Then use:

```text
target_xyz = translation(T_base_pregrasp)
```

This keeps handle/rim direction consistent with the measured object orientation in both Env A and Env B.

---

## 22.2 Observation

The trained policy observation definition must remain identical during inference.

Typical existing representation:

```text
q[6]
dq[6]
target_xyz - ee_xyz [3]
```

or whatever exact observation was used by the trained checkpoint.

> **[CRITICAL]** Do not change observation dimension/order/normalization when merely adapting the checkpoint to the six deterministic tea-table goals.

If obstacle positions are already encoded in the trained policy observation, preserve that encoding exactly.

If obstacle positions are **not** observed by the policy, do not assume the network can infer arbitrary new obstacle layouts. First test whether Env A and Env B lie inside the obstacle distribution already learned by the checkpoint.

---

## 22.3 Action

Preserve the trained action definition exactly.

Example:

```text
action[6] ∈ [-1, 1]
     ↓
scaled joint target increment
     ↓
q_target = q_current + Δq
```

Do not change action scaling, joint ordering, clipping, or control frequency during adaptation unless retraining/fine-tuning is intentionally started.

---

# 23. **[MODIFIED] Resolve six pre-grasp task instances from object poses**

The six skills are six logical task instances of the same checkpoint, but each task now resolves a **full deterministic pre-grasp pose** from the measured object 6D pose.

Resolution pipeline:

```text
(env_id, object_id)
       ↓
object 6D pose from tea_table_objects.yaml
       ↓
object-relative grasp transform from tea_table_grasps.yaml
       ↓
nominal grasp pose
       ↓
pre-grasp offset
       ↓
full pre-grasp pose
       │
       ├── pregrasp_xyz → existing RL checkpoint
       └── pregrasp_orientation → deterministic ALIGN controller
```

Required API concepts:

```python
def get_object_pose(env_id: str, object_id: str):
    ...

def get_grasp_spec(object_id: str):
    ...

def compute_grasp_pose(env_id: str, object_id: str):
    ...

def compute_pregrasp_pose(env_id: str, object_id: str):
    ...
```

Create:

```text
src/kinova_mjlab_reaching/tasks/tea_table/object_pose_registry.py
src/kinova_mjlab_reaching/tasks/tea_table/grasp_registry.py
```

The policy checkpoint must never be selected by object identity.

Wrong architecture:

```text
if kettle: load kettle_policy.pt
if mug: load mug_policy.pt
if infuser: load infuser_policy.pt
```

Correct architecture:

```text
load reaching_policy.pt once
resolve object pose
compute object-relative pre-grasp pose
pass pregrasp_xyz to same policy
```

---

# 23A. **[NEW] Deterministic grasp state machine in simulation**

After the RL reach succeeds, execute the remaining local grasp sequence deterministically.

```text
REACH
  │  existing goal-conditioned RL policy
  │  target = pregrasp_xyz
  ↓
REACH_SUCCESS?
  ↓
ALIGN
  │  deterministic controller enforces pregrasp orientation
  ↓
ALIGN_SUCCESS?
  ↓
DESCEND
  │  fixed Cartesian/local-frame displacement toward grasp pose
  ↓
DESCEND_SUCCESS?
  ↓
CLOSE
  │  deterministic gripper command
  ↓
DONE
```

Failure at any stage must stop the sequence and report a stage-specific failure code.

Recommended checks:

### REACH success

```text
||p_ee - p_pregrasp|| < reach_threshold
```

### ALIGN success

Check both:

```text
position remains inside pre-grasp tolerance
orientation error < orientation_threshold
```

### DESCEND success

Check:

```text
expected Cartesian displacement completed
no forbidden collision
gripper is positioned around the intended handle/rim region
```

### CLOSE success

For the current milestone, success may initially mean:

```text
gripper close command completed
no catastrophic penetration / invalid contact
```

A stronger grasp-quality metric may be added later. Do not require lifting yet.

## 23A.1 **[FINDING, 2026-09-01] REACH does not settle at the goal - transition to ALIGN immediately on success**

Confirmed empirically (logged actual simulated EE position/joint velocity directly, not through the viewer): once `REACH_SUCCESS?` first trips, the policy does **not** come to rest. Measured over the remainder of a full-length episode: EE position wanders ~2-6 cm (per-axis range), joint speed averages ~26 deg/s with peaks up to ~104 deg/s, even while nominally "at the goal."

Root cause: nothing in the M2.1/M2.2 reward incentivizes stopping. The dense reaching reward and the per-step target-reached bonus both reward *being close*, not *being still* - a policy that oscillates gently around the target scores nearly as well as one that holds. Training episodes also run the full fixed duration regardless of success, so during evaluation/viewing the policy keeps acting long after reaching the target with nothing productive to do.

**Required consequence for the state machine**: `REACH → REACH_SUCCESS?` must transition to `ALIGN` the instant `||p_ee - p_pregrasp|| < reach_threshold` is first true, and must stop querying the RL policy for further actions at that point - not continue running it for a fixed duration. Do not assume REACH leaves the arm stationary or that querying it longer improves anything; the opposite is closer to true. This is a real constraint on `grasp_skill_node.py`'s REACH→ALIGN transition (section 31.3), not just an evaluation-script detail.

If ALIGN/DESCEND turn out to need a genuinely stationary starting point that immediate hand-off doesn't provide, revisit the reward (e.g. a velocity-at-goal penalty) rather than assuming a fix is impossible - but try the immediate-handoff fix first, since it requires no retraining.

---

# 24. Reward baseline

For deterministic evaluation with an already-trained checkpoint, **do not change the reward** because reward is not needed for inference.

Retain the original reward implementation only for:

- regression testing,
- fine-tuning if adaptation fails,
- additional robustness training.

Original distance definition:

```text
d_t = ||p_ee - p_goal||
```

Success should continue to use the same tolerance used to validate the trained reaching policy unless the tea-table experiment requires a different task tolerance.

If the original criterion is `3 cm`, retain it for the first six-goal evaluation.

---

# 25. Episode termination

Retain:

### Success

```text
EE-target distance < threshold
```

### Timeout

Use the same simulated task horizon as the original trained policy.

### Failure / safety

- severe self collision,
- table collision if forbidden by task definition,
- collision with non-target containers if forbidden,
- forbidden workspace,
- joint-limit violation,
- unsafe state.

> **[NEW]** Explicitly decide whether contacting the selected target container is allowed. For the first reaching-only task, prefer a non-contact target offset so success can be validated without relying on contact dynamics.

---

# 26. **[MODIFIED] Mandatory deterministic tea-table and grasp validation**

Before any new PPO/fine-tuning and before ROS 2 integration, validate the full deterministic task in MuJoCo/MJLab.

## 26.1 Scene and collision-geometry validation

For Env A and Env B verify:

- table pose is correct,
- all six measured object poses are reproduced correctly,
- kettle handle collision geometry is present and correctly positioned,
- infuser handle collision geometry is present and correctly positioned,
- mug body/rim geometry is sufficient for the planned grasp,
- no object starts in penetration,
- non-target objects are represented as collision obstacles,
- object frames and grasp transforms are visualized and sensible.

## 26.2 Six RL pre-grasp reach tests

Run:

```text
A / kettle
A / mug
A / infuser
B / kettle
B / mug
B / infuser
```

At this stage stop after RL reaches `pregrasp_xyz`.

Log:

```text
success
final EE-pregrasp position error
time-to-target
collision flag
minimum clearance if available
trajectory
```

Do not proceed until the pre-grasp target definitions themselves are validated.

## 26.3 Six pre-grasp pose-alignment tests

Run all six again with:

```text
RL REACH → ALIGN → STOP
```

Check:

```text
pre-grasp XYZ error
pre-grasp orientation error
collision during alignment
joint limits
wrist/gripper clearance
```

## 26.4 Six approach/descent tests

Run:

```text
RL REACH → ALIGN → DESCEND → STOP
```

Check visually and numerically that:

- kettle gripper fingers approach the intended handle region,
- infuser gripper fingers approach the intended handle region,
- mug gripper approaches the intended rim region,
- the gripper/wrist does not intersect the main object body unexpectedly,
- the arm does not strike the table,
- the arm does not collide with non-target tea objects.

## 26.5 Six close-gripper tests

Finally run:

```text
RL REACH → ALIGN → DESCEND → CLOSE → DONE
```

Do **not** lift.

Record:

```text
stage success/failure
final pose error
collision/contact summary
gripper command result
whether geometry/contact is physically plausible
```

Create:

```text
docs/validation/tea_table_six_goal_pregrasp.md
docs/validation/tea_table_six_grasp_sequence.md
```

> **[HARD GATE]** The complete `REACH → ALIGN → DESCEND → CLOSE` sequence must work in simulation for all six deterministic task instances before connecting the task logic to the physical Kinova arm.

---

# 27. **[IMPORTANT CHANGE] Retraining decision gate**

The default assumption is:

> Changing only `target_xyz` does **not** require retraining a correctly trained goal-conditioned policy.

Use this decision logic.

```text
Existing checkpoint
      ↓
Run six deterministic tea-table goals
      ↓
Are all scenes/goals inside learned distribution?
      │
 ┌────┴────┐
 │         │
YES       NO / UNCERTAIN
 │         │
No new   Diagnose failure
training     │
            ├── frame / goal offset error → fix config, no training
            ├── unreachable goal → fix scene/goal, no training
            ├── collision geometry issue → fix model, no training
            ├── obstacle layout outside training distribution → fine-tune
            └── dynamics mismatch → robustness/domain randomization
```

### No retraining required when

- all six goals are inside the original reachable-goal distribution,
- the existing observation/action definitions are unchanged,
- obstacle layouts are sufficiently represented by the training distribution,
- six-goal evaluation meets acceptance criteria.

### Fine-tuning may be required when

- policy reaches the correct XYZ but repeatedly collides with the tea-table objects,
- Env A succeeds but Env B fails due to obstacle rearrangement,
- the trained obstacle distribution is materially narrower than the new scenes,
- sim-to-real perturbations cause systematic failure.

### Do not create six independent policies as the first fix

If adaptation is required, first fine-tune the **same checkpoint** over both environments and all three goals.

### **[NOTE, 2026-09-01]** Retraining is not off the table when it's the right fix

The decision gate above biases toward config/geometry fixes first because they're cheaper and lower-risk to try, not because retraining is forbidden. Where this project's own history shows evidence should override that default bias (e.g. the M2.2 initial-state-randomization and joint_vel_l2 experiments both retrained/fine-tuned when the evidence called for it, and both were abandoned when the evidence didn't), keep doing that: retrain when it's genuinely justified and worth the cost, don't default to it, and don't refuse it out of habit either.

---

# 28. **[NEW] Six-goal success criterion**

Evaluate the checkpoint on the complete deterministic matrix:

| Environment | Kettle | Mug | Infuser |
|---|---|---|---|
| Env A | pass/fail | pass/fail | pass/fail |
| Env B | pass/fail | pass/fail | pass/fail |

Recommended metrics:

```text
success threshold: preserve trained-policy threshold initially
success rate across repeated resets
median final position error
90th percentile final position error
time-to-target
collision rate
minimum clearance
```

Minimum milestone requirement before ROS deployment:

> All six deterministic task instances must work reliably from the intended initial-state distribution.

The agent must record both aggregate metrics and per-goal failures.

---

# 29. **[MODIFIED] P2 — Tea-table robustness and domain randomization**

Only add randomization **after** the deterministic six-goal baseline is understood.

The order matters.

## 29.1 **[MODIFIED — FIRST] Scene / object-pose randomization**

Because the task depends on manually measured physical object **6D poses** and known handle/rim geometry, scene randomization should be the first robustness layer.

Randomize around each nominal Env A / Env B configuration rather than immediately sampling the entire workspace.

Example conceptual perturbation:

```text
object x/y: nominal +- small measurement/layout tolerance
object z: nominal +- small height tolerance
object yaw: nominal +- small angular tolerance
object roll/pitch: only if physical placement uncertainty makes them relevant
handle/rim geometry dimensions: small perturbations only if real geometry uncertainty matters
table-to-base transform: small translation/yaw perturbation
goal measurement: small XYZ noise
```

Do not hard-code perturbation magnitudes without recording why they are reasonable. Prefer values derived from:

- manual measurement repeatability,
- table placement repeatability,
- object placement variability,
- robot/base calibration uncertainty.

## 29.2 Initial-state randomization

Retain:

```text
q0 ~ safe distribution
```

Use the same or a subset of the initial-state distribution used by the original policy.

## 29.3 Goal randomization

The nominal six goals remain the task anchors.

For robustness training/evaluation:

```text
g = g_nominal + epsilon_goal
```

This tests tolerance to measurement and placement error while preserving semantic goal identity.

## 29.4 **[LATER] Dynamics randomization**

Add only after scene robustness is established or if real-arm testing reveals a concrete dynamics mismatch:

- damping,
- friction,
- actuator response,
- control delay,
- encoder/state noise,
- action latency,
- payload effects if relevant.

## 29.5 Safety terms

Retain/add as needed:

- soft joint-limit penalties,
- collision penalties,
- action-rate penalty,
- velocity limits,
- table clearance,
- non-target-object clearance.

## 29.6 Fine-tuning protocol if needed

If the deterministic checkpoint is inadequate, continue from the existing checkpoint rather than restarting PPO from random weights.

Train across:

```text
Env A nominal + perturbations
Env B nominal + perturbations
all three target identities
```

Goal-conditioned structure must be preserved:

```text
one checkpoint
multiple goals
multiple scenes
```

Save the result as a new version of the same skill family, for example:

```text
reaching_policy_base.pt
reaching_policy_tea_table_ft.pt
```

Do not name checkpoints by individual container unless intentionally running an ablation.

---

# 30. Keep ROS 2 out of the RL training loop

Recommended architecture remains:

```text
MJLab training / fine-tuning:
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

# 31. **[MODIFIED] P2 — ROS 2 integration after simulation grasp success**

ROS 2 integration begins only after Section 26 passes in simulation.

Create or adapt:

```text
src/kinova_mjlab_reaching/ros/policy_node.py
src/kinova_mjlab_reaching/ros/goal_manager_node.py
src/kinova_mjlab_reaching/ros/grasp_skill_node.py
config/tea_table_objects.yaml
config/tea_table_grasps.yaml
```

Recommended architecture:

```text
(env_id, object_id) request
          ↓
   goal_manager_node
          │
          ├── object 6D pose
          ├── grasp pose
          └── pre-grasp pose
                    ↓
              grasp_skill_node
                    │
     ┌──────────────┼──────────────┬──────────────┐
     │              │              │              │
   REACH          ALIGN          DESCEND         CLOSE
     │              │              │              │
 RL policy      deterministic   Cartesian       gripper
 node/checkpt    pose ctrl       motion          command
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                    ↓
               safety layer
                    ↓
             ros2_control/Kinova
```

## 31.1 RL policy node responsibility

The policy node should:

```text
read joint state
build the exact training observation
receive pregrasp_xyz
run reaching_policy.pt
post-process action exactly as in training
output safe reaching command
```

It should **not** decide kettle/mug/infuser grasp orientation.

### 31.1.1 **[FINDING, 2026-09-01] Raw policy output is fast - the safety clamp downstream of this node is not optional**

Measured in simulation: mean joint speed ~172 deg/s during the REACH approach, peaks up to ~259 deg/s. That is the *raw trained policy's* behavior, unmodified.

Do **not** try to slow this down by editing `action_scale`, the joint actuator PD gains (`kp`/`kv`), or the control/decimation frequency post-hoc for deployment. Those are exactly the dynamics the policy learned against; changing any of them at inference time without retraining creates a train/inference mismatch that can degrade accuracy or destabilize the policy, not just slow it down safely.

The correct place to cap real-world speed is the **safety clamp stage** already in this section's architecture diagram, between this node's output and `ros2_control` - a downstream rate/velocity limiter on the actual command sent to the robot, independent of the trained policy and the simulated dynamics it was validated against. This is consistent with, and reinforces, section 33's staged conservative rollout ("extremely small action scale / conservative limits" as the first real-hardware step) - that step means limiting the *command sent to the robot*, not retuning the policy's own action scale.

## 31.2 Goal manager responsibility

The goal manager should:

```text
receive env_id + object_id
load measured object pose
apply fixed object-relative grasp transform
compute grasp pose
compute pre-grasp pose
provide pregrasp_xyz + pregrasp_orientation + descent specification
```

## 31.3 Grasp skill node responsibility

Implement the state machine:

```text
IDLE
 ↓
REACH
 ↓
ALIGN
 ↓
DESCEND
 ↓
CLOSE
 ↓
DONE
```

Provide explicit failure transitions for:

```text
reach timeout
alignment failure
collision/safety stop
descent failure
gripper failure
```

Do not add `LIFT` yet.

Critical deployment equivalences remain:

```text
joint order
joint sign
units
home pose
control frequency
action scale
EE frame
joint limits
observation normalization
object-pose coordinate frame
gripper tool frame
```

---

# 32. **[MODIFIED] Test ROS 2 orchestration before real hardware**

Use one or more:

1. Kinova fake hardware,
2. ROS-side simulation,
3. MuJoCo + ROS integration,
4. offline recorded joint-state replay for policy I/O tests.

For the full skill, preferred validation is a ROS-controlled simulation/fake-hardware path that exercises the same state transitions intended for the real robot.

Verify for all six task instances:

```text
goal manager resolves the correct object 6D pose
pre-grasp transform is correct
policy observation is identical to training
policy checkpoint remains the same
REACH success transition works
ALIGN command uses the correct fixed orientation
DESCEND direction/distance is correct
CLOSE command is issued only after descent succeeds
safety stop aborts later stages
switching env/object does not reload a different RL checkpoint
```

Add a ROS-side six-task regression test:

```text
(env_a, kettle)
(env_a, mug)
(env_a, infuser)
(env_b, kettle)
(env_b, mug)
(env_b, infuser)
```

---

# 33. **[MODIFIED] P2 — Conservative tea-table Sim-to-Real**

First physical tests should use the manually measured deterministic layout.

Recommended progression:

```text
1. establish robot base <-> table/object measurement convention
2. place Env A only
3. measure all three object positions and orientations
4. update tea_table_objects.yaml
5. verify computed grasp/pre-grasp transforms without commanding the robot
6. policy inference with command output disabled
7. inspect pregrasp_xyz and predicted RL actions
8. extremely small action scale / conservative limits
9. Env A: REACH only for one target
10. Env A: REACH + ALIGN for one target
11. Env A: REACH + ALIGN + DESCEND for one target
12. Env A: full REACH + ALIGN + DESCEND + CLOSE
13. Env A: repeat for all three objects
14. Env B: measure/update all object 6D poses
15. Env B: repeat staged tests for all three objects
16. repeat with small placement perturbations after deterministic success
```

Use robot-side limits and an external emergency stop.

Do not bypass Kinova safety mechanisms.

### **[IMPORTANT] Updating measured object positions does not itself imply retraining**

If a container is moved and its new measured target remains inside the learned goal/scene distribution, update the scene/goal configuration and re-evaluate. Retrain/fine-tune only when performance demonstrates a distribution-shift problem rather than a coordinate/configuration problem.

---

# 34. **[MODIFIED] Real-robot parameters and measurements to record**

Before deployment, record the original robot parameters:

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

Also record tea-table task geometry:

```text
robot base frame
table pose relative to robot base
Env A kettle/mug/infuser 6D poses
Env B kettle/mug/infuser 6D poses
object-frame conventions
fixed object-relative grasp transforms
pre-grasp offsets / approach directions
descent distance per object
gripper close command per object
measurement uncertainty
object placement tolerance
simplified collision-geometry dimensions relevant to grasp
```

Only perform deeper system identification if sim-to-real testing shows a concrete mismatch.

Possible later identification:

- control latency,
- velocity response,
- actuator lag,
- damping,
- friction,
- payload effects.

---

# 35. **[MODIFIED] P3 — Shared Autonomy extension for three candidate tea goals**

The same reaching checkpoint remains the autonomous skill:

```text
pi_R(s, g) → a_R
```

For a single environment, define three candidate goals:

```text
g_kettle
g_mug
g_infuser
```

The same policy can be queried three times:

```text
a_kettle  = pi_R(s, g_kettle)
a_mug     = pi_R(s, g_mug)
a_infuser = pi_R(s, g_infuser)
```

This preserves a clean separation between:

```text
goal inference / human intent
        ↓
goal selection or probability
        ↓
one goal-conditioned autonomous skill
```

Add human input:

```text
u_H
```

## Stage SA-1 — Fixed blending

Once a goal is selected/assumed:

```text
a = (1 - alpha) u_H + alpha a_R
```

Compare:

```text
human only
vs
autonomy only
vs
fixed shared control
```

## Stage SA-2 — Goal uncertainty

Maintain:

```text
P(g | human input, state)
```

over:

```text
{kettle, mug, infuser}
```

Use:

```text
human joystick history
robot state
candidate target positions
```

## Stage SA-3 — Adaptive assistance

Learn or design:

```text
alpha = f(
    goal confidence,
    task state,
    human behavior,
    safety margin
)
```

## Stage SA-4 — Simulated human for scaling

Create simulated human policies with:

- skill variation,
- Gaussian joystick noise,
- delay,
- hesitation,
- incorrect goal movement,
- correction behavior.

Run these against the same three candidate goals in both Env A and Env B.

## Stage SA-5 — Safety layer

Potential architecture:

```text
human action / intent
       ↓
SA goal inference / assistance
       ↓
same goal-conditioned reaching policy
       ↓
safety filter
       ↓
Gen3 Lite
```

Possible research directions:

- control barrier functions,
- constrained optimization,
- collision avoidance,
- human-aware constraints.

---

# 36. **[MODIFIED] Suggested Git milestones**

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
Goal-conditioned arbitrary reaching environment
```

## M2.1

```text
Existing trained reaching checkpoint validated
```

## M2.2 — **[MODIFIED]**

```text
Tea-table assets + simplified collision-faithful kettle/mug/infuser models
+ measured Env A / Env B object 6D poses
```

## M2.3 — **[NEW]**

```text
Object-relative grasp transforms + six pre-grasp pose definitions
```

## M2.4 — **[NEW]**

```text
Six-task deterministic simulation validation:
REACH → ALIGN → DESCEND → CLOSE (no lift)
```

## M2.5 — **[MODIFIED]**

```text
Tea-table scene/object-pose randomization + robustness evaluation
```

## M2.6 — **[OPTIONAL]**

```text
Single-checkpoint fine-tuning only if deterministic/robustness tests show real distribution shift
```

## M3

```text
ROS2 goal manager + single-policy deployment
```

## M3.1

```text
Conservative physical Env A / Env B reaching
```

## M4

```text
Three-goal shared-autonomy fixed-blending baseline
```

## M4.1

```text
Goal inference + adaptive / learning-based shared autonomy
```

---

# 37. **[MODIFIED] What the agent should do next**

This section depends on project status.

If M1/M2 and the arbitrary-reaching checkpoint are already complete, **do not repeat model conversion or restart PPO**.

The immediate adaptation sequence is:

```text
[1] Identify and freeze the exact trained checkpoint + policy config
        ↓
[2] Record observation order / normalization / action scaling / control dt
        ↓
[3] Build table + simplified collision-faithful kettle/mug/infuser assets
        ↓
[4] Define object frames for kettle, mug, infuser
        ↓
[5] Measure Env A object XYZ + orientation in robot-base frame
        ↓
[6] Measure Env B object XYZ + orientation in robot-base frame
        ↓
[7] Populate config/tea_table_objects.yaml
        ↓
[8] Define one fixed object-relative grasp transform per object class
        ↓
[9] Define pre-grasp offsets, descent distances, and gripper commands
        ↓
[10] Populate config/tea_table_grasps.yaml
        ↓
[11] Validate both deterministic scenes and all collision geometry
        ↓
[12] Run six RL REACH-only pre-grasp tests
        ↓
[13] Run six REACH → ALIGN tests
        ↓
[14] Run six REACH → ALIGN → DESCEND tests
        ↓
[15] Run six full REACH → ALIGN → DESCEND → CLOSE tests (no lift)
        ↓
[16] Write six-task simulation validation report
        ↓
[17] Decide: no retraining vs one-checkpoint fine-tuning
        ↓
[18] Add small measured scene/object-pose randomization
        ↓
[19] Run robustness evaluation
        ↓
[20] Implement ROS2 goal manager + grasp skill state machine + existing policy node
        ↓
[21] ROS-side six-task regression test in simulation/fake hardware
        ↓
[22] Conservative Env A physical test, staged REACH→ALIGN→DESCEND→CLOSE
        ↓
[23] Conservative Env B physical test, staged REACH→ALIGN→DESCEND→CLOSE
```

Agent stop condition:

> Do not begin fine-tuning merely because the scene changed. First produce the deterministic six-goal evaluation and diagnose whether any failure comes from frames, target definition, reachability, collision geometry, or actual policy distribution shift.

---

# 38. **[MODIFIED] Immediate success definition**

The next concrete objective is:

> **Use one existing trained goal-conditioned Kinova Gen3 Lite reaching policy as the shared autonomous REACH component for six tea-table grasp task instances — `{Env A, Env B} x {kettle, mug, infuser}` — where measured object 6D poses and simplified collision-faithful geometry define fixed object-relative pre-grasp/grasp transforms; validate `REACH → ALIGN → DESCEND → CLOSE` in simulation first, then reproduce the same state machine through ROS 2 without changing the RL checkpoint.**

Success does **not** mean six separately trained neural networks.

The expected artifact set is:

```text
reaching_policy.pt                    # one shared RL checkpoint
config/tea_table_objects.yaml        # measured Env A/B object 6D poses
config/tea_table_grasps.yaml         # object-relative grasp/pre-grasp specs
assets/tea_table/kettle.xml          # simplified collision-faithful object model
assets/tea_table/mug.xml
assets/tea_table/infuser.xml
TeaTable Env A configuration
TeaTable Env B configuration
goal_manager_node.py
policy_node.py
grasp_skill_node.py
docs/validation/tea_table_six_goal_pregrasp.md
docs/validation/tea_table_six_grasp_sequence.md
```

If fine-tuning becomes necessary, retain one goal-conditioned policy family and document the new checkpoint separately.

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

# 42. **[MODIFIED] Research roadmap in one line**

```text
Validated MuJoCo Robot
    → Goal-Conditioned Arbitrary Reaching
    → EXISTING Trained Reaching Checkpoint
    → Tea-Table Env A / Env B
    → Six Goal Instances (kettle / mug / infuser)
    → Deterministic Validation
    → Scene Randomization
    → Optional Single-Checkpoint Fine-Tuning
    → ROS2 Goal Manager + Policy Deployment
    → Sim-to-Real Tea-Table Reaching
    → Three-Goal Shared Autonomy
    → Adaptive Assistance / User Study
```

---

# 43. **[NEW] Agent invariants for this adaptation**

The coding agent must preserve the following invariants unless the user explicitly changes the experiment design:

1. **One trained policy checkpoint is the default autonomous skill.**
2. `kettle`, `mug`, and `infuser` are goal identities, not separate neural-network identities.
3. Env A and Env B are scene configurations, not separate robot models.
4. Goal changes are made through `target_xyz`, not by loading another network.
5. Manual physical measurements are acceptable for the first baseline if converted into a documented robot-relative frame.
6. Updating measured object positions does not automatically trigger retraining.
7. Deterministic six-goal evaluation must precede domain randomization.
8. Scene/pose randomization should precede broad dynamics randomization for this task.
9. If adaptation training is required, fine-tune one goal-conditioned checkpoint across both scenes and all goals.
10. ROS 2 must expose goal selection separately from policy inference.
11. Shared Autonomy should query the same policy under multiple candidate goals rather than maintain one network per candidate goal.
12. Every change to observation ordering, normalization, action scaling, control frequency, frame convention, or goal definition must be explicitly documented because any of these can invalidate a trained checkpoint.
