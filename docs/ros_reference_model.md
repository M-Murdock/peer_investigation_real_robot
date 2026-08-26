# ROS Reference Model — Kinova Gen3 Lite

## Source

| Field | Value |
|---|---|
| Repository | https://github.com/Kinovarobotics/ros2_kortex |
| Branch | jazzy |
| Commit | `63944dee465d836ea714a114ea3657dfa1617d95` |
| Xacro entry | `kortex_description/robots/gen3_lite_gen3_lite_2f.xacro` |
| Xacro args | `use_fake_hardware:=true sim_gazebo:=false` (all other args at default) |

## Generated files

| File | Description |
|---|---|
| `assets/kinova_gen3_lite/gen3_lite.urdf` | Raw output from `xacro` (package:// and file:// URIs intact) |
| `assets/kinova_gen3_lite/gen3_lite_resolved.urdf` | All mesh URIs rewritten to relative `meshes/<name>.STL` |
| `assets/kinova_gen3_lite/meshes/` | 11 STL meshes (6 arm + 5 gripper) copied from ros2_kortex source |
| `assets/kinova_gen3_lite/gen3_lite.xml` | MuJoCo MJCF exported from the resolved URDF |

## Model summary

| Property | Value |
|---|---|
| nq | 10 |
| nv | 10 |
| njnt | 10 |
| nbody | 11 |
| nu | 0 (no actuators yet — to be added in MJCF) |
| Arm joints | joint_1 … joint_6 (revolute, hinge) |
| Gripper joints | right_finger_bottom, right_finger_tip, left_finger_bottom, left_finger_tip |
| Base | Fixed (world → base_link via fixed joint) |

## Kinematic tree

```
world (fixed)
└── base_link
    └── shoulder_link  (joint_1)
        └── arm_link   (joint_2)
            └── forearm_link  (joint_3)
                └── lower_wrist_link  (joint_4)
                    └── upper_wrist_link  (joint_5)
                        └── end_effector_link  (joint_6)
                            ├── tool_frame  (fixed)
                            └── gripper_base_link  (fixed)
                                ├── right_finger_prox_link  (right_finger_bottom_joint)
                                │   └── right_finger_dist_link  (right_finger_tip_joint)
                                └── left_finger_prox_link  (left_finger_bottom_joint)
                                    └── left_finger_dist_link  (left_finger_tip_joint)
```

## Status

- [x] URDF generated and validated with check_urdf
- [x] Mesh paths resolved (no remaining package:// or file:// URIs)
- [x] MJCF loads in MuJoCo 3.12.0 without errors
- [x] Joint table written to `mujoco_joint_table.md`
- [x] Visual inspection — confirmed 2026-08-25 via test_joint_directions.py
- [x] Joint axes verified — all 6 correct, no sign flips needed
- [x] FK cross-validation — MuJoCo ee_site vs kinpy URDF FK, max error 0.001 mm (PASS)
- [x] Actuators added — 6 position actuators (kp=100), joint damping=1.0, forcerange from URDF limits
- [x] ee_site added — pos="0 0 0.130" in end_effector_link (matches Kinova tool_frame_joint offset)
- [x] Actuator smoke test passed — joint_1 reaches target, no NaN, no saturation at zero pose
