# Joint Mapping — Kinova Gen3 Lite (ROS ↔ MuJoCo)

Verified 2026-08-25 using `test_joint_directions.py` (viewer + joint axis arrows).

## Arm joints

| Idx | ROS/Kinova joint | MuJoCo joint | Axis (local z) | Lower (deg) | Upper (deg) | Expected positive motion | Observed positive motion | Sign OK? | Notes |
|----:|------------------|--------------|---------------|------------:|------------:|--------------------------|--------------------------|:--------:|-------|
| 0 | joint_1 | joint_1 | z (world-z) | -153.6 | 153.6 | CCW from above — base swivel | Correct | ✓ | |
| 1 | joint_2 | joint_2 | z (local) | -149.5 | 149.5 | Shoulder pitch | Correct | ✓ | |
| 2 | joint_3 | joint_3 | z (local) | -149.5 | 149.5 | Elbow bends, forearm up | Correct | ✓ | |
| 3 | joint_4 | joint_4 | z (local) | -149.0 | 149.0 | Forearm rolls CW from elbow tip | Correct | ✓ | |
| 4 | joint_5 | joint_5 | z (local) | -145.0 | 145.0 | Wrist bends up | Correct | ✓ | |
| 5 | joint_6 | joint_6 | z (local) | -149.0 | 149.0 | End-effector rolls CW from tip | Correct | ✓ | |

## Gripper joints (passive — not used in reaching policy)

| Idx | Joint | Lower (deg) | Upper (deg) | Notes |
|----:|-------|------------:|------------:|-------|
| 6 | right_finger_bottom_joint | -5.7 | 55.0 | Passive in RL reaching task |
| 7 | right_finger_tip_joint | -28.6 | 12.0 | Passive |
| 8 | left_finger_bottom_joint | -5.7 | 55.0 | Passive |
| 9 | left_finger_tip_joint | -28.6 | 12.0 | Passive |

## Sign flip notes

If any arm joint moves in the opposite direction from expected, do NOT change
the MJCF — keep it faithful to the URDF. Instead, negate that joint's action
in the policy action-scaling layer.

| Joint | Sign flip needed? | Flip applied where? |
|-------|:-----------------:|---------------------|
| joint_1 | No | — |
| joint_2 | No | — |
| joint_3 | No | — |
| joint_4 | No | — |
| joint_5 | No | — |
| joint_6 | No | — |

## Verification status

- [x] All 6 arm joints visually inspected (2026-08-25)
- [x] Observed motion vs expected recorded above
- [x] Sign flip table completed — no flips needed
- [x] Ready to proceed to actuator addition (Section 15)
