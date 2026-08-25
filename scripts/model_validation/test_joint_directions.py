"""
Interactive joint direction tester.

The viewer stays open the whole time. Type commands in the terminal to
move arm joints and watch which way they rotate. Joint axes are drawn
as colored arrows in the viewer so you can see the rotation axis.

Commands:
    1+ / 1-   move joint 1 by +/- 17° (0.3 rad)
    2+ / 2-   move joint 2 ...
    ...up to 6
    r         reset all joints to zero
    q         quit

Usage:
    uv run python scripts/model_validation/test_joint_directions.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MJCF = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite" / "gen3_lite.xml"
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
STEP = 0.3  # rad ≈ 17°

HINTS = [
    "swivel whole arm — look from above",
    "shoulder — look from the side (X-Z plane)",
    "elbow — look from the side",
    "forearm roll — look along the forearm from elbow toward wrist",
    "wrist pitch — look from the side of the wrist",
    "tool roll — look along end-effector axis from tip inward",
]


def joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name!r}")
    return int(model.jnt_qposadr[jid])


def print_state(data: mujoco.MjData, addrs: list[int]) -> None:
    vals = [f"j{i+1}={np.degrees(data.qpos[a]):+.1f}°" for i, a in enumerate(addrs)]
    print("  " + "  ".join(vals))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    addrs = [joint_qpos_addr(model, n) for n in ARM_JOINTS]

    print("\nOpening viewer — joint axes drawn as colored arrows.")
    print("Commands:  1+  1-  2+  2-  ... 6+  6-  |  r = reset  |  q = quit\n")
    for i, hint in enumerate(HINTS):
        print(f"  joint_{i+1}: {hint}")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show joint rotation axes as arrows and body coordinate frames
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        mujoco.mj_forward(model, data)
        viewer.sync()

        while viewer.is_running():
            cmd = input("cmd> ").strip().lower()

            if cmd == "q":
                break

            if cmd == "r":
                data.qpos[:] = 0
                mujoco.mj_forward(model, data)
                viewer.sync()
                print("  reset to zero")
                print_state(data, addrs)
                continue

            if len(cmd) == 2 and cmd[0] in "123456" and cmd[1] in "+-":
                jidx = int(cmd[0]) - 1
                sign = +1 if cmd[1] == "+" else -1
                adr = addrs[jidx]
                lo = model.jnt_range[jidx][0]
                hi = model.jnt_range[jidx][1]
                data.qpos[adr] = float(np.clip(data.qpos[adr] + sign * STEP, lo, hi))
                mujoco.mj_forward(model, data)
                viewer.sync()
                print_state(data, addrs)
                continue

            print("  unrecognised — use e.g. '1+', '3-', 'r', 'q'")

    print("Done.")


if __name__ == "__main__":
    main()
