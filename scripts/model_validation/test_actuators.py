"""
Actuator validation — Section 16 of the runbook.

Commands each of the 6 arm joints to a small target position one at a time
while physics runs. The viewer stays open so you can watch the motion.

Terminal interface:
    1+  move act_joint_1 target to +0.3 rad
    1-  move act_joint_1 target to -0.3 rad
    1r  reset act_joint_1 target to 0
    r   reset all joints to 0
    q   quit

The script also prints per-step diagnostics so you can confirm:
  - joint moves toward target (not away)
  - other joints stay near zero
  - no NaN in qpos / qvel
  - force stays within the actuatorfrcrange

Usage:
    uv run python scripts/model_validation/test_actuators.py
"""

from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

MJCF = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite" / "gen3_lite.xml"

ARM_JOINTS   = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ACT_NAMES    = [f"act_joint_{i+1}" for i in range(6)]
FORCE_LIMITS = [10, 14, 10, 7, 7, 7]   # N·m, from actuatorfrcrange in MJCF
TARGET_STEP  = 0.3   # rad ≈ 17°
SIM_DT       = 0.002 # matches MuJoCo default timestep


def _run_sim(model, data, viewer, stop_event):
    """Physics loop — runs in a background thread."""
    while not stop_event.is_set() and viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(SIM_DT)


def print_state(data, model, act_ids, jnt_addrs):
    q   = [data.qpos[a] for a in jnt_addrs]
    qd  = [data.qvel[a] for a in jnt_addrs]
    f   = [data.actuator_force[i] for i in range(6)]
    ctrl = data.ctrl[:6].tolist()
    nan_q  = any(np.isnan(v) for v in q)
    nan_qd = any(np.isnan(v) for v in qd)

    flag = "  NaN!" if (nan_q or nan_qd) else ""
    print(f"  q   : {[f'{np.degrees(v):+6.1f}°' for v in q]}{flag}")
    print(f"  ctrl: {[f'{np.degrees(v):+6.1f}°' for v in ctrl]}")
    print(f"  f   : {[f'{v:+6.2f}N' for v in f]}")


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data  = mujoco.MjData(model)

    jnt_addrs = [int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    ]) for n in ARM_JOINTS]

    act_ids = list(range(6))

    print("\nActuator test — physics is running.")
    print("Commands:  1+  1-  1r  ...  6+  6-  6r  |  r = reset all  |  q = quit\n")

    stop = threading.Event()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]     = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]  = True

        sim_thread = threading.Thread(target=_run_sim,
                                      args=(model, data, viewer, stop),
                                      daemon=True)
        sim_thread.start()

        while viewer.is_running():
            cmd = input("cmd> ").strip().lower()

            if cmd == "q":
                break

            if cmd == "r":
                data.ctrl[:] = 0
                data.qpos[:] = 0
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                print("  reset all")
                print_state(data, model, act_ids, jnt_addrs)
                continue

            if len(cmd) == 2 and cmd[0] in "123456" and cmd[1] in "+-":
                i    = int(cmd[0]) - 1
                sign = +1 if cmd[1] == "+" else -1
                lo, hi = model.actuator_ctrlrange[i]
                data.ctrl[i] = float(np.clip(data.ctrl[i] + sign * TARGET_STEP, lo, hi))
                print(f"  act_joint_{i+1} target → {np.degrees(data.ctrl[i]):+.1f}°")
                time.sleep(1.5)   # let the joint settle
                print_state(data, model, act_ids, jnt_addrs)
                continue

            if len(cmd) == 2 and cmd[0] in "123456" and cmd[1] == "r":
                i = int(cmd[0]) - 1
                data.ctrl[i] = 0.0
                print(f"  act_joint_{i+1} target → 0°")
                time.sleep(1.5)
                print_state(data, model, act_ids, jnt_addrs)
                continue

            print("  unrecognised — try '2+', '3-', '4r', 'r', 'q'")

        stop.set()

    print("Done.")


if __name__ == "__main__":
    main()
