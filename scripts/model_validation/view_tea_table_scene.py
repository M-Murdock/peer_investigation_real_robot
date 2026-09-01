"""
Launch the MuJoCo viewer for visual inspection of one tea-table environment
(runbook v2 section 21A/21B/26.1): the Gen3 Lite arm + table + kettle + mug
+ infuser, positioned per config/tea_table_objects.yaml.

All object geometry and poses are currently PLACEHOLDERS - see
config/tea_table_objects.yaml, config/tea_table_grasps.yaml, and
tasks/tea_table/objects.py. This script exists to validate the scaffolding
(does it load, do objects sit on the table, is nothing already colliding)
before real measurements replace the placeholders.

Check in the viewer:
  - table sits at a sensible height/position relative to the robot base
  - all three objects rest ON the tabletop, not floating or clipping through
  - kettle/infuser handles and the mug rim site are visible and plausible
  - arm's home pose does not intersect the table or any object
  - no unexpected contacts at t=0 (printed below)
  - computed grasp/pre-grasp points (printed below) land near the visible
    handle/rim markers, not off in space - a quick sanity check on the
    transform chain in grasp_registry.py independent of the real geometry

Usage:
    uv run python scripts/model_validation/view_tea_table_scene.py --env-id env_a
    uv run python scripts/model_validation/view_tea_table_scene.py --env-id env_b

If there's no display (headless/SSH, "GLFWError: DISPLAY environment
variable is missing"), add --browser to use mjlab's Viser web viewer
instead (same one scripts/watch_training.py uses for policy playback) -
no X11 needed, just open the printed http://localhost:8080 URL:

    uv run python scripts/model_validation/view_tea_table_scene.py --env-id env_a --browser
"""

import argparse
import time

import mujoco
import mujoco.viewer

from kinova_mjlab_reaching.robots.gen3_lite import get_gen3_lite_robot_cfg
from kinova_mjlab_reaching.tasks.tea_table.grasp_registry import (
    compute_grasp_pose,
    compute_pregrasp_pose,
)
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import OBJECT_IDS
from kinova_mjlab_reaching.tasks.tea_table.scene_registry import (
    get_tea_table_scene_object_cfgs,
)
from mjlab.scene import Scene, SceneCfg


def _launch_browser_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import viser

    from mjlab.viewer.viser import ViserMujocoScene

    server = viser.ViserServer(label="Tea-Table Scene Viewer")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.update_from_mjdata(data)
    print("Launching Viser viewer — open the printed URL in your browser.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=("env_a", "env_b"), default="env_a")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Use the browser-based Viser viewer instead of the native GLFW "
        "window (needed when there's no X11 display).",
    )
    args = parser.parse_args()

    entities = {
        "robot": get_gen3_lite_robot_cfg(),
        **get_tea_table_scene_object_cfgs(args.env_id),
    }
    scene = Scene(SceneCfg(entities=entities), device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Environment: {args.env_id}")
    print(f"Model loaded: {model.nbody} bodies, {model.nu} actuators")
    print(f"Contacts at home pose: {data.ncon} (expect 0)")
    print()
    print("Computed grasp / pre-grasp poses (placeholder transform chain):")
    for object_id in OBJECT_IDS:
        grasp_pos, _ = compute_grasp_pose(args.env_id, object_id)
        pregrasp_pos, _ = compute_pregrasp_pose(args.env_id, object_id)
        print(
            f"  {object_id:8s} grasp_xyz={tuple(round(v, 3) for v in grasp_pos)}"
            f"  pregrasp_xyz={tuple(round(v, 3) for v in pregrasp_pos)}"
        )
    print()
    if args.browser:
        _launch_browser_viewer(model, data)
    else:
        print("Launching viewer — close window to exit.")
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
