"""
Launch the MuJoCo viewer for visual inspection of the reaching scene: the
Gen3 Lite arm mounted on a tea-serving table with static obstacles
(teapot + 2 cups).

Check:
  - table's back edge aligns with the robot's base (no gap, no overlap)
  - obstacles rest on the tabletop without clipping through it
  - arm's home pose does not intersect the table or obstacles
  - no unexpected contacts at t=0 (printed below)

Usage:
    uv run python scripts/model_validation/view_reaching_scene.py
"""

import mujoco
import mujoco.viewer

from kinova_mjlab_reaching.robots.gen3_lite import get_gen3_lite_robot_cfg
from kinova_mjlab_reaching.tasks.reaching.scene import get_scene_object_cfgs
from mjlab.scene import Scene, SceneCfg


def main() -> None:
    entities = {"robot": get_gen3_lite_robot_cfg(), **get_scene_object_cfgs()}
    scene = Scene(SceneCfg(entities=entities), device="cpu")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Model loaded: {model.nbody} bodies, {model.nu} actuators")
    print(f"Contacts at home pose: {data.ncon} (expect 0)")
    print("Launching viewer — close window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
