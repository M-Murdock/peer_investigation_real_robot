"""
Launch the MuJoCo viewer for visual inspection of the Gen3 Lite model.

Check:
  - correct arm orientation (vertical base, z-up)
  - fixed base (no drift)
  - all 11 meshes visible and aligned
  - no exploding links at t=0
  - sensible zero pose

Usage:
    uv run python scripts/model_validation/view_model.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer

MJCF = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite" / "gen3_lite.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Model loaded: {model.nbody} bodies, {model.njnt} joints")
    print("Launching viewer — close window to exit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
