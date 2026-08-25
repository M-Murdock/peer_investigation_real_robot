"""
Load the resolved URDF in MuJoCo and save as MJCF.

MuJoCo resolves relative mesh paths from the URDF's directory, so we load
from the asset dir and write the MJCF alongside it.

Usage:
    uv run python scripts/model_conversion/urdf_to_mjcf.py
"""

import sys
from pathlib import Path

import mujoco

ASSET_DIR = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite"
INPUT_URDF = ASSET_DIR / "gen3_lite_resolved.urdf"
OUTPUT_MJCF = ASSET_DIR / "gen3_lite.xml"


def main() -> None:
    if not INPUT_URDF.exists():
        sys.exit(f"ERROR: {INPUT_URDF} not found — run resolve_mesh_paths.py first")

    print(f"Loading: {INPUT_URDF}")
    model = mujoco.MjModel.from_xml_path(str(INPUT_URDF))

    print(f"nq   : {model.nq}")
    print(f"nv   : {model.nv}")
    print(f"njnt : {model.njnt}")
    print(f"nbody: {model.nbody}")
    print(f"nu   : {model.nu}")

    mujoco.mj_saveLastXML(str(OUTPUT_MJCF), model)
    print(f"\nMJCF written to: {OUTPUT_MJCF}")


if __name__ == "__main__":
    main()
