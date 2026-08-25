"""
Resolve package:// URIs in the Kinova Gen3 Lite URDF, copy referenced meshes
into the self-contained asset directory, and rewrite paths to relative URIs.

Usage:
    uv run python scripts/model_conversion/resolve_mesh_paths.py
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

KORTEX_DESCRIPTION_ROOT = Path.home() / "workspace/ros2_kortex_ws/src/ros2_kortex/kortex_description"
ASSET_DIR = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite"
INPUT_URDF = ASSET_DIR / "gen3_lite.urdf"
OUTPUT_URDF = ASSET_DIR / "gen3_lite_resolved.urdf"
MESH_DEST = ASSET_DIR / "meshes"

PACKAGE_MAP = {
    "kortex_description": KORTEX_DESCRIPTION_ROOT,
}


def resolve_package_uri(uri: str) -> Path:
    if not uri.startswith("package://"):
        raise ValueError(f"Not a package URI: {uri}")
    rest = uri[len("package://"):]
    pkg, *parts = rest.split("/")
    if pkg not in PACKAGE_MAP:
        raise KeyError(f"Unknown package '{pkg}' in URI: {uri}")
    return PACKAGE_MAP[pkg] / "/".join(parts)


def main() -> None:
    if not INPUT_URDF.exists():
        sys.exit(f"ERROR: input URDF not found: {INPUT_URDF}")

    MESH_DEST.mkdir(parents=True, exist_ok=True)

    ET.register_namespace("", "")
    tree = ET.parse(INPUT_URDF)
    root = tree.getroot()

    copied: list[str] = []
    missing: list[str] = []

    for mesh_el in root.iter("mesh"):
        uri = mesh_el.get("filename", "")

        if uri.startswith("package://"):
            try:
                src = resolve_package_uri(uri)
            except (ValueError, KeyError) as exc:
                missing.append(f"  RESOLVE FAILED  {uri}  ({exc})")
                continue
        elif uri.startswith("file://"):
            src = Path(uri[len("file://"):])
        else:
            continue  # already relative or unknown scheme

        if not src.exists():
            missing.append(f"  FILE NOT FOUND  {src}")
            continue

        dest = MESH_DEST / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        relative = f"meshes/{src.name}"
        mesh_el.set("filename", relative)
        copied.append(f"  OK  {src.name}")

    if missing:
        print("ERRORS — missing meshes:")
        for m in missing:
            print(m)
        sys.exit(1)

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_URDF, xml_declaration=True, encoding="unicode")

    print(f"Resolved URDF written to: {OUTPUT_URDF}")
    print(f"Meshes ({len(set(copied))}) copied to: {MESH_DEST}")
    for line in sorted(set(copied)):
        print(line)


if __name__ == "__main__":
    main()
