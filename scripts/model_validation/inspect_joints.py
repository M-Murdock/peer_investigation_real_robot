"""
Print a table of all MuJoCo joints and their properties, then write
docs/mujoco_joint_table.md.

Usage:
    uv run python scripts/model_validation/inspect_joints.py
"""

from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite"
MJCF = ASSET_DIR / "gen3_lite.xml"
DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "validation"

JOINT_TYPE_MAP = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))

    rows: list[dict] = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<unnamed_{i}>"
        jtype = JOINT_TYPE_MAP.get(model.jnt_type[i], str(model.jnt_type[i]))
        axis = model.jnt_axis[i].tolist()
        limited = bool(model.jnt_limited[i])
        lower = model.jnt_range[i][0] if limited else float("nan")
        upper = model.jnt_range[i][1] if limited else float("nan")
        body_id = model.jnt_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<body_{body_id}>"
        qpos_adr = model.jnt_qposadr[i]
        dof_adr = model.jnt_dofadr[i]

        rows.append({
            "idx": i,
            "name": name,
            "type": jtype,
            "axis": f"[{axis[0]:.2f},{axis[1]:.2f},{axis[2]:.2f}]",
            "lower": f"{np.degrees(lower):.1f}°" if limited else "—",
            "upper": f"{np.degrees(upper):.1f}°" if limited else "—",
            "body": body_name,
            "qpos_adr": qpos_adr,
            "dof_adr": dof_adr,
        })

    # Console output
    header = f"{'Idx':>4}  {'Name':<35}  {'Type':<6}  {'Axis':<18}  {'Lower':>10}  {'Upper':>10}  {'Body':<25}  {'qpos':>4}  {'dof':>4}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['idx']:>4}  {r['name']:<35}  {r['type']:<6}  {r['axis']:<18}  "
              f"{r['lower']:>10}  {r['upper']:>10}  {r['body']:<25}  {r['qpos_adr']:>4}  {r['dof_adr']:>4}")

    print(f"\nnq={model.nq}  nv={model.nv}  njnt={model.njnt}  nbody={model.nbody}  nu={model.nu}")

    # Markdown table
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "mujoco_joint_table.md"
    lines = [
        "# MuJoCo Joint Table — Kinova Gen3 Lite\n",
        "| Idx | Name | Type | Axis | Lower (deg) | Upper (deg) | Body | qpos_adr | dof_adr |",
        "|----:|------|------|------|------------:|------------:|------|:--------:|:-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['idx']} | {r['name']} | {r['type']} | {r['axis']} "
            f"| {r['lower']} | {r['upper']} | {r['body']} | {r['qpos_adr']} | {r['dof_adr']} |"
        )
    lines += [
        "",
        f"**nq={model.nq}  nv={model.nv}  njnt={model.njnt}  nbody={model.nbody}  nu={model.nu}**",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"\nTable written to: {out}")


if __name__ == "__main__":
    main()
