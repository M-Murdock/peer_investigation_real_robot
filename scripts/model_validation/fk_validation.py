"""
FK cross-validation — Section 18 of the runbook.

For N joint configurations, computes forward kinematics from two independent
sources and compares the end-effector (tool_frame) position:

  Source A — MuJoCo:  ee_site world position  (from gen3_lite.xml)
  Source B — kinpy:   tool_frame transform     (from gen3_lite_resolved.urdf)

Both sources derive from the same URDF, so agreement confirms that the
URDF→MJCF conversion preserved the kinematic chain correctly.

Acceptance criterion (from runbook Section 18):
  position error < 1 mm  (numerical precision of URDF→MJCF round-trip)

Writes results to docs/validation/fk_validation.md.

Usage:
    uv run python scripts/model_validation/fk_validation.py
"""

from pathlib import Path

import kinpy as kp
import mujoco
import numpy as np

ASSET_DIR  = Path(__file__).parent.parent.parent / "assets" / "kinova_gen3_lite"
MJCF       = ASSET_DIR / "gen3_lite.xml"
URDF       = ASSET_DIR / "gen3_lite_resolved.urdf"
DOCS_DIR   = Path(__file__).parent.parent.parent / "docs" / "validation"

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# tool_frame is the Kinova reference EE frame (matches ee_site offset in MJCF)
EE_FRAME   = "tool_frame"
BASE_FRAME = "base_link"

np.random.seed(42)


# ── MuJoCo FK ────────────────────────────────────────────────────────────────

def mujoco_fk(model: mujoco.MjModel, data: mujoco.MjData,
              q: np.ndarray) -> np.ndarray:
    """Set arm joints to q (rad) and return ee_site world position."""
    addrs = [int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    ]) for n in ARM_JOINTS]
    data.qpos[:] = 0.0
    for i, a in enumerate(addrs):
        data.qpos[a] = q[i]
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    return data.site_xpos[sid].copy()


# ── kinpy FK ─────────────────────────────────────────────────────────────────

def kinpy_fk(chain: kp.Chain, q: np.ndarray) -> np.ndarray:
    """Return tool_frame position via kinpy FK."""
    th = {n: float(q[i]) for i, n in enumerate(ARM_JOINTS)}
    tf = chain.forward_kinematics(th)
    return np.array(tf.pos)


# ── Test configurations ───────────────────────────────────────────────────────

def make_configs(model: mujoco.MjModel) -> list[np.ndarray]:
    """Zero pose + 5 reproducible random safe configs."""
    configs = [np.zeros(6)]
    lo = np.array([model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
        for n in ARM_JOINTS])
    hi = np.array([model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
        for n in ARM_JOINTS])
    safe_lo = lo * 0.5
    safe_hi = hi * 0.5
    for _ in range(5):
        configs.append(np.random.uniform(safe_lo, safe_hi))
    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data  = mujoco.MjData(model)

    chain = kp.build_serial_chain_from_urdf(
        URDF.read_bytes(), EE_FRAME, BASE_FRAME
    )

    configs = make_configs(model)

    rows: list[dict] = []
    max_err = 0.0

    print(f"\n{'Config':<8} {'MuJoCo ee (m)':>32} {'kinpy ee (m)':>32} {'err (mm)':>10}")
    print("-" * 88)

    for idx, q in enumerate(configs):
        p_mj = mujoco_fk(model, data, q)
        p_kp = kinpy_fk(chain, q)
        err  = np.linalg.norm(p_mj - p_kp) * 1000  # mm
        max_err = max(max_err, err)

        label = "zero" if idx == 0 else f"rand{idx}"
        mj_str = f"[{p_mj[0]:.4f}, {p_mj[1]:.4f}, {p_mj[2]:.4f}]"
        kp_str = f"[{p_kp[0]:.4f}, {p_kp[1]:.4f}, {p_kp[2]:.4f}]"
        print(f"{label:<8} {mj_str:>32} {kp_str:>32} {err:>9.3f}")
        rows.append({"label": label, "q": q.tolist(),
                     "p_mj": p_mj.tolist(), "p_kp": p_kp.tolist(), "err_mm": err})

    passed = max_err < 1.0
    print(f"\nMax position error: {max_err:.3f} mm  →  {'PASS ✓' if passed else 'FAIL ✗  (> 1 mm)'}")

    # Write markdown report
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "fk_validation.md"
    lines = [
        "# FK Cross-Validation — Kinova Gen3 Lite\n",
        "**Sources compared:**",
        "- Source A: MuJoCo `ee_site` world position (`gen3_lite.xml`)",
        "- Source B: kinpy `tool_frame` transform (`gen3_lite_resolved.urdf`)\n",
        "**Acceptance criterion:** position error < 1 mm\n",
        "| Config | MuJoCo ee [x,y,z] m | kinpy ee [x,y,z] m | Error (mm) |",
        "|--------|---------------------|--------------------|:----------:|",
    ]
    for r in rows:
        mj = [f"{v:.4f}" for v in r["p_mj"]]
        kp_ = [f"{v:.4f}" for v in r["p_kp"]]
        lines.append(
            f"| {r['label']} | [{', '.join(mj)}] | [{', '.join(kp_)}] | {r['err_mm']:.3f} |"
        )
    lines += [
        "",
        f"**Max error: {max_err:.3f} mm** — {'PASS ✓' if passed else 'FAIL ✗'}",
        "",
        "## Status",
        f"- [{'x' if passed else ' '}] FK validated — MuJoCo ee_site matches URDF kinpy FK",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"Report written to: {out}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
