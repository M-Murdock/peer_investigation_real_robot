"""Kinova Gen3 Lite MJLab robot asset.

Wraps the M1-validated MJCF (assets/kinova_gen3_lite/gen3_lite.xml) as an
mjlab EntityCfg. Do not re-derive the model here — see docs/ for
the FK, joint-mapping, and actuator validation this file depends on.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEN3_LITE_XML: Path = PROJECT_ROOT / "assets" / "kinova_gen3_lite" / "gen3_lite.xml"
assert GEN3_LITE_XML.exists(), f"MJCF not found: {GEN3_LITE_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GEN3_LITE_XML))


##
# Actuator config.
##

# Wrap the 6 <position> actuators already defined and validated in the MJCF
# (kp=100, kv=10, ctrlrange = joint range — see
# docs/ros_reference_model.md) instead of re-specifying gains.
# Gripper joints (right/left_finger_*) are intentionally left unactuated —
# passive per docs/joint_mapping.md.
ARM_ACTUATORS = (XmlActuatorCfg(target_names_expr=("joint_[1-6]",)),)

##
# Keyframe config.
##

# All-zero pose matches the "zero" configuration validated in
# docs/fk_validation.md.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(joint_pos={".*": 0.0})

##
# Final config.
##

# Self-collision left at MJCF defaults (mesh geoms, no dedicated collision
# hulls) — no self-collision issues were observed during M1 validation.
# Revisit under runbook §19/§29 (robustness stage) if training speed demands
# simplified collision geometry.
ARTICULATION = EntityArticulationInfoCfg(
    actuators=ARM_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)


def get_gen3_lite_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_gen3_lite_robot_cfg())

    viewer.launch(robot.spec.compile())
