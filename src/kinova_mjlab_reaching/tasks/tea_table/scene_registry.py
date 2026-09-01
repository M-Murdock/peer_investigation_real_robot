"""Builds the table + kettle + mug + infuser scene for one tea-table
environment (runbook v2 section 21A/21B), positioned per the measured (for
now: placeholder) poses in config/tea_table_objects.yaml.

Mirrors reaching/scene.py's get_scene_object_cfgs() shape (a dict of
EntityCfg keyed by entity name) so it drops into the same
`SceneCfg(entities={"robot": ..., **get_tea_table_scene_object_cfgs(env_id)})`
pattern used there - see reach_env_cfg.py.
"""

from mjlab.entity import EntityCfg

from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import (
    ENV_IDS,
    OBJECT_IDS,
    get_object_pose,
    get_table_pose,
)
from kinova_mjlab_reaching.tasks.tea_table.objects import (
    OBJECT_SPEC_FNS,
    TABLE_HALF_EXTENTS,
    get_arm_base_fixture_spec,
    get_table_spec,
)


def get_tea_table_scene_object_cfgs(env_id: str) -> dict[str, EntityCfg]:
    if env_id not in ENV_IDS:
        raise KeyError(f"Unknown env_id {env_id!r}; expected one of {ENV_IDS}")

    table_pos, table_quat = get_table_pose()
    table_top_z = table_pos[2] + TABLE_HALF_EXTENTS[2]
    cfgs: dict[str, EntityCfg] = {
        "table": EntityCfg(
            spec_fn=get_table_spec,
            init_state=EntityCfg.InitialStateCfg(pos=table_pos, rot=table_quat),
        ),
        "arm_base_fixture": EntityCfg(
            spec_fn=get_arm_base_fixture_spec,
            # Fixed under the arm (world x=0, y=0), base resting exactly on
            # the tabletop surface - not read from tea_table_objects.yaml,
            # see objects.py's ARM_BASE_FIXTURE section for why.
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, table_top_z)),
        ),
    }
    for object_id in OBJECT_IDS:
        pos, quat = get_object_pose(env_id, object_id)
        cfgs[object_id] = EntityCfg(
            spec_fn=OBJECT_SPEC_FNS[object_id],
            init_state=EntityCfg.InitialStateCfg(pos=pos, rot=quat),
        )
    return cfgs
