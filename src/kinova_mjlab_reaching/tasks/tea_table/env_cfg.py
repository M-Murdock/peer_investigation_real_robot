"""Tea-table environment config (runbook v2 section 22.1, "Mode 2:
tea_table_reaching"): the exact same observation/action/reward/termination
definitions as the trained reaching checkpoint (tasks/reaching/reach_env_cfg
.get_reaching_env_cfg), with only the scene swapped for one of the two
tea-table layouts - per section 22.2's hard requirement not to change the
policy's observation when adapting it to a new scene.

Does NOT define a new reward/observation/action from scratch - reuses
get_reaching_env_cfg's parameterization added for exactly this purpose.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import (
    ENV_IDS,
    OBJECT_IDS,
)
from kinova_mjlab_reaching.tasks.tea_table.scene_registry import (
    get_tea_table_scene_object_cfgs,
)


def get_tea_table_env_cfg(env_id: str, num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    if env_id not in ENV_IDS:
        raise KeyError(f"Unknown env_id {env_id!r}; expected one of {ENV_IDS}")
    return get_reaching_env_cfg(
        num_envs=num_envs,
        scene_object_cfgs_fn=lambda: get_tea_table_scene_object_cfgs(env_id),
        obstacle_names=OBJECT_IDS,  # ("kettle", "mug", "infuser")
        # arm_base_fixture: hard-terminate on contact like the table, but
        # skip dense obstacle_proximity_penalty shaping - it's a fixed
        # installation directly under the arm's own base, not a container
        # to route around during a normal reach.
        extra_collision_names=("arm_base_fixture",),
    )
