"""Registers the reaching task with mjlab's task registry.

Importing this module (or kinova_mjlab_reaching.tasks.reaching) has the
side effect of calling register_mjlab_task, making "Kinova-Reach-v0"
available to scripts that look it up by task_id.

play_env_cfg reuses the same config as training for now — no curriculum or
train-only noise exists yet that a play variant would need to strip out.
"""

from mjlab.tasks.registry import register_mjlab_task

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg

TASK_ID = "Kinova-Reach-v0"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=get_reaching_env_cfg(num_envs=256),
    play_env_cfg=get_reaching_env_cfg(num_envs=1),
    rl_cfg=get_reaching_ppo_cfg(),
)
