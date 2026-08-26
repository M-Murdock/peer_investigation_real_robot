"""Reward baseline for the reaching task, per runbook section 24:

    reward = reaching (dense) + target_reached (success bonus) - action_rate - joint_pos_limits

Kept deliberately simple — no collision-distance shaping, no orientation
term, no curriculum yet. Obstacle avoidance is handled by hard termination
on contact (see terminations.py), not by reward shaping, per the runbook's
"do not add many reward terms before the baseline is understood."
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", site_names=("ee_site",))


def reaching_distance_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense Gaussian kernel over end-effector-to-target distance."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    target_pos_w = env.command_manager.get_command(command_name)
    position_error = torch.sum(torch.square(target_pos_w - ee_pos_w), dim=-1)
    return torch.exp(-position_error / std**2)


def target_reached_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
    success_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary bonus while the end effector is within success_threshold of target."""
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    target_pos_w = env.command_manager.get_command(command_name)
    distance = torch.norm(target_pos_w - ee_pos_w, dim=-1)
    return (distance < success_threshold).float()
