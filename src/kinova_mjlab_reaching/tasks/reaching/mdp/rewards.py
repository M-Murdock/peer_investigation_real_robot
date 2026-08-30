"""Reward for the reaching task. Baseline (runbook section 24):

    reward = reaching (dense) + target_reached (success bonus) - action_rate - joint_pos_limits

kept deliberately simple at first — no collision-distance shaping, no
orientation term, no curriculum. Obstacle avoidance was handled purely by
hard termination on contact (see terminations.py), per the runbook's "do
not add many reward terms before the baseline is understood."

`obstacle_proximity_penalty` adds dense collision-distance shaping on top
of that baseline (runbook section 29, M2.2 robustness stage) — added only
after the M2.1 baseline converged and held-out evaluation showed a
residual 6.3% teapot-collision rate the hard termination alone wasn't
resolving (see docs/ppo_baseline.md).
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


def obstacle_proximity_penalty(
    env: ManagerBasedRlEnv,
    obstacle_names: tuple[str, ...],
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dense Gaussian penalty for the end effector nearing any named obstacle.

    Mirrors reaching_distance_reward's kernel shape but as a danger signal
    instead of a goal signal — one term per obstacle, summed, so proximity
    to multiple obstacles at once compounds. Distance is EE-site-to-body-
    origin, not true surface distance; std should be picked to roughly
    cover the obstacle's own extent (see scene.py geometry) plus a small
    margin, since this is meant to discourage the arm from lingering near
    an obstacle, not to model exact collision geometry (that's what the
    hard illegal_contact termination in terminations.py is for).
    """
    robot: Entity = env.scene[asset_cfg.name]
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
    penalty = torch.zeros(env.num_envs, device=ee_pos_w.device)
    for name in obstacle_names:
        obstacle: Entity = env.scene[name]
        obstacle_pos_w = obstacle.data.root_link_pos_w
        distance_sq = torch.sum(torch.square(obstacle_pos_w - ee_pos_w), dim=-1)
        penalty = penalty + torch.exp(-distance_sq / std**2)
    return penalty
