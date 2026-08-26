"""Goal-conditioned reaching command: samples a target end-effector position
each episode, within the tea table's footprint (mount frame), per
runbook section 23 ("Reachable goal sampling") — extended here to bias
sampling toward the region where the static obstacles actually are, rather
than the arm's full unconstrained workspace, since the point of this task is
reaching *around* those obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

from kinova_mjlab_reaching.tasks.reaching.scene import MOUNT_YAW_DEG

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_yaw = math.radians(MOUNT_YAW_DEG)
_COS, _SIN = math.cos(_yaw), math.sin(_yaw)


def _mount_to_world(depth: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
    """Batched version of scene._mount_to_world, using the same rotation."""
    x = _COS * depth - _SIN * lateral
    y = _SIN * depth + _COS * lateral
    return torch.stack([x, y], dim=-1)


class ReachingCommand(CommandTerm):
    cfg: ReachingCommandCfg

    def __init__(self, cfg: ReachingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        site_ids, _ = self.robot.find_sites((cfg.ee_site_name,))
        self._site_ids = site_ids

        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    @property
    def ee_pos_w(self) -> torch.Tensor:
        return self.robot.data.site_pos_w[:, self._site_ids].squeeze(1)

    def _update_metrics(self) -> None:
        position_error = torch.norm(self.target_pos - self.ee_pos_w, dim=-1)
        at_goal = (position_error < self.cfg.success_threshold).float()
        self.episode_success = torch.maximum(self.episode_success, at_goal)

        self.metrics["position_error"] = position_error
        self.metrics["at_goal"] = at_goal
        self.metrics["episode_success"] = self.episode_success

    def compute_success(self) -> torch.Tensor:
        return self.metrics["position_error"] < self.cfg.success_threshold

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.episode_success[env_ids] = 0.0

        r = self.cfg.target_range
        depth = sample_uniform(r.depth[0], r.depth[1], (n,), device=self.device)
        lateral = sample_uniform(r.lateral[0], r.lateral[1], (n,), device=self.device)
        height = sample_uniform(r.height[0], r.height[1], (n,), device=self.device)

        xy = _mount_to_world(depth, lateral)
        target = torch.cat([xy, height.unsqueeze(-1)], dim=-1)
        self.target_pos[env_ids] = target + self._env.scene.env_origins[env_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids  # Target is fixed for the episode; nothing to advance per-step.

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            target_pos = self.target_pos[batch].cpu().numpy()
            visualizer.add_sphere(
                center=target_pos,
                radius=0.02,
                color=self.cfg.viz.target_color,
                label=f"reach_target_{batch}",
            )


@dataclass(kw_only=True)
class ReachingCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    ee_site_name: str = "ee_site"
    success_threshold: float = 0.03
    """Meters. Runbook section 24: success = ee-to-target distance < 3 cm."""

    @dataclass
    class TargetRangeCfg:
        """Target position sampling range, in the scene's mount frame
        (see tasks/reaching/scene.py): depth out from the table's back edge,
        lateral offset, height above the tabletop surface. Sized to overlap
        the obstacle layout rather than the arm's full workspace, since the
        task is reaching around the obstacles, not reaching anywhere."""

        depth: tuple[float, float] = (0.15, 0.55)
        lateral: tuple[float, float] = (-0.28, 0.28)
        height: tuple[float, float] = (0.03, 0.30)

    target_range: TargetRangeCfg = field(default_factory=TargetRangeCfg)

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.1, 0.9, 0.1, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> ReachingCommand:
        return ReachingCommand(self, env)
