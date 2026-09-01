"""
Watch a trained reaching checkpoint attempt a fixed tea-table pre-grasp
target live in the browser, via mjlab's Viser play viewer (same one
watch_training.py uses for training playback).

Unlike watch_training.py, the command here must be pinned to a FIXED
pregrasp_xyz instead of ReachingCommand's normal random-target sampling.
This monkey-patches the command term's _resample_command at the source
(called on every reset, whenever/however that reset is triggered inside
the viewer's own render loop) rather than trying to intercept resets from
the outside - simpler and more robust than the "overwrite target_pos after
every step's dones" approach evaluate_tea_table_reach.py uses, which only
works because that script fully owns its own step loop.

Usage:
    uv run python scripts/watch_tea_table_reach.py --env-id env_a --object-id kettle \\
        --checkpoint checkpoints/2026-08-27_12-52-48/model_9999.pt
"""

import argparse
import types
from dataclasses import asdict
from pathlib import Path

import torch

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.mdp.commands import ReachingCommand
from kinova_mjlab_reaching.tasks.tea_table.env_cfg import get_tea_table_env_cfg
from kinova_mjlab_reaching.tasks.tea_table.grasp_registry import compute_pregrasp_pose
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import OBJECT_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.viewer.viser import ViserPlayViewer


def _pin_command_to_fixed_target(
    command: ReachingCommand, raw_env: ManagerBasedRlEnv, target_local: torch.Tensor
) -> None:
    """Replaces _resample_command so every reset - including ones the
    viewer's own loop triggers internally - lands on the same fixed
    pregrasp target instead of a random one."""

    def _fixed_resample(self: ReachingCommand, env_ids: torch.Tensor) -> None:
        self.episode_success[env_ids] = 0.0
        self.target_pos[env_ids] = target_local + raw_env.scene.env_origins[env_ids]

    command._resample_command = types.MethodType(_fixed_resample, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=("env_a", "env_b"), default="env_a")
    parser.add_argument("--object-id", choices=OBJECT_IDS, default="kettle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env_cfg = get_tea_table_env_cfg(args.env_id, num_envs=1)
    raw_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=None)

    agent_cfg = asdict(get_reaching_ppo_cfg())
    runner = MjlabOnPolicyRunner(
        env, agent_cfg, str(args.checkpoint.parent), args.device
    )
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    command = raw_env.command_manager.get_term("reach_target")
    assert isinstance(command, ReachingCommand)
    pregrasp_pos, _ = compute_pregrasp_pose(args.env_id, args.object_id)
    target_local = torch.tensor(pregrasp_pos, dtype=torch.float32, device=args.device)
    _pin_command_to_fixed_target(command, raw_env, target_local)

    print(f"Environment: {args.env_id}, target object: {args.object_id}")
    print(f"Pinned pregrasp target (robot base frame): {pregrasp_pos}")
    print("Launching Viser viewer — open the printed URL in your browser.")
    print("Use the Reset button in the GUI to watch fresh attempts; the")
    print("target stays pinned to the same point every time.")
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
    main()
