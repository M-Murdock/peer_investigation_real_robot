"""
Held-out evaluation of a trained reaching policy (runbook section 28).

Training-time metrics conflate two things that matter separately for this
task: "did the arm ever reach the target" and "did it collide with
something" are not mutually exclusive within one episode (a command is
sampled once per episode, so nothing stops the arm from reaching the goal
and then drifting into an obstacle afterward, still counting as a training
"success"). This script runs a frozen, deterministic policy (no PPO
exploration noise — get_inference_policy() returns the distribution mean)
against a fresh batch of episodes and classifies each one explicitly:

    clean_success   reached the target, episode ended by time_out
    success_then_collision   reached the target, but a later collision
                              ended the episode anyway
    collision_before_success reached a collision termination without ever
                              getting within the success threshold
    failure_no_collision     never reached the target, ran out of time

Usage:
    uv run python scripts/evaluate_reaching.py --checkpoint checkpoints/kinova_reach/<run>/model_4999.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.mdp.commands import ReachingCommand
from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

_COLLISION_TERMS = (
    "teapot_collision",
    "cup_1_collision",
    "cup_2_collision",
    "table_collision",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument(
        "--episodes-per-env",
        type=int,
        default=8,
        help="Rollout length is this many full episode durations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Different from the training seed (42) so this is a held-out sample.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env_cfg = get_reaching_env_cfg(num_envs=args.num_envs)
    env_cfg.seed = args.seed
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
    success_threshold = command.cfg.success_threshold
    step_dt = raw_env.step_dt
    max_episode_length = raw_env.max_episode_length
    num_steps = max_episode_length * args.episodes_per_env

    ever_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
    steps_to_success = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
    step_in_episode = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)

    episodes: list[dict] = []

    # command.metrics["position_error"] gets overwritten for any env that
    # auto-resets *within* the same env.step() call that reports it done —
    # the command manager resamples a new target and recomputes the metric
    # against it before step() returns, so reading it post-hoc for a
    # just-terminated env gives "distance from the old arm pose to the
    # brand-new episode's target," not the real final error. Snapshotting
    # the previous step's value sidesteps this: it was captured while that
    # env was still mid-episode, one step (8ms) before termination.
    prev_pos_err = torch.full((env.num_envs,), float("nan"), device=args.device)

    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(num_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step_in_episode += 1

            pos_err = command.metrics["position_error"]
            newly_succeeded = (~ever_succeeded) & (pos_err < success_threshold)
            steps_to_success[newly_succeeded] = step_in_episode[newly_succeeded]
            ever_succeeded |= newly_succeeded

            done_ids = dones.nonzero().flatten().tolist()
            for i in done_ids:
                cause = "time_out"
                for name in _COLLISION_TERMS:
                    if bool(raw_env.termination_manager.get_term(name)[i]):
                        cause = name
                        break
                success = bool(ever_succeeded[i])
                episodes.append(
                    {
                        "success": success,
                        "termination": cause,
                        "position_error": float(prev_pos_err[i]),
                        "time_to_success_s": (
                            float(steps_to_success[i]) * step_dt if success else None
                        ),
                    }
                )

            prev_pos_err = pos_err.clone()

            if done_ids:
                done_idx = torch.as_tensor(done_ids, device=args.device)
                ever_succeeded[done_idx] = False
                steps_to_success[done_idx] = 0
                step_in_episode[done_idx] = 0

    n = len(episodes)
    if n == 0:
        raise SystemExit("No episodes completed — increase --episodes-per-env.")

    successes = [e for e in episodes if e["success"]]
    collisions = [e for e in episodes if e["termination"] != "time_out"]
    clean_success = [e for e in successes if e["termination"] == "time_out"]
    success_then_collision = [e for e in successes if e["termination"] != "time_out"]
    collision_before_success = [e for e in collisions if not e["success"]]
    failure_no_collision = [
        e for e in episodes if not e["success"] and e["termination"] == "time_out"
    ]

    position_errors = np.array(
        [e["position_error"] for e in episodes if not np.isnan(e["position_error"])]
    )
    times_to_success = np.array(
        [
            e["time_to_success_s"]
            for e in successes
            if e["time_to_success_s"] is not None
        ]
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes evaluated: {n} (num_envs={args.num_envs}, seed={args.seed})")
    print()
    print(f"Success rate (ever reached target):     {len(successes) / n:.1%}")
    print(f"  clean success (reached, no collision): {len(clean_success) / n:.1%}")
    print(
        f"  reached but later collided:            {len(success_then_collision) / n:.1%}"
    )
    print(f"Collision rate (any episode):            {len(collisions) / n:.1%}")
    print(
        f"  collided without ever reaching target:  {len(collision_before_success) / n:.1%}"
    )
    print(
        f"Failure, no collision (timed out short):  {len(failure_no_collision) / n:.1%}"
    )
    print()
    print("Collision breakdown (of all episodes):")
    for name in _COLLISION_TERMS:
        rate = sum(1 for e in episodes if e["termination"] == name) / n
        print(f"  {name}: {rate:.1%}")
    print()
    print("Final position error (m):")
    print(f"  median: {np.median(position_errors):.4f}")
    print(f"  p90:    {np.percentile(position_errors, 90):.4f}")
    print(f"  max:    {position_errors.max():.4f}")
    if len(times_to_success) > 0:
        print()
        print("Time-to-target, successful episodes only (s):")
        print(f"  median: {np.median(times_to_success):.3f}")
        print(f"  p90:    {np.percentile(times_to_success, 90):.3f}")


if __name__ == "__main__":
    main()
