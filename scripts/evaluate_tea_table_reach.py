"""
Six-goal pre-grasp REACH test (runbook v2 section 26.2): run the existing
trained reaching checkpoint against the tea-table scene's real pre-grasp
targets, one environment at a time. Stops after REACH - no ALIGN/DESCEND/
CLOSE yet (those don't exist).

Not the same as evaluate_reaching.py: that script lets ReachingCommand
sample random targets. Here the target for every object is FIXED (the
computed pregrasp_xyz from grasp_registry.py), so after every reset -
including the auto-resets ReachingCommand performs mid-rollout - the
command's target_pos is overwritten back to the fixed value. This is
poking at ReachingCommand's public tensor state directly rather than
adding a "fixed target" mode to the command itself, since nothing else
needs that mode yet.

Usage:
    uv run python scripts/evaluate_tea_table_reach.py --env-id env_a --checkpoint checkpoints/2026-08-27_12-52-48/model_9999.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.mdp.commands import ReachingCommand
from kinova_mjlab_reaching.tasks.tea_table.env_cfg import get_tea_table_env_cfg
from kinova_mjlab_reaching.tasks.tea_table.grasp_registry import compute_pregrasp_pose
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import OBJECT_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

_COLLISION_TERMS = tuple(f"{name}_collision" for name in OBJECT_IDS) + (
    "table_collision",
    "arm_base_fixture_collision",
)


def evaluate_one_object(
    raw_env: ManagerBasedRlEnv,
    env,
    policy,
    command: ReachingCommand,
    fixed_target_local: torch.Tensor,
    episodes_per_env: int,
    device: str,
) -> list[dict]:
    """Runs `episodes_per_env` episodes per parallel env, all pinned to the
    same fixed pregrasp target, and classifies each episode exactly like
    evaluate_reaching.py does."""
    num_envs = env.num_envs
    success_threshold = command.cfg.success_threshold
    step_dt = raw_env.step_dt
    max_episode_length = raw_env.max_episode_length
    num_steps = max_episode_length * episodes_per_env

    target_w = fixed_target_local.unsqueeze(0).expand(num_envs, 3) + (
        raw_env.scene.env_origins
    )

    ever_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    steps_to_success = torch.zeros(num_envs, dtype=torch.long, device=device)
    step_in_episode = torch.zeros(num_envs, dtype=torch.long, device=device)
    episodes: list[dict] = []

    # Same reasoning as evaluate_reaching.py: command.metrics["position_error"]
    # is recomputed against a freshly-resampled target the instant an env
    # auto-resets, so it's contaminated for that step - snapshot one step
    # earlier instead. Here the "fresh target" a reset produces is
    # immediately overwritten back to target_w below, so this mainly
    # matters for the transient value read before that overwrite happens.
    prev_pos_err = torch.full((num_envs,), float("nan"), device=device)

    # reset() and the whole step loop must share one inference_mode context:
    # command.episode_success gets reassigned (torch.maximum(...)) inside
    # the step loop, and if that happens under inference_mode while a LATER
    # reset() (for the next object) runs outside it, PyTorch refuses the
    # next in-place write to it ("Inplace update to inference tensor
    # outside InferenceMode"). Callers wrap repeated calls to this function
    # in one outer inference_mode block for the same reason.
    obs, _ = env.reset()
    command.target_pos[:] = target_w
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
            done_idx = torch.as_tensor(done_ids, device=device)
            ever_succeeded[done_idx] = False
            steps_to_success[done_idx] = 0
            step_in_episode[done_idx] = 0
            # Undo ReachingCommand's auto-resample for the envs that
            # just reset - pin them back to the fixed target.
            command.target_pos[done_idx] = target_w[done_idx]

    return episodes


def summarize(object_id: str, episodes: list[dict]) -> None:
    n = len(episodes)
    if n == 0:
        print(f"{object_id}: no episodes completed - increase --episodes-per-env")
        return

    successes = [e for e in episodes if e["success"]]
    collisions = [e for e in episodes if e["termination"] != "time_out"]
    # Same breakdown as evaluate_reaching.py: "success" and "collision" are
    # not mutually exclusive within one episode - a run keeps executing for
    # the full fixed duration after first success (see runbook v2 section
    # 23A.1), so an episode can reach the target fine and only collide
    # later while continuing to move with nothing left to do.
    success_then_collision = [e for e in successes if e["termination"] != "time_out"]
    collision_before_success = [e for e in collisions if not e["success"]]
    position_errors = np.array(
        [e["position_error"] for e in episodes if not np.isnan(e["position_error"])]
    )
    times = np.array(
        [
            e["time_to_success_s"]
            for e in successes
            if e["time_to_success_s"] is not None
        ]
    )

    print(f"\n=== {object_id} ({n} episodes) ===")
    print(f"Success rate (ever reached pregrasp_xyz):        {len(successes) / n:.1%}")
    # This is the metric that actually reflects REACH's fitness for the
    # deployed state machine: episodes here keep running the RL policy for
    # the full fixed duration after first success (nothing hands off to
    # ALIGN, unlike the real grasp_skill_node), so "any collision in the
    # padded episode" conflates real REACH failures with harmless
    # post-success wandering (runbook v2 section 23A.1) - sometimes almost
    # entirely so. collision-before-ever-reaching is what the deployed
    # REACH->ALIGN handoff would actually be exposed to.
    print(f"Collision BEFORE ever reaching (the metric that matters for "
          f"deployment): {len(collision_before_success) / n:.1%}")
    print(f"Collision rate, any (misleading on its own - see above): "
          f"{len(collisions) / n:.1%}")
    if successes:
        print(f"  of which reached-then-collided-later "
              f"(expected to disappear once ALIGN hands off immediately "
              f"on success): {len(success_then_collision) / len(successes):.1%} of successes")
    if collisions:
        for name in _COLLISION_TERMS:
            rate = sum(1 for e in episodes if e["termination"] == name) / n
            if rate > 0:
                print(f"  {name}: {rate:.1%}")
    if len(position_errors) > 0:
        print(
            f"Final position error (m): median {np.median(position_errors):.4f} "
            f"p90 {np.percentile(position_errors, 90):.4f} "
            f"max {position_errors.max():.4f}"
        )
    if len(times) > 0:
        print(
            f"Time-to-target (s), successes only: median {np.median(times):.3f} "
            f"p90 {np.percentile(times, 90):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", choices=("env_a", "env_b"), default="env_a")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--episodes-per-env",
        type=int,
        default=8,
        help="Rollout length is this many full episode durations, per object.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env_cfg = get_tea_table_env_cfg(args.env_id, num_envs=args.num_envs)
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

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env_id} (num_envs={args.num_envs}, seed={args.seed})")

    with torch.inference_mode():
        for object_id in OBJECT_IDS:
            pregrasp_pos, _ = compute_pregrasp_pose(args.env_id, object_id)
            target_local = torch.tensor(
                pregrasp_pos, dtype=torch.float32, device=args.device
            )
            episodes = evaluate_one_object(
                raw_env, env, policy, command, target_local, args.episodes_per_env,
                args.device,
            )
            summarize(object_id, episodes)


if __name__ == "__main__":
    main()
