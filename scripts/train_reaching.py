"""
Train the reaching PPO baseline (runbook section 27).

Usage:
    uv run python scripts/train_reaching.py                        # default run
    uv run python scripts/train_reaching.py --max-iterations 20    # smoke run
    uv run python scripts/train_reaching.py --num-envs 64 --max-iterations 20
"""

import argparse

import kinova_mjlab_reaching.tasks.reaching  # noqa: F401  (registers the task)
from kinova_mjlab_reaching.tasks.reaching import TASK_ID
from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from mjlab.scripts.train import TrainConfig, launch_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()

    env_cfg = get_reaching_env_cfg(num_envs=args.num_envs)
    agent_cfg = get_reaching_ppo_cfg(max_iterations=args.max_iterations)
    train_cfg = TrainConfig(env=env_cfg, agent=agent_cfg, log_root="")

    launch_training(TASK_ID, train_cfg)


if __name__ == "__main__":
    main()
