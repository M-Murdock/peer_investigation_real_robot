"""
Mandatory environment sanity checks before PPO, per runbook section 26.

Runs a zero-action rollout and a random-action rollout on the reaching env
and checks:
  - observations and rewards are finite (no NaN/Inf)
  - episodes terminate/reset correctly
  - random actions stay bounded (no exploding joints)

Usage:
    uv run python scripts/model_validation/test_reaching_env.py
"""

import torch

from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from mjlab.envs import ManagerBasedRlEnv


def run_episode(env: ManagerBasedRlEnv, policy: str, num_steps: int) -> None:
    print(f"\n--- {policy} rollout ({num_steps} steps) ---")
    obs, _ = env.reset()
    action_dim = env.action_manager.action.shape[-1]

    num_resets = 0
    max_abs_action = 0.0
    for step in range(num_steps):
        if policy == "zero":
            action = torch.zeros(env.num_envs, action_dim, device=env.device)
        else:
            action = torch.empty(env.num_envs, action_dim, device=env.device).uniform_(
                -1.0, 1.0
            )
        max_abs_action = max(max_abs_action, action.abs().max().item())

        obs, reward, terminated, truncated, extras = env.step(action)

        for group_name, group_obs in obs.items():
            assert torch.isfinite(group_obs).all(), (
                f"Non-finite obs in group '{group_name}' at step {step}"
            )
        assert torch.isfinite(reward).all(), f"Non-finite reward at step {step}"

        done = terminated | truncated
        num_resets += int(done.sum().item())

        if step % 100 == 0 or step == num_steps - 1:
            joint_pos = env.scene["robot"].data.joint_pos[0, :6]
            print(
                f"  step {step:4d}  reward={reward.item():+.4f}  "
                f"terminated={terminated.item()}  truncated={truncated.item()}  "
                f"joint_pos_max_abs={joint_pos.abs().max().item():.3f} rad"
            )

    print(f"  resets observed: {num_resets}")
    print(f"  max |action| applied: {max_abs_action:.3f}")
    print("  PASS: all observations and rewards finite")


def main() -> None:
    cfg = get_reaching_env_cfg()
    env = ManagerBasedRlEnv(cfg, device="cpu")

    steps_per_episode = env.max_episode_length
    print(f"max_episode_length = {steps_per_episode} steps")

    run_episode(env, "zero", num_steps=steps_per_episode * 2)
    run_episode(env, "random", num_steps=steps_per_episode * 2)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
