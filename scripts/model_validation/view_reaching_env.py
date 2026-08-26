"""
Interactive viewer for the reaching env, driven by a dummy (zero or random)
policy — runbook section 26's "uv run play <task> --agent zero/random", done
directly against reach_env_cfg since the task isn't registered in mjlab's
task registry yet.

Unlike scripts/model_validation/view_reaching_scene.py (which only shows the
static geometry), this drives the actual ManagerBasedRlEnv, so the sampled
reach target is visible as a debug-vis sphere and episodes reset on
timeout/collision like they will during training.

Usage:
    uv run python scripts/model_validation/view_reaching_env.py zero
    uv run python scripts/model_validation/view_reaching_env.py random
"""

import sys

import torch

from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer


class PolicyZero:
    def __init__(self, action_shape: tuple[int, ...], device: str) -> None:
        self._action_shape = action_shape
        self._device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        del obs
        return torch.zeros(self._action_shape, device=self._device)


class PolicyRandom:
    def __init__(self, action_shape: tuple[int, ...], device: str) -> None:
        self._action_shape = action_shape
        self._device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        del obs
        return 2 * torch.rand(self._action_shape, device=self._device) - 1


def main() -> None:
    agent = sys.argv[1] if len(sys.argv) > 1 else "zero"
    if agent not in ("zero", "random"):
        raise SystemExit(f"Unknown agent '{agent}' — use 'zero' or 'random'")

    cfg = get_reaching_env_cfg()
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    action_shape = env.unwrapped.action_space.shape
    policy = (
        PolicyZero(action_shape, env.unwrapped.device)
        if agent == "zero"
        else PolicyRandom(action_shape, env.unwrapped.device)
    )

    print(f"[INFO] Launching viewer with '{agent}' policy — close the window to exit.")
    viewer = NativeMujocoViewer(env, policy)
    viewer.run()


if __name__ == "__main__":
    main()
