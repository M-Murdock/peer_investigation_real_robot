"""Shared checkpoint-loading path for real-hardware deployment.

Extracted from scripts/evaluate_reaching.py's env/runner setup so that
policy_server.py (and evaluate_reaching.py itself) load a checkpoint
exactly one way. Do not hand-reimplement the actor's forward pass
elsewhere — get_inference_policy() already returns the deterministic
(no exploration noise) policy, including the obs_normalization layer
loaded from the checkpoint, and that combination is what must run
unchanged on the real robot.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


def load_reaching_policy(
    checkpoint: Path, device: str = 'cpu'
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Loads a trained reaching-task checkpoint and returns its inference policy.

    The returned callable takes raw, unnormalized observations shaped
    (batch, 21) and returns actions shaped (batch, 6) — the actor's own
    obs_normalization handles standardization internally.

    num_envs=1 only bootstraps observation/action shapes for the runner;
    this environment is never stepped (no MuJoCo simulation runs on the
    real-hardware deployment path).
    """
    env_cfg = get_reaching_env_cfg(num_envs=1)
    raw_env = ManagerBasedRlEnv(env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=None)

    agent_cfg = asdict(get_reaching_ppo_cfg())
    runner = MjlabOnPolicyRunner(env, agent_cfg, str(checkpoint.parent), device)
    runner.load(
        str(checkpoint),
        load_cfg={'actor': True},
        strict=True,
        map_location=device,
    )
    return runner.get_inference_policy(device=device)
