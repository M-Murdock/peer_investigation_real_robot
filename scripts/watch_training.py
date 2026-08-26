"""
Watch a currently-running (or finished) training run live in the browser via
mjlab's Viser viewer, hot-swapping to newer checkpoints as they're saved.

This mirrors the TRAINED_MODE + viser path in mjlab.scripts.play, scoped
directly to the reaching task so it doesn't depend on mjlab's own task
registry import (kinova_mjlab_reaching.tasks.reaching registers itself).

Usage:
    uv run python scripts/watch_training.py
    uv run python scripts/watch_training.py --checkpoint-dir checkpoints/2026-08-25_22-35-31
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
from mjlab.viewer.viser import ViserPlayViewer

from kinova_mjlab_reaching.tasks.reaching.agents.ppo_cfg import get_reaching_ppo_cfg
from kinova_mjlab_reaching.tasks.reaching.reach_env_cfg import get_reaching_env_cfg

CHECKPOINTS_ROOT = Path("checkpoints")


def latest_checkpoint_dir() -> Path:
    run_dirs = [d for d in CHECKPOINTS_ROOT.iterdir() if d.is_dir()]
    if not run_dirs:
        raise SystemExit(f"No training runs found under {CHECKPOINTS_ROOT}")
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def wait_for_first_checkpoint(ckpt_dir: Path, timeout_s: float = 120.0) -> Path:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        checkpoints = sorted(ckpt_dir.glob("model_*.pt"))
        if checkpoints:
            return checkpoints[0]
        time.sleep(1.0)
    raise SystemExit(f"No checkpoint appeared in {ckpt_dir} within {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Defaults to the most recently modified run under checkpoints/",
    )
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoint_dir or latest_checkpoint_dir()
    print(f"[INFO] Watching checkpoint directory: {ckpt_dir}")
    first_checkpoint = wait_for_first_checkpoint(ckpt_dir)

    env_cfg = get_reaching_env_cfg(num_envs=1)
    env = ManagerBasedRlEnv(env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg = asdict(get_reaching_ppo_cfg())
    runner = MjlabOnPolicyRunner(env, agent_cfg, str(ckpt_dir), device="cpu")

    def load_and_get_policy(path: str):
        runner.load(path, load_cfg={"actor": True}, strict=True, map_location="cpu")
        return runner.get_inference_policy(device="cpu")

    policy = load_and_get_policy(str(first_checkpoint))

    def fetch_available() -> list[tuple[str, str]]:
        now = time.time()
        entries = []
        for f in sorted(ckpt_dir.glob("model_*.pt")):
            try:
                step = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                step = 0
            ago = format_time_ago(int(now - f.stat().st_mtime))
            entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

    ckpt_manager = CheckpointManager(
        current_name=first_checkpoint.name,
        fetch_available=fetch_available,
        load_checkpoint=lambda name: load_and_get_policy(str(ckpt_dir / name)),
    )

    print("[INFO] Launching Viser viewer — open the printed URL in your browser.")
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()


if __name__ == "__main__":
    main()
