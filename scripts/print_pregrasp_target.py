"""
Print a tea-table object's real-world pregrasp target as JSON.

Resolves config/tea_table_objects.yaml + config/tea_table_grasps.yaml via
grasp_registry.compute_pregrasp_pose() into a single [x, y, z] position in
the robot base frame — the same frame rl_reach.launch.py's `target_xyz`
expects. Meant to be called by that launch file (via `uv run`) when it's
given `object_id`/`env_id` instead of a raw `target_xyz`, but also usable
standalone:

Usage:
    uv run python scripts/print_pregrasp_target.py --env-id env_a --object-id kettle
"""

import argparse
import json

from kinova_mjlab_reaching.tasks.tea_table.grasp_registry import compute_pregrasp_pose
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import OBJECT_IDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', choices=('env_a', 'env_b'), required=True)
    parser.add_argument('--object-id', choices=OBJECT_IDS, required=True)
    args = parser.parse_args()

    pos, _quat_wxyz = compute_pregrasp_pose(args.env_id, args.object_id)
    print(json.dumps({'target_xyz': list(pos)}))


if __name__ == '__main__':
    main()
