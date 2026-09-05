"""
Print every tea-table object's pregrasp target, for one env, as JSON.

Like print_pregrasp_target.py but resolves all of OBJECT_IDS in one call
-- used by shared_autonomy.launch.py to build its candidate goal set
(kettle/mug/infuser all at once), since shared autonomy needs every goal's
target, not just one.

Usage:
    uv run python scripts/print_all_pregrasp_targets.py --env-id env_a
"""

import argparse
import json

from kinova_mjlab_reaching.tasks.tea_table.grasp_registry import compute_pregrasp_pose
from kinova_mjlab_reaching.tasks.tea_table.object_pose_registry import ENV_IDS, OBJECT_IDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', choices=ENV_IDS, required=True)
    args = parser.parse_args()

    targets = {}
    for object_id in OBJECT_IDS:
        pos, _quat_wxyz = compute_pregrasp_pose(args.env_id, object_id)
        targets[object_id] = list(pos)
    print(json.dumps(targets))


if __name__ == '__main__':
    main()
