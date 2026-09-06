r"""
Bring up the shared-autonomy blend of joystick teleop + RL policy belief.

Resolves a whole env's tea-table objects (kettle/mug/infuser) into a
candidate goal set via scripts/print_all_pregrasp_targets.py (uv side),
then launches shared_autonomy_node alongside the same policy_server.py
process rl_reach.launch.py uses — one shared server instance, since every
goal queries the same checkpoint with a different target_xyz.

Requires a joystick already publishing /joy (e.g. `ros2 run joy joy_node`)
and joint_trajectory_controller active. Do NOT also run
gen3-lite-ros2/joy_teleop.py — it drives twist_controller, which cannot be
active at the same time as joint_trajectory_controller, and this node
reads /joy directly itself.

Each goal's suggested joint delta can come from either of two
interchangeable backends (shared_autonomy_node's `action_source`
parameter): `potential_field` (default — an analytic Jacobian-based
attractor via scripts/potential_field_server.py, no trained weights, no
checkpoint-loading wait) or `policy` (the trained RL policy via
scripts/policy_server.py, unmodified from Milestone 2). Only the server
matching the active `action_source` needs to actually run —
`start_policy_server`/`start_potential_field_server` default to match
`action_source`'s own default, override both explicitly if you change it.

Usage:
    ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \\
        mode:=observe_only start_potential_field_server:=false
    ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \\
        mode:=inference_only env_id:=env_a
    ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \\
        mode:=live env_id:=env_a

    # To use the trained RL policy instead of the potential field:
    ros2 launch kinova_gen3_lite_rl_bridge shared_autonomy.launch.py \\
        mode:=live env_id:=env_a action_source:=policy \\
        start_policy_server:=true start_potential_field_server:=false
"""

import json
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

_DEFAULT_RL_PROJECT_DIR = '/home/mavis/ros2_ws/src/peer_investigation_real_robot'
_DEFAULT_CHECKPOINT_REL = 'checkpoints/2026-08-27_12-52-48/model_9999.pt'


def _resolve_goal_set(uv_executable: str, rl_project_dir: str, env_id: str):
    """
    Run scripts/print_all_pregrasp_targets.py under `uv run` and return goals.

    Returns (goal_names, goal_targets_flat) — the grasp registry lives in
    the uv (mjlab/scipy) environment, not this launch process's plain
    system Python.
    """
    result = subprocess.run(
        [
            uv_executable, 'run', 'python', 'scripts/print_all_pregrasp_targets.py',
            '--env-id', env_id,
        ],
        cwd=rl_project_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'print_all_pregrasp_targets.py failed for env_id={env_id!r}:\n{result.stderr}'
        )
    targets = json.loads(result.stdout)
    goal_names = list(targets.keys())
    goal_targets_flat = [float(v) for name in goal_names for v in targets[name]]
    return goal_names, goal_targets_flat


def _launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)
    action_source = LaunchConfiguration('action_source').perform(context)
    env_id = LaunchConfiguration('env_id').perform(context)
    rl_project_dir = LaunchConfiguration('rl_project_dir').perform(context)
    checkpoint_rel = LaunchConfiguration('checkpoint_rel').perform(context)
    uv_executable = LaunchConfiguration('uv_executable').perform(context)
    policy_device = LaunchConfiguration('policy_device').perform(context)
    socket_path = LaunchConfiguration('socket_path').perform(context)
    field_socket_path = LaunchConfiguration('field_socket_path').perform(context)
    control_rate_hz = float(LaunchConfiguration('control_rate_hz').perform(context))
    base_frame = LaunchConfiguration('base_frame').perform(context)
    ee_frame = LaunchConfiguration('ee_frame').perform(context)
    joint_trajectory_topic = LaunchConfiguration('joint_trajectory_topic').perform(context)
    temperature = float(LaunchConfiguration('temperature').perform(context))
    jog_scale_rad_s = float(LaunchConfiguration('jog_scale_rad_s').perform(context))
    assist_step_rad = float(LaunchConfiguration('assist_step_rad').perform(context))
    k_p = LaunchConfiguration('k_p').perform(context)
    damping_lambda = LaunchConfiguration('damping_lambda').perform(context)

    goal_names, goal_targets_flat = _resolve_goal_set(uv_executable, rl_project_dir, env_id)

    shared_autonomy = Node(
        package='kinova_gen3_lite_rl_bridge',
        executable='shared_autonomy_node',
        name='shared_autonomy_node',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('kinova_gen3_lite_rl_bridge'), 'config', 'shared_autonomy.yaml']
            ),
            {
                'mode': mode,
                'action_source': action_source,
                'base_frame': base_frame,
                'ee_frame': ee_frame,
                'goal_names': goal_names,
                'goal_targets_flat': goal_targets_flat,
                'control_rate_hz': control_rate_hz,
                'policy_socket_path': socket_path,
                'potential_field_socket_path': field_socket_path,
                'joint_trajectory_topic': joint_trajectory_topic,
                'temperature': temperature,
                'jog_scale_rad_s': jog_scale_rad_s,
                'assist_step_rad': assist_step_rad,
            },
        ],
    )

    policy_server = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('start_policy_server')),
        cmd=[
            uv_executable,
            'run',
            'python',
            'scripts/policy_server.py',
            '--checkpoint',
            PathJoinSubstitution([rl_project_dir, checkpoint_rel]),
            '--socket-path',
            socket_path,
            '--device',
            policy_device,
        ],
        cwd=rl_project_dir,
        output='screen',
        respawn=False,
    )

    potential_field_server = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('start_potential_field_server')),
        cmd=[
            uv_executable,
            'run',
            'python',
            'scripts/potential_field_server.py',
            '--socket-path',
            field_socket_path,
            '--k-p',
            k_p,
            '--damping-lambda',
            damping_lambda,
        ],
        cwd=rl_project_dir,
        output='screen',
        respawn=False,
    )

    return [shared_autonomy, policy_server, potential_field_server]


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('mode', default_value='observe_only'),
        DeclareLaunchArgument(
            'action_source', default_value='potential_field',
            description='Per-goal delta source: potential_field (default, analytic, no '
                        'trained weights) or policy (the trained RL policy).',
        ),
        DeclareLaunchArgument('env_id', default_value='env_a'),
        DeclareLaunchArgument('rl_project_dir', default_value=_DEFAULT_RL_PROJECT_DIR),
        DeclareLaunchArgument('checkpoint_rel', default_value=_DEFAULT_CHECKPOINT_REL),
        DeclareLaunchArgument('uv_executable', default_value='uv'),
        DeclareLaunchArgument('policy_device', default_value='cpu'),
        DeclareLaunchArgument('socket_path', default_value='/tmp/kinova_rl_policy.sock'),
        DeclareLaunchArgument(
            'field_socket_path', default_value='/tmp/kinova_potential_field.sock'
        ),
        DeclareLaunchArgument('control_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('ee_frame', default_value='tool_frame'),
        DeclareLaunchArgument(
            'joint_trajectory_topic',
            default_value='/joint_trajectory_controller/joint_trajectory',
        ),
        DeclareLaunchArgument('temperature', default_value='3.0'),
        DeclareLaunchArgument('jog_scale_rad_s', default_value='0.3'),
        DeclareLaunchArgument(
            'assist_step_rad', default_value='0.01',
            description='Per-cycle magnitude each goal direction is normalized to before '
                        'belief-weighting and summing with the human jog input, keeping the '
                        'human meaningfully influential regardless of a goal deltas own raw '
                        'magnitude (distance-to-goal for the potential field, or an '
                        'undertrained policys uncalibrated action scale).',
        ),
        DeclareLaunchArgument('k_p', default_value='2.0'),
        DeclareLaunchArgument('damping_lambda', default_value='0.05'),
        # Defaults match action_source's own default (potential_field) —
        # override both together if you switch action_source:=policy.
        DeclareLaunchArgument('start_policy_server', default_value='false'),
        DeclareLaunchArgument('start_potential_field_server', default_value='true'),
    ]
    return LaunchDescription(declared_args + [OpaqueFunction(function=_launch_setup)])
