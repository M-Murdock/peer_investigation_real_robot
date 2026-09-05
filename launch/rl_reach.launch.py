r"""
Brings up the RL reaching policy bridge for the real Kinova Gen3 Lite.

Launches both halves of the pipeline together:
  - policy_bridge_node (this package, ROS 2 / Python 3.12): all ROS I/O.
  - policy_server.py (the kinova_mjlab_reaching uv project, Python 3.13):
    loads the trained checkpoint and serves inference over a local socket.
    Runs via `uv run` from the source checkout — uv operates on the
    checkout, not a colcon install space, so rl_project_dir must point at
    the actual source directory on this machine.

This does NOT bring up kortex_bringup itself (joint_trajectory_controller,
robot_state_publisher, etc.) — bring that up separately first, per the
project's staged-rollout verification plan.

Usage:
    ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \\
        mode:=observe_only start_policy_server:=false
    ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \\
        mode:=inference_only target_xyz:="[0.35, 0.05, 0.28]"
    ros2 launch kinova_gen3_lite_rl_bridge rl_reach.launch.py \\
        mode:=live target_xyz:="[0.35, 0.05, 0.28]"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

_DEFAULT_RL_PROJECT_DIR = '/home/mavis/ros2_ws/src/peer_investigation_real_robot'
_DEFAULT_CHECKPOINT_REL = 'checkpoints/2026-08-27_12-52-48/model_9999.pt'


def _launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context)
    target_xyz = yaml.safe_load(LaunchConfiguration('target_xyz').perform(context))
    rl_project_dir = LaunchConfiguration('rl_project_dir').perform(context)
    checkpoint_rel = LaunchConfiguration('checkpoint_rel').perform(context)
    uv_executable = LaunchConfiguration('uv_executable').perform(context)
    policy_device = LaunchConfiguration('policy_device').perform(context)
    socket_path = LaunchConfiguration('socket_path').perform(context)
    control_rate_hz = float(LaunchConfiguration('control_rate_hz').perform(context))
    base_frame = LaunchConfiguration('base_frame').perform(context)
    ee_frame = LaunchConfiguration('ee_frame').perform(context)
    joint_trajectory_topic = LaunchConfiguration('joint_trajectory_topic').perform(context)

    policy_bridge = Node(
        package='kinova_gen3_lite_rl_bridge',
        executable='policy_bridge_node',
        name='policy_bridge_node',
        output='screen',
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare('kinova_gen3_lite_rl_bridge'), 'config', 'safety_limits.yaml']
            ),
            {
                'mode': mode,
                'base_frame': base_frame,
                'ee_frame': ee_frame,
                'target_xyz': [float(v) for v in target_xyz],
                'control_rate_hz': control_rate_hz,
                'policy_socket_path': socket_path,
                'joint_trajectory_topic': joint_trajectory_topic,
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

    return [policy_bridge, policy_server]


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument('mode', default_value='observe_only'),
        DeclareLaunchArgument('target_xyz', default_value='[0.3, 0.0, 0.3]'),
        DeclareLaunchArgument('rl_project_dir', default_value=_DEFAULT_RL_PROJECT_DIR),
        DeclareLaunchArgument('checkpoint_rel', default_value=_DEFAULT_CHECKPOINT_REL),
        DeclareLaunchArgument('uv_executable', default_value='uv'),
        DeclareLaunchArgument('policy_device', default_value='cpu'),
        DeclareLaunchArgument('socket_path', default_value='/tmp/kinova_rl_policy.sock'),
        DeclareLaunchArgument('control_rate_hz', default_value='10.0'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument('ee_frame', default_value='tool_frame'),
        DeclareLaunchArgument(
            'joint_trajectory_topic',
            default_value='/joint_trajectory_controller/joint_trajectory',
        ),
        DeclareLaunchArgument('start_policy_server', default_value='true'),
    ]
    return LaunchDescription(declared_args + [OpaqueFunction(function=_launch_setup)])
