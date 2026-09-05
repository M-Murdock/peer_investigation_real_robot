#!/usr/bin/env python3
"""
Bridge a real Kinova Gen3 Lite arm to the RL reaching policy.

Served by scripts/policy_server.py (see ipc_protocol.py for the wire
format, and the project plan for the full staged-rollout rationale).

Three operator-selectable modes (runtime-changeable via `ros2 param set`),
matching the project runbook's staged rollout (do not skip stages when
moving toward the real robot):

    observe_only     builds observations, logs them, never calls the
                      policy server, never publishes — validates
                      joint-state/TF wiring in isolation.
    inference_only    also calls the policy server and logs
                      (obs, raw_action, would-be-clamped-command) every
                      cycle, but never publishes a robot command.
    live              publishes the safety-clamped command.

Single node by design (not split into goal_manager/policy/safety nodes):
this milestone is REACH-only with one fixed target, so a multi-node split
would add IPC/lifecycle overhead for no present benefit. The internals
(observation_builder, safety_clamp, policy_client) have no rclpy
dependency and are reusable unchanged by a future multi-goal
(shared-autonomy) controller.
"""

from kinova_gen3_lite_rl_bridge.observation_builder import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    build_observation,
    extract_arm_state,
)
from kinova_gen3_lite_rl_bridge.policy_client import PolicyClient
from kinova_gen3_lite_rl_bridge.safety_clamp import SafetyClamp, SafetyLimits
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_VALID_MODES = ('observe_only', 'inference_only', 'live')

# Training-time initial-state randomization perturbed joints by a small
# uniform offset around the zero/home pose (see reach_env_cfg.py's
# reset_robot_joints) — the policy's behavior starting further from that
# distribution is untested. This is a warn-not-block sanity check.
_TRAINED_JOINT_BOX_RAD = 0.2


class PolicyBridgeNode(Node):
    def __init__(self):
        super().__init__('policy_bridge_node')

        self.declare_parameter('mode', 'observe_only')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'tool_frame')
        self.declare_parameter('target_xyz', [0.3, 0.0, 0.3])
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('policy_socket_path', '/tmp/kinova_rl_policy.sock')
        self.declare_parameter(
            'joint_trajectory_topic', '/joint_trajectory_controller/joint_trajectory'
        )
        self.declare_parameter(
            'safety.max_step_rad', [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        )
        self.declare_parameter(
            'safety.max_velocity_rad_s', [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        )
        self.declare_parameter(
            'safety.joint_lower_rad', [-2.68, -2.61, -2.61, -2.60, -2.53, -2.60]
        )
        self.declare_parameter(
            'safety.joint_upper_rad', [2.68, 2.61, 2.61, 2.60, 2.53, 2.60]
        )

        self._mode = self.get_parameter('mode').value
        self._base_frame = self.get_parameter('base_frame').value
        self._ee_frame = self.get_parameter('ee_frame').value
        self._target_xyz = np.array(self.get_parameter('target_xyz').value, dtype=np.float32)
        control_rate_hz = self.get_parameter('control_rate_hz').value
        self._control_dt = 1.0 / control_rate_hz
        self._joint_trajectory_topic = self.get_parameter('joint_trajectory_topic').value

        self._safety_clamp = SafetyClamp(
            SafetyLimits(
                max_step_rad=np.array(self.get_parameter('safety.max_step_rad').value),
                max_velocity_rad_s=np.array(
                    self.get_parameter('safety.max_velocity_rad_s').value
                ),
                joint_lower_rad=np.array(self.get_parameter('safety.joint_lower_rad').value),
                joint_upper_rad=np.array(self.get_parameter('safety.joint_upper_rad').value),
            ),
            control_dt=self._control_dt,
        )

        self.add_on_set_parameters_callback(self._on_set_parameters)

        socket_path = self.get_parameter('policy_socket_path').value
        self._policy_client = PolicyClient(socket_path, logger=self.get_logger())

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_joint_state: JointState | None = None
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._seen_first_live_cycle = False

        self._traj_pub = self.create_publisher(
            JointTrajectory, self._joint_trajectory_topic, 10
        )
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_timer(self._control_dt, self._control_cycle)

        if self._mode != 'observe_only':
            self._policy_ready = False
            self._policy_check_timer = self.create_timer(1.0, self._check_policy_ready)
        else:
            self._policy_ready = False

        self.get_logger().info(
            f'policy_bridge_node started: mode={self._mode}, '
            f'target_xyz={self._target_xyz.tolist()}, rate={control_rate_hz}Hz'
        )

    # ── Parameter validation ─────────────────────────────────────────────
    def _on_set_parameters(self, params):
        from rcl_interfaces.msg import SetParametersResult

        for param in params:
            if param.name == 'mode' and param.value not in _VALID_MODES:
                return SetParametersResult(
                    successful=False, reason=f'mode must be one of {_VALID_MODES}'
                )
            if param.name == 'target_xyz' and len(param.value) != 3:
                return SetParametersResult(
                    successful=False, reason='target_xyz must have 3 elements'
                )

        for param in params:
            if param.name == 'mode':
                self._mode = param.value
                self.get_logger().info(f'mode changed to {self._mode}')
            elif param.name == 'target_xyz':
                self._target_xyz = np.array(param.value, dtype=np.float32)
                self.get_logger().info(f'target_xyz changed to {self._target_xyz.tolist()}')

        return SetParametersResult(successful=True)

    # ── Policy server readiness (PING/PONG handshake) ────────────────────
    def _check_policy_ready(self):
        if self._policy_client.ping():
            self._policy_ready = True
            self._policy_check_timer.cancel()
            self.get_logger().info('policy server ready (PING/PONG ok)')
        else:
            self.get_logger().warn('waiting for policy server...', throttle_duration_sec=3.0)

    # ── State caching ─────────────────────────────────────────────────────
    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    # ── Main control loop ────────────────────────────────────────────────
    def _control_cycle(self):
        if self._latest_joint_state is None:
            return

        try:
            joint_pos, joint_vel = extract_arm_state(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
                self._latest_joint_state.velocity,
            )
        except KeyError as exc:
            self.get_logger().warn(
                f'joint state missing arm joint: {exc}', throttle_duration_sec=3.0
            )
            return

        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._ee_frame, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self.get_logger().warn(f'TF lookup failed: {exc}', throttle_duration_sec=3.0)
            return

        ee_pos_base = np.array(
            [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        )

        obs = build_observation(
            joint_pos, joint_vel, self._target_xyz, ee_pos_base, self._last_action
        )

        if self._mode == 'observe_only':
            self.get_logger().info(f'obs={obs.tolist()}', throttle_duration_sec=1.0)
            return

        if not self._policy_ready:
            return

        raw_action = self._policy_client.infer(obs)
        if raw_action is None:
            self.get_logger().warn(
                'policy inference failed, skipping cycle', throttle_duration_sec=3.0
            )
            return
        self._last_action = raw_action

        # Clamp relative to the arm's actual, currently-measured position —
        # not a previously-commanded target — so the commanded step is
        # bounded by real velocity regardless of any controller tracking
        # lag. Clamping against an internally-tracked "previous target"
        # instead would let the commanded reference run ahead of the real
        # arm if it can't fully reach each step in time: every cycle would
        # then add another "small" step on top of an increasingly
        # fictional position, silently compounding into a large real
        # jump — this is what caused the shake-and-fault seen in testing.
        clamped_q = self._safety_clamp.clamp(raw_action, joint_pos)

        if self._mode == 'inference_only':
            self.get_logger().info(
                f'raw_action={raw_action.tolist()} clamped={clamped_q.tolist()}',
                throttle_duration_sec=1.0,
            )
            return

        # mode == 'live'
        if np.any(np.abs(joint_pos) > _TRAINED_JOINT_BOX_RAD) and not self._seen_first_live_cycle:
            self.get_logger().warn(
                f'starting joint_pos {joint_pos.tolist()} is outside the '
                f'trained ±{_TRAINED_JOINT_BOX_RAD} rad box around home — '
                'policy behavior here is untested'
            )
        self._seen_first_live_cycle = True

        self._publish_trajectory(clamped_q)

    def _publish_trajectory(self, target_q: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = list(ARM_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = target_q.tolist()
        point.time_from_start = Duration(seconds=1.5 * self._control_dt).to_msg()
        msg.points = [point]
        self._traj_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node._policy_client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
