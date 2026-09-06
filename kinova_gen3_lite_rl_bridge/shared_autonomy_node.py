#!/usr/bin/env python3
"""
Blend a human joystick input with belief-weighted RL-policy suggestions.

Sibling to policy_bridge_node.py, not a replacement — that node stays the
validated single-target REACH tool. This node holds a set of candidate
goals (e.g. the tea-table objects) instead of one target, subscribes to
raw `/joy` directly (it must own publishing itself — the joystick-driven
`twist_controller` and this node's `joint_trajectory_controller` output
can't both be active in ros2_control at once, so this node cannot run
alongside gen3-lite-ros2's joy_teleop.py), and blends each goal's policy
action with the human's own input via goal_belief.py before publishing.

Same three staged-rollout modes as policy_bridge_node, for the same
safety reasons:

    observe_only     validates joint-state/TF wiring, logs, never calls
                      either backend, never publishes.
    inference_only    also computes per-goal deltas and updates goal
                      belief, logs (goal_deltas, probs, blended_delta),
                      never publishes.
    live              publishes the safety-clamped blended command.

Each goal's suggested joint delta can come from either of two
interchangeable backends, selected by the `action_source` parameter:
`policy` (the trained RL policy, via policy_client.py — unmodified from
Milestone 2) or `potential_field` (an analytic Jacobian-based attractor,
via potential_field_client.py — Milestone 3, for when the policy is
undertrained/doesn't generalize from arbitrary starting configurations).
Both produce the same shape of per-goal joint delta, so everything
downstream — GoalBelief, blend_action, SafetyClamp — is identical either
way.

The human's own jog input similarly comes from either of two
interchangeable sources, selected by `jog_mode`: `joint` (sticks map
straight to joints 1-4, via joy_joint_jog.py — unmodified from
Milestone 2) or `cartesian` (sticks map to X/Y/Z end-effector motion,
converted to a joint delta via the same Jacobian machinery as the
potential field — Milestone 4, for intuitive "move in 3D space" jogging
regardless of the arm's current configuration). `cartesian` needs the
potential field server for its Jacobian *even when action_source=policy*.
See the project plan's "Milestone 2" through "Milestone 4" sections for
the full rationale.
"""

from kinova_gen3_lite_rl_bridge.goal_belief import blend_action, GoalBelief
from kinova_gen3_lite_rl_bridge.joy_cartesian_jog import axes_to_cartesian_delta
from kinova_gen3_lite_rl_bridge.joy_joint_jog import axes_to_joint_delta
from kinova_gen3_lite_rl_bridge.observation_builder import (
    ACTION_DIM,
    ARM_JOINT_NAMES,
    build_observation,
    extract_arm_state,
)
from kinova_gen3_lite_rl_bridge.policy_client import PolicyClient
from kinova_gen3_lite_rl_bridge.potential_field_client import PotentialFieldClient
from kinova_gen3_lite_rl_bridge.safety_clamp import SafetyClamp, SafetyLimits
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


_VALID_MODES = ('observe_only', 'inference_only', 'live')
_VALID_ACTION_SOURCES = ('policy', 'potential_field')
_VALID_JOG_MODES = ('joint', 'cartesian')


class SharedAutonomyNode(Node):
    def __init__(self):
        super().__init__('shared_autonomy_node')

        self.declare_parameter('mode', 'observe_only')
        self.declare_parameter('action_source', 'potential_field')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'tool_frame')
        self.declare_parameter('goal_names', [''])
        self.declare_parameter('goal_targets_flat', [0.0])
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('policy_socket_path', '/tmp/kinova_rl_policy.sock')
        self.declare_parameter(
            'potential_field_socket_path', '/tmp/kinova_potential_field.sock'
        )
        self.declare_parameter(
            'joint_trajectory_topic', '/joint_trajectory_controller/joint_trajectory'
        )
        self.declare_parameter('temperature', 3.0)
        self.declare_parameter('jog_mode', 'cartesian')
        self.declare_parameter('jog_scale_rad_s', 0.3)
        self.declare_parameter('jog_scale_m_s', 0.15)
        self.declare_parameter('assist_step_rad', 0.01)
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
        self._action_source = self.get_parameter('action_source').value
        if self._action_source not in _VALID_ACTION_SOURCES:
            raise ValueError(
                f'action_source must be one of {_VALID_ACTION_SOURCES}, '
                f'got {self._action_source!r}'
            )
        self._jog_mode = self.get_parameter('jog_mode').value
        if self._jog_mode not in _VALID_JOG_MODES:
            raise ValueError(
                f'jog_mode must be one of {_VALID_JOG_MODES}, got {self._jog_mode!r}'
            )
        self._base_frame = self.get_parameter('base_frame').value
        self._ee_frame = self.get_parameter('ee_frame').value

        goal_names = list(self.get_parameter('goal_names').value)
        targets_flat = list(self.get_parameter('goal_targets_flat').value)
        if len(targets_flat) != 3 * len(goal_names):
            raise ValueError(
                f'goal_targets_flat has {len(targets_flat)} values, expected '
                f'3 * len(goal_names) = {3 * len(goal_names)}'
            )
        self._goal_names = goal_names
        self._goal_targets = {
            name: np.array(targets_flat[3 * i:3 * i + 3], dtype=np.float32)
            for i, name in enumerate(goal_names)
        }

        control_rate_hz = self.get_parameter('control_rate_hz').value
        self._control_dt = 1.0 / control_rate_hz
        self._joint_trajectory_topic = self.get_parameter('joint_trajectory_topic').value
        self._jog_scale_rad_s = self.get_parameter('jog_scale_rad_s').value
        self._jog_scale_m_s = self.get_parameter('jog_scale_m_s').value
        self._assist_step_rad = self.get_parameter('assist_step_rad').value

        temperature = self.get_parameter('temperature').value
        self._goal_belief = GoalBelief(self._goal_names, temperature=temperature)

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

        # Both clients are constructed regardless of action_source (cheap —
        # neither connects until first used) so a future runtime-switchable
        # action_source wouldn't need restructuring; only the active one is
        # readiness-checked and called.
        policy_socket_path = self.get_parameter('policy_socket_path').value
        self._policy_client = PolicyClient(policy_socket_path, logger=self.get_logger())
        field_socket_path = self.get_parameter('potential_field_socket_path').value
        self._potential_field_client = PotentialFieldClient(
            field_socket_path, logger=self.get_logger()
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_joint_state: JointState | None = None
        self._latest_joy: Joy | None = None
        self._last_action = {
            name: np.zeros(ACTION_DIM, dtype=np.float32) for name in self._goal_names
        }

        self._traj_pub = self.create_publisher(
            JointTrajectory, self._joint_trajectory_topic, 10
        )
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_subscription(Joy, '/joy', self._on_joy, 10)
        self.create_timer(self._control_dt, self._control_cycle)

        # potential_field is needed whenever it's the action_source *or*
        # jog_mode needs its Jacobian for Cartesian jog conversion — even
        # under action_source=policy.
        self._needed_backends = {self._action_source}
        if self._jog_mode == 'cartesian':
            self._needed_backends.add('potential_field')
        self._backend_ready = {name: False for name in self._needed_backends}

        if self._mode != 'observe_only':
            self._backend_check_timer = self.create_timer(1.0, self._check_backends_ready)

        self.get_logger().info(
            f'shared_autonomy_node started: mode={self._mode}, '
            f'action_source={self._action_source}, jog_mode={self._jog_mode}, '
            f'goals={self._goal_names}, rate={control_rate_hz}Hz'
        )

    # ── Backend server readiness (PING/PONG handshake) ───────────────────
    def _check_backends_ready(self):
        for name in self._needed_backends:
            if self._backend_ready[name]:
                continue
            client = self._policy_client if name == 'policy' else self._potential_field_client
            if client.ping():
                self._backend_ready[name] = True
                self.get_logger().info(f'{name} server ready (PING/PONG ok)')

        if all(self._backend_ready.values()):
            self._backend_check_timer.cancel()
        else:
            still_waiting = [n for n in self._needed_backends if not self._backend_ready[n]]
            self.get_logger().warn(
                f'waiting for {still_waiting} server(s)...', throttle_duration_sec=3.0
            )

    # ── State caching ─────────────────────────────────────────────────────
    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    def _on_joy(self, msg: Joy):
        self._latest_joy = msg

    # ── Main control loop ────────────────────────────────────────────────
    def _control_cycle(self):
        if self._latest_joint_state is None or self._latest_joy is None:
            missing = [
                name
                for name, value in (
                    ('/joint_states', self._latest_joint_state),
                    ('/joy', self._latest_joy),
                )
                if value is None
            ]
            self.get_logger().warn(
                f'waiting for {missing} before the first control cycle can run...',
                throttle_duration_sec=3.0,
            )
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

        if self._mode == 'observe_only':
            self._log_observe_only(joint_pos, joint_vel, ee_pos_base)
            return

        if not all(self._backend_ready.values()):
            return

        goal_deltas = self._compute_goal_deltas(joint_pos, joint_vel, ee_pos_base)
        if goal_deltas is None:
            return  # already logged by _compute_goal_deltas

        human_delta = self._compute_human_delta(joint_pos)
        if human_delta is None:
            return  # already logged by _compute_human_delta

        self._goal_belief.update(human_delta, goal_deltas)
        probs = self._goal_belief.probabilities()
        blended_delta = blend_action(
            human_delta, goal_deltas, probs, self._goal_names, self._assist_step_rad
        )

        if self._mode == 'inference_only':
            self.get_logger().info(
                f'probs={dict(zip(self._goal_names, probs.tolist()))} '
                f'blended_delta={blended_delta.tolist()}',
                throttle_duration_sec=1.0,
            )
            return

        # mode == 'live'
        blended_target_q = joint_pos + blended_delta
        clamped_q = self._safety_clamp.clamp(blended_target_q, joint_pos)
        self._publish_trajectory(clamped_q)

    def _log_observe_only(self, joint_pos, joint_vel, ee_pos_base):
        if self._action_source == 'policy':
            observations = {
                name: build_observation(
                    joint_pos, joint_vel, target, ee_pos_base, self._last_action[name]
                )
                for name, target in self._goal_targets.items()
            }
            obs_summary = {name: obs.tolist() for name, obs in observations.items()}
            self.get_logger().info(f'observations={obs_summary}', throttle_duration_sec=1.0)
        else:
            goal_targets_summary = {n: t.tolist() for n, t in self._goal_targets.items()}
            self.get_logger().info(
                f'joint_pos={joint_pos.tolist()} ee_pos_base={ee_pos_base.tolist()} '
                f'goal_targets={goal_targets_summary}',
                throttle_duration_sec=1.0,
            )

    def _compute_goal_deltas(self, joint_pos, joint_vel, ee_pos_base):
        """Return {goal_name: joint_delta}, or None (and already logged) on failure."""
        if self._action_source == 'policy':
            deltas = {}
            for name, target in self._goal_targets.items():
                obs = build_observation(
                    joint_pos, joint_vel, target, ee_pos_base, self._last_action[name]
                )
                raw_action = self._policy_client.infer(obs)
                if raw_action is None:
                    self.get_logger().warn(
                        f'policy inference failed for goal {name!r}, skipping cycle',
                        throttle_duration_sec=3.0,
                    )
                    return None
                self._last_action[name] = raw_action
                deltas[name] = raw_action - joint_pos
            return deltas

        deltas = {}
        for name, target in self._goal_targets.items():
            delta = self._potential_field_client.compute_delta(joint_pos, target)
            if delta is None:
                self.get_logger().warn(
                    f'potential field computation failed for goal {name!r}, skipping cycle',
                    throttle_duration_sec=3.0,
                )
                return None
            deltas[name] = delta
        return deltas

    def _compute_human_delta(self, joint_pos):
        """Return this cycle's human joint-space delta, or None (already logged) on failure."""
        if self._jog_mode == 'joint':
            return axes_to_joint_delta(
                self._latest_joy.axes, self._jog_scale_rad_s, self._control_dt
            )

        cartesian_delta = axes_to_cartesian_delta(
            self._latest_joy.axes, self._jog_scale_m_s, self._control_dt
        )
        human_delta = self._potential_field_client.compute_jog_delta(joint_pos, cartesian_delta)
        if human_delta is None:
            self.get_logger().warn(
                'cartesian jog conversion failed, skipping cycle', throttle_duration_sec=3.0
            )
        return human_delta

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
    node = SharedAutonomyNode()
    try:
        rclpy.spin(node)
    finally:
        node._policy_client.close()
        node._potential_field_client.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
