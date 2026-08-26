"""Reaching-with-avoidance environment: reach a randomly sampled target
position on the tea table while avoiding the static teapot/cup obstacles.

Action space: JointPositionActionCfg with use_default_offset=True, i.e.
target = action * scale + default_joint_pos (default_joint_pos = 0, the home
keyframe) — a fixed-anchor absolute position target, not raw torque.

Runbook section 15 describes the target scheme as "q_target = q_current +
Δq" and initially this was implemented with RelativeJointPositionActionCfg,
which computes target = *live measured* current_pos + action every step
(mjlab.envs.mdp.actions.actions.RelativeJointPositionAction.apply_actions).
The zero-action sanity check (runbook section 26) caught that this cannot
hold a position against gravity: with action=0 it continuously re-targets to
wherever the joint has already sagged to between control steps, so it locks
in drift instead of correcting it — confirmed by comparing against a fixed
ctrl target, which holds the arm still (<0.015 rad drift over 0.5s, matching
the M1 actuator validation). JointPositionActionCfg's fixed default-offset
anchor does not have this failure mode and is what mjlab's own reference
tasks (e.g. YAM lift-cube) use.

Collision avoidance: a hard termination on any contact between the robot's
arm subtree and each obstacle *and the table itself* (see
mdp/terminations.py), not a reward penalty — kept out of the reward per
runbook section 24 ("do not add many reward terms before the baseline is
understood"); distance-based avoidance shaping is left for the robustness
stage (section 29) if the baseline needs it. Table contact was added after
an early training run showed the arm wedging its elbow against the table
with no explicit signal to route around it — see docs/validation/ppo_baseline.md.

No custom reset events are configured — the default `reset_scene_to_default`
event (see mjlab.envs.mdp.events) already resets every entity (robot, table,
obstacles) to the EntityCfg-configured init_state, including env_origins
offsetting, which is all this task needs since the obstacles are static.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from kinova_mjlab_reaching.robots.gen3_lite import get_gen3_lite_robot_cfg
from kinova_mjlab_reaching.tasks.reaching.mdp.commands import ReachingCommandCfg
from kinova_mjlab_reaching.tasks.reaching.mdp.observations import ee_to_target_distance
from kinova_mjlab_reaching.tasks.reaching.mdp.rewards import (
    reaching_distance_reward,
    target_reached_bonus,
)
from kinova_mjlab_reaching.tasks.reaching.mdp.terminations import illegal_contact
from kinova_mjlab_reaching.tasks.reaching.scene import get_scene_object_cfgs

_ARM_JOINTS_CFG = SceneEntityCfg("robot", joint_names=("joint_[1-6]",))
_EE_SITE_CFG = SceneEntityCfg("robot", site_names=("ee_site",))

_OBSTACLE_NAMES = ("teapot", "cup_1", "cup_2")
_TABLE_NAME = "table"
# Table contact terminates the episode too, not just the loose obstacles —
# added after observing training runs where the arm wedged its elbow
# against the table and could never reach the goal from there. Physically
# the table always blocked motion (real MuJoCo contact dynamics), but
# without this termination that failure mode had no explicit learning
# signal beyond "the goal is unreachable from here" (2026-08-25).
_COLLISION_NAMES = _OBSTACLE_NAMES + (_TABLE_NAME,)
_COMMAND_NAME = "reach_target"
_SUCCESS_THRESHOLD = 0.03  # meters, runbook section 24.


def _collision_contact_sensor_cfg(body_name: str) -> ContactSensorCfg:
    """Contact sensor between the arm subtree and a single named entity.

    Works for both the discrete obstacles and the table — each is its own
    top-level scene entity whose single body shares the entity's name.
    """
    return ContactSensorCfg(
        name=f"{body_name}_collision",
        primary=ContactMatch(mode="subtree", pattern="shoulder_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern=body_name, entity=body_name),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


def get_reaching_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ARM_JOINTS_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ARM_JOINTS_CFG},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "ee_to_target": ObservationTermCfg(
            func=ee_to_target_distance,
            params={"command_name": _COMMAND_NAME, "asset_cfg": _EE_SITE_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=1.0,
            use_default_offset=True,
            # TODO: unvalidated placeholder — tune once the PPO baseline
            # (runbook section 27) is running. 1.0 rad covers ~40-60% of
            # each joint's half-range around the home (zero) pose.
        )
    }

    commands: dict[str, CommandTermCfg] = {
        _COMMAND_NAME: ReachingCommandCfg(
            resampling_time_range=(4.0, 4.0),
            debug_vis=True,
            success_threshold=_SUCCESS_THRESHOLD,
        )
    }

    rewards = {
        "reaching": RewardTermCfg(
            func=reaching_distance_reward,
            weight=1.0,
            params={
                "command_name": _COMMAND_NAME,
                "std": 0.2,
                "asset_cfg": _EE_SITE_CFG,
            },
        ),
        "target_reached": RewardTermCfg(
            func=target_reached_bonus,
            weight=10.0,
            params={
                "command_name": _COMMAND_NAME,
                "success_threshold": _SUCCESS_THRESHOLD,
                "asset_cfg": _EE_SITE_CFG,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={"asset_cfg": _ARM_JOINTS_CFG},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        **{
            f"{name}_collision": TerminationTermCfg(
                func=illegal_contact, params={"sensor_name": f"{name}_collision"}
            )
            for name in _COLLISION_NAMES
        },
    }

    return ManagerBasedRlEnvCfg(
        decimation=4,
        scene=SceneCfg(
            num_envs=num_envs,
            entities={"robot": get_gen3_lite_robot_cfg(), **get_scene_object_cfgs()},
            sensors=tuple(_collision_contact_sensor_cfg(n) for n in _COLLISION_NAMES),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        episode_length_s=4.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="shoulder_link",
            distance=1.2,
            elevation=-20.0,
            azimuth=90.0,
        ),
    )
